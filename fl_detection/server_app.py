"""
このファイルの役割（Server側）：

- Federated Learning の全体制御
- グローバルモデルの初期化
- クライアントへの学習設定配布（train_config）
- FedAvg による集約
- 各ラウンド後のグローバル評価
- 最終モデル保存

※ 攻撃設定（attack_type 等）やデータ分割の設定はここからクライアントへ配布される。
※ 固定画像による可視化ログ出力は，現在使っていないため unused_features/fixed_eval.py へ移してある。
"""

# ============================================================
# 標準ライブラリ
# ============================================================
import os
from datetime import datetime

# ============================================================
# Flower / PyTorch 関連
# ============================================================
# 【import 文の見方】
#   import X            : ライブラリ X を丸ごと取り込む。使うときは X.関数名 と書く。
#   from A.B import C   : 「A フォルダの中の B.py から C を取り込む」。使うときは C だけで書ける。
#                         ドット（.）はフォルダ区切り（/）に相当し，拡張子 .py は書かない。
#                         例: fl_detection.log_output -> fl_detection/log_output.py
#   from A import (C, D): 取り込むものが多いときに，カッコで複数行に分けて書いたもの。
#   連合学習の実行中に動くコードは，すべて fl_detection フォルダの中にある。
#   そのため，このフォルダの中のファイルどうしの import は，どれも from fl_detection.〜 import ... の形になる。
#
# 【import がすること】
#   import は「その関数・クラスを名前で呼べる状態にする」だけで，この時点では何も実行されない。
#   （道具箱から道具を取り出して机に置くだけ。使うのは後で 関数名(...) と書いた行。）
#   そのため，import の横のコメントは「その関数が呼ばれたときに何をするか」の説明であり，
#   import の行で CSV への書き込みなどが起きるわけではない。
#   例: log_results_csv
#       - import（下の from ... import）   : 名前を使えるようにする。まだ何も書き込まれない。
#       - configure_log_dir(...) の呼び出し : 書き込み先フォルダを決める（main() 内）。
#       - log_results_csv(...) の呼び出し   : 実際に1行追記する（global_evaluate() 内，毎ラウンド1回）。
import torch
from flwr.app import ArrayRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

from fl_detection.log_output import (
    configure_log_dir,      # ログ（CSV等）の出力先フォルダを切り替える関数（デフォルトは logs/）
    print_round_summary,    # ラウンドごとの結果を画面（標準出力）に表示する関数
    log_run_config,         # その実行で使われた設定の全項目を run_config.txt に保存する関数
    log_results_csv,        # ラウンドごとの Accuracy/Loss を実験条件つきで results.csv に追記する関数
    log_summary_csv,        # 最終ラウンドの Accuracy/Loss を1実行1行で summary.csv に追記する関数
)
from fl_detection.task import Net, load_centralized_dataset, resolve_device, test
from fl_detection.robust_strategy import RobustFedAvg
from fl_detection.malicious import generate_malicious_ids

# サーバアプリ本体を作る。Flower は pyproject.toml の serverapp = "fl_detection.server_app:app" を見てこの app を探し，下の @app.main() を付けた main() をサーバ側の入口として実行する。
app = ServerApp()

# ============================================================
# メイン処理（FL全体制御）
#
# 毎ラウンドの config 配布・攻撃の差し込み・検知ログ・集約の実体は
# fl_detection/robust_strategy.py の RobustFedAvg にある。
# ============================================================
# 【この部分の説明】
#   flwr run で実行されるサーバ側の本体。ここから下は「今の実験で毎回動く」部分。
#   main() 自身は計算をほとんどせず，次の順で全体を指揮する。
#       1. 実験設定を読む
#       2. 出力先フォルダとログの保存先を決める
#       3. 悪性クライアントを決める
#       4. 設定を画面に表示し，ファイルにも記録する
#       5. 乱数の種を固定する
#       6. グローバルモデルを初期化する
#       7. 集約戦略（RobustFedAvg）を組み立てる
#       8. 毎ラウンドの評価のしかた（global_evaluate）を定義する
#       9. 連合学習を開始する（ここで全ラウンドが回る）
#      10. 最終モデルと最終結果を保存する
#
#   @app.main()
#       デコレータ（関数の上に @ で付ける目印）。「すぐ下の関数をサーバの入口として app に登録する」。
#   def main(grid: Grid, context: Context) -> None:
#       grid    : クライアント群との通信窓口。Flower が渡してくる。
#       context : 実行時の設定が入ったもの。Flower が渡してくる。
#       -> None : 「この関数は値を返さない」という型ヒント。
@app.main()
def main(grid: Grid, context: Context) -> None:
    # ===================== 実験設定読み込み（pyproject.toml） =====================
    # context.run_config は「設定名 -> 値」の対応表（辞書）。pyproject.toml の [tool.flwr.app.config] の値が入り，
    # run_experiments.py が --run-config で渡した値があれば，そちらで上書きされる。
    #   run_config["名前"]            : その名前の値を取り出す。名前が無いとエラーになる。
    #                                   設定値は pyproject.toml だけに書く方針なので，このファイルには既定値を書かない
    #                                   （pyproject.toml に項目を書き忘れると，別の値で黙って動く代わりにエラーで止まる）。
    #   float(...) / int(...) / str(...) / bool(...)
    #                                 : 値を 小数 / 整数 / 文字列 / 真偽値 にそろえる（型変換）。
    #   左辺（= の左）は，取り出した値に付ける名前（変数）。以降はこの名前で使う。
    # local_epochs と batch_size は，ここでは表示・記録に使うだけ
    # （実際の学習ではクライアント側が同じ設定を自分で読む）。
    fraction_evaluate = context.run_config["fraction-evaluate"]
    num_rounds = context.run_config["num-server-rounds"]
    lr = context.run_config["learning-rate"]
    local_epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]
    aggregation_method = context.run_config["aggregation-method"]   # 集約手法（現在は "fedavg" だけ）

    # ---- 追加：データ分割 / 攻撃 設定 ----
    data_split = context.run_config["data-split"]                   # データ分割方式（現在は "dirichlet" だけ）
    alpha = float(context.run_config["dirichlet-alpha"])            # Dirichlet の濃度（小さいほど偏りが強い）
    attack_type = context.run_config["attack-type"]                 # "none" / "sign_flip" / "alie"
    alie_z = float(context.run_config["alie-z"])                    # ALIE攻撃（sotsuron_plan2.tex 4.3 A2）の摂動強度 z
    trial = int(context.run_config["trial"])                        # Fang et al. (2020)の"trial"に対応：再現性用シード兼繰り返し試行番号
    device = resolve_device(context.run_config["device"])           # 計算に使う装置（"auto" なら GPU があれば GPU，なければ CPU）

    # 使えない値は，学習を始める前にここで止める（途中まで回ってから気づくのを防ぐ）。
    #   値 not in (候補, ...) : 値が候補のどれでもなければ True。
    #   raise ValueError(...) : エラーを起こして実行を止める。
    if attack_type not in ("none", "sign_flip", "alie"):
        raise ValueError(
            f"未対応の attack-type です: {attack_type!r}（対応: 'none' / 'sign_flip' / 'alie'）。"
            "'scale' / 'adaptive' は unused_features/attack_adaptive.py に移してあります。"
        )

    # ---- 追加：実験計画書7.6：実験ごとに出力先を分けるためのディレクトリ ----
    # run_experiments.py が実験条件ごとに別ディレクトリを渡すことで、
    # final_model.pt が実験間で上書きされないようにする。
    #   output_dir = ...      : 出力先フォルダの名前を設定から読む。pyproject.toml の値は "."（今いるフォルダ）。
    #   os.makedirs(...)      : そのフォルダが無ければ作る。
    output_dir =str(context.run_config["output-dir"])
    os.makedirs(output_dir, exist_ok=True)

    # ---- 追加：results.csv/summary.csv/検知ログ の出力先を output_dir に紐付ける ----
    # log_output.py のデフォルト（logs/、プロジェクト直下固定）のままだと、
    # run_experiments.py で条件ごとに output_dir を分けても集計ログは全条件・全実行で
    # 1つのファイルに混ざってしまい、あとから解析する際に条件の切り分けができない。
    # output_dir が指定されている（"." 以外の）場合は、そのスイープの親ディレクトリ
    # （例: experiments_detection_v2/<条件>/ -> experiments_detection_v2/_logs/）に集約する。
    #   if output_dir not in (".", ""): : output_dir が "." でも空でもないとき（＝出力先が指定されたとき）だけ実行。
    #   os.path.normpath(...)           : パスの書き方を整える（末尾の / を取るなど）。
    #   os.path.dirname(...)            : 1つ上のフォルダ名を取り出す。
    #                                     例: experiments_detection_v2/条件名 -> experiments_detection_v2
    #   ... or "."                      : 取り出した結果が空だったら，代わりに "." を使う。
    #   configure_log_dir(...)          : ログの保存先を <1つ上のフォルダ>/_logs に切り替える。
    if output_dir not in (".", ""):
        sweep_root = os.path.dirname(os.path.normpath(output_dir)) or "."
        configure_log_dir(os.path.join(sweep_root, "_logs"))

    # ---- 追加：悪性クライアント生成（実験計画書 7.3 対応） ----
    # num-clients / malicious-ratio / malicious-seed から，悪性にするクライアントを選ぶ。
    #   下の3行                     : クライアント総数・悪性の割合・乱数の種（seed）を設定から読む。
    #   generate_malicious_ids(...) : malicious.py の関数を呼び，悪性クライアントの番号のリストを受け取る。
    #                                 引数を 名前=値 の形で渡している（キーワード引数。順番を間違えにくい）。
    #                                 例: 20台・割合0.2・seed 1 なら [8, 9, 14, 19] の4台。
    #                                 seed が同じなら毎回同じ番号が選ばれる（再現性のため）。
    #                                 台数 × 割合 が整数にならない設定は，ここでエラーになって止まる。
    num_clients = int(context.run_config["num-clients"])
    malicious_ratio = float(context.run_config["malicious-ratio"])
    malicious_seed = int(context.run_config["malicious-seed"])
    malicious_ids = generate_malicious_ids(
        num_clients=num_clients,
        malicious_ratio=malicious_ratio,
        seed=malicious_seed,
    )

    # 攻撃なし（attack-type="none"）のときは，悪性クライアントを0人として扱う。
    # こうしないと，pyproject.toml の malicious-ratio が 0 でないまま攻撃なしで回したときに，
    # 何も攻撃していないクライアントが「悪性」としてログに記録されてしまう。
    #   [] は空のリスト（＝悪性クライアント無し）。
    if attack_type == "none":
        malicious_ratio = 0.0
        malicious_ids = []

    # 表示（確認用）
    # print(...) は画面（ターミナル）に文字を出すだけで，学習には影響しない。
    # カンマで区切ると，間に空白を入れて続けて表示される。
    # f"...{変数}..." は変数の値を埋め込む書き方。
    print("=== Experiment Configuration ===")
    print("rounds        :", num_rounds)
    print("fraction_eval :", fraction_evaluate)
    print("local_epochs  :", local_epochs)
    print("batch_size    :", batch_size)
    print("learning_rate :", lr)
    print("optimizer     : SGD")
    print("device        :", device, f"(設定: {context.run_config['device']})")
    print("data_split    :", data_split, f"(alpha={alpha})")
    print("attack_type   :", attack_type, f"(alie_z={alie_z})" if attack_type == "alie" else "")
    print("num_clients   :", num_clients)
    print("malicious_ratio:", malicious_ratio, f"(seed={malicious_seed})")
    print("malicious_ids :", malicious_ids)
    print("aggregation   :", aggregation_method)
    print("trial         :", trial)
    print("================================")

    # ---- 追加：実験計画書7.5：results.csv / summary.csv 用の実行識別子 ----
    # この1回の実行を見分けるための名前（run_id）を作る。CSV の各行に付けて，どの実行の行かを区別する。
    #   attack_label                 : 攻撃の名前。ALIE のときだけ z の値を後ろに付ける（例: alie_z0.5）。
    #   ( f"..." f"..." )            : カッコの中に文字列を並べると，1つにつながる（長い行を2行に分けているだけ）。
    #   datetime.now().strftime(...) : 現在の日時を 年月日-時分秒 の文字列にする。
    #   例: fedavg_dirichlet_a0.5_alie_z0.5_ratio0.2_trial1_20261009-081959
    attack_label = f"{attack_type}_z{alie_z}" if attack_type == "alie" else attack_type
    run_id = (
        f"{aggregation_method}_{data_split}_a{alpha}_{attack_label}_ratio{malicious_ratio}"
        f"_trial{trial}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )

    # ログの各行に付ける実験条件（log_output.py の _CONDITION_COLUMNS と同じ項目）。
    # 全条件のログが1つのファイルに追記されるので，この列で条件を切り分ける。
    #   { "名前": 値, ... } は辞書。None は「値なし」（CSV では空欄になる）。
    condition = {
        "aggregation_method": aggregation_method,
        "data_split": data_split,
        "alpha": alpha,
        "attack_type": attack_type,
        "alie_z": alie_z if attack_type == "alie" else None,
        "malicious_ratio": malicious_ratio,
        "trial": trial,
    }

    # この実行で実際に使われた設定を，全項目そのまま run_config.txt に書き残す。
    #   dict(context.run_config) : pyproject.toml の値に --run-config の上書きを反映した，設定の全項目。
    #   {**辞書, "名前": 値}      : 辞書の中身をすべて写したうえで，項目を足した新しい辞書を作る書き方。
    log_run_config(
        run_id,
        {
            **dict(context.run_config),
            "malicious-ratio (effective)": malicious_ratio,
            "malicious-ids (effective)": malicious_ids,
            "device (effective)": str(device),
        },
    )

    # 最終ラウンドの結果を summary.csv に書くために覚えておく入れ物
    #   last_eval = {} : 空の辞書（名前と値の対応表）。最終ラウンドの結果を後で入れる。
    last_eval = {}

    # ===================== 乱数の種の固定 =====================
    # モデルの初期値（最初の重み）は乱数で決まる。種を決めておかないと，同じ条件で回し直しても
    # 初期値が変わり，結果が一致しなくなる。trial 番号を種にするので，
    #   同じ trial なら毎回同じ初期値，trial が違えば別の初期値
    # になる（データ分割・悪性クライアントの選出と同じく，trial に連動する）。
    torch.manual_seed(trial)

    # ===================== グローバルモデル初期化 =====================
    # 学習前の（ランダムな重みの）モデルを1つ作り，Flower で配れる形に包む。
    #   Net()                       : task.py のモデル（CNN）を1つ作る。Net はクラス（設計図）で，() で実体を作る。
    #   global_model.state_dict()   : モデルの全ての重みを「層の名前 -> Tensor」の辞書として取り出す。
    #   ArrayRecord(...)            : その重みを Flower の入れ物に入れる。これが最初にクライアントへ配られる。
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # ===================== 集約戦略（毎ラウンド config 配布 + 攻撃の差し込み + 検知ログ + 集約） =====================
    # 「毎ラウンド何をするか」を担当する戦略オブジェクトを作る（robust_strategy.py の RobustFedAvg）。
    # ここまでに読んだ設定を 名前=値 の形でまとめて渡している。この時点ではまだ学習は始まらない。
    # 攻撃の差し込み・検知スコアの計算・集約は，すべてこの strategy の中で行われる
    # （このファイルに attack や detection が出てこないのはそのため）。
    strategy = RobustFedAvg(
        fraction_evaluate=fraction_evaluate,
        lr=lr,
        data_split=data_split,
        alpha=alpha,
        attack_type=attack_type,
        malicious_ids=malicious_ids,
        trial=trial,
        alie_z=alie_z,
        num_clients=num_clients,
        aggregation_method=aggregation_method,
        malicious_ratio=malicious_ratio,
        run_id=run_id,
    )

    # ===================== 各ラウンド後のグローバル評価 =====================
    # 実験計画書7.5：results.csv に条件つきで書き込むため、main() 内のクロージャにして
    # run_id / condition（実験条件）をそのまま参照できるようにしている。最終ラウンドの値は last_eval に残しておき、
    # strategy.start() 完了後に summary.csv へ1行だけ書き込むのに使う。
    #
    # 【この関数の説明】
    #   集約後のグローバルモデルを，サーバが持つテストデータで採点する関数。
    #   ここでは「定義している」だけで，実行されるのは下の strategy.start() の中（毎ラウンド1回）。
    #   関数の中に関数を書いているので，main() の変数（run_id，condition など）をそのまま使える。
    #     server_round : 何ラウンド目か（Flower が渡してくる）
    #     arrays       : そのラウンドの集約後の重み（Flower が渡してくる）
    #     -> MetricRecord : 精度と損失を MetricRecord に入れて返す，という型ヒント
    #
    #   モデルの準備（下の4行）
    #     model = Net()                        : 空のモデルを作る。
    #     model.load_state_dict(...)           : 受け取った重みをモデルに読み込ませる。
    #     model.to(device)                     : モデルを，計算に使う装置（main() の最初に決めた device）に載せる。
    #     model.eval()                         : 評価モードにする（学習時だけの挙動を止める）。
    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        model = Net()
        model.load_state_dict(arrays.to_torch_state_dict())

        model.to(device)
        model.eval()

        # ===================== 全体テスト評価 =====================
        # テストデータ全体（MNIST の 10,000 枚）でモデルを採点する。毎ラウンド必ず動く。
        #   load_centralized_dataset() : テストデータを少しずつ取り出せる形（DataLoader）で用意する。
        #   test(model, データ, 装置)  : task.py の関数。損失と精度の2つを返す。
        #   test_loss, test_acc = ...  : 返ってきた2つの値を，左から順に2つの変数で受け取る。
        test_dataloader = load_centralized_dataset()
        test_loss, test_acc = test(model, test_dataloader, device)

        # ===================== ログ出力 =====================
        # このラウンドの精度と損失を記録する。毎ラウンド必ず動く（import した関数を実際に呼んでいる場所）。
        #   print_round_summary(...)  : 画面に表示する。
        #   log_results_csv(...)      : 実験条件（condition）つきで results.csv に1行追記する。
        #   last_eval["round"] = ...  : 辞書に 名前 -> 値 を入れる。毎ラウンド上書きされるので，
        #                               最後には最終ラウンドの値が残る（下の summary.csv で使う）。
        #   return MetricRecord({...}): 精度と損失を Flower に返して，この関数を終える。
        print_round_summary(server_round, accuracy=test_acc, loss_func=test_loss)

        log_results_csv(
            run_id=run_id,
            round_=server_round,
            accuracy=test_acc,
            loss_func=test_loss,
            condition=condition,
        )
        last_eval["round"] = server_round
        last_eval["accuracy"] = test_acc
        last_eval["loss_func"] = test_loss

        return MetricRecord({"accuracy": test_acc, "loss_func": test_loss})

    # ===================== FL開始 =====================
    # この Flower 版では start() が train_config_fn を受け取らないので、
    # strategy 側（configure_train/evaluate）で毎ラウンド注入する。
    #
    # ここで実際に連合学習が始まり，全ラウンドが終わるまでこの1行から先へ進まない（実行時間のほとんどがここ）。
    #   grid=grid                   : クライアントとの通信窓口を渡す。
    #   initial_arrays=arrays       : 最初に配るモデルの重みを渡す。
    #   num_rounds=num_rounds       : 何ラウンド回すかを渡す。
    #   evaluate_fn=global_evaluate : 上で定義した評価関数を渡す。() を付けていないので，ここでは呼び出さず
    #                                 「関数そのもの」を渡している。Flower が毎ラウンド代わりに呼んでくれる。
    #   result = ...                : 終了後の結果（最終的な重みなど）を受け取る。
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    # ===================== 最終モデル保存 =====================
    # 全ラウンド終了後のモデルの重みを final_model.pt として保存する。
    #   model_path = ...                       : 保存先のファイル名（パス）を作る。
    #   result.arrays.to_torch_state_dict()    : 最終的な重みを PyTorch の形（辞書）に戻す。
    #   torch.save(state_dict, model_path)     : それをファイルに書き出す。
    # run_experiments.py はこのファイルがあるかどうかで「その条件は完了済みか」を判断している。
    model_path = os.path.join(output_dir, "final_model.pt")
    print(f"\nSaving final model to {model_path} ...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, model_path)

    # ===================== 実験計画書7.5：summary.csv（最終ラウンドの結果を1行） =====================
    # 最終ラウンドの精度と損失を，summary.csv に1行だけ追記する。
    #   if last_eval:        : 辞書が空でなければ（＝評価が1回でも行われていれば）実行する。
    #                          Python では空の辞書は False，中身があれば True として扱われる。
    #   last_eval["round"]   : 辞書から，その名前の値を取り出す。
    if last_eval:
        log_summary_csv(
            run_id=run_id,
            final_round=last_eval["round"],
            final_accuracy=last_eval["accuracy"],
            final_loss=last_eval["loss_func"],
            condition=condition,
        )