"""
このファイルの役割（Client側）：

- 連合学習に参加する「クライアント1台ぶん」の処理を書いたファイル。
- サーバからの依頼は2種類あり，それぞれに対応する関数がある。
    train()    : 「このモデルを手元のデータで学習して，結果を返して」という依頼への応答
    evaluate() : 「このモデルを手元のデータで採点して，結果を返して」という依頼への応答
- このファイルは20台ぶん別々に用意されているわけではなく，同じコードが
  クライアントの台数ぶん呼び出される。自分が何番のクライアントかは，
  実行時に context から受け取る（partition-id）。
- 符号反転攻撃（sign_flip）は，このファイルの train() の中で行われる。
  ALIE 攻撃はここではなくサーバ側（robust_strategy.py）で行われる。
"""

# 【import の説明】（import 文の読み方は server_app.py の冒頭を参照）
import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
#   ArrayRecord  : モデルの重みを入れる入れ物
#   Context      : 実行時の設定（pyproject.toml の値や，自分のクライアント番号）を持つもの
#   Message      : サーバとクライアントの間でやり取りする「手紙」
#   MetricRecord : 損失や精度などの数値を入れる入れ物
#   RecordDict   : 上の入れ物を 名前 -> 入れ物 の形でまとめる箱（Message の中身になる）
from flwr.clientapp import ClientApp  # クライアントアプリの本体クラス

from fl_detection.log_output import print_round_summary  # 画面表示用の関数
from fl_detection.task import Net, load_data, resolve_device  # モデルの定義，担当データの読み込み，計算に使う装置の決定
# 「as 別名」は，取り込んだものに別の名前を付ける書き方。
# このファイルでも train / evaluate という関数を定義するので，名前がぶつからないよう
# task.py の train / test を train_fn / test_fn という名前で取り込んでいる。
from fl_detection.task import test as test_fn    # モデルを採点して（損失, 精度）を返す関数
from fl_detection.task import train as train_fn  # モデルを学習させて平均の損失を返す関数
from fl_detection.attack import sign_flip_delta                 # 攻撃用の関数（更新の符号を反転する）


# クライアントアプリ
# クライアントアプリ本体を作る。Flower は pyproject.toml の
# clientapp = "fl_detection.client_app:app" を見てこの app を探す。
# 下の @app.train() / @app.evaluate() を付けた関数が，サーバからの依頼の受付窓口として登録される。
app = ClientApp()


# 【この部分の説明】
#   サーバから「学習して」という依頼が来るたびに実行される関数。
#   毎ラウンド，クライアント1台につき1回呼ばれる（20台なら1ラウンドに20回）。
#   流れは次のとおり。
#       0. サーバから届いた設定を取り出す
#       1. サーバから届いた重みでモデルを作る
#       2. 自分の担当データを読み込む
#       3. 手元のデータで学習する（ローカル学習）
#       4. 学習の結果を記録する
#       5. 学習後の重みを取り出す
#       6. 悪性クライアントなら，ここで更新を書き換える（攻撃）
#       7. 重みと数値をサーバへ返す
#
#   @app.train()
#       デコレータ。「すぐ下の関数を，学習の依頼の受付窓口として app に登録する」。
#   def train(msg: Message, context: Context):
#       msg     : サーバから届いた手紙。中に モデルの重み（"arrays"）と 設定（"config"）が入っている。
#       context : 実行時の設定。自分のクライアント番号や pyproject.toml の値を取り出せる。
#       どちらも Flower が渡してくる。
@app.train()
def train(msg: Message, context: Context):
    # ============================================================
    # 0) サーバから受け取った設定（config）を一度だけ取り出す
    #    ★修正：ConfigRecord/TypedDict 等でも安全に扱えるよう dict 化
    # ============================================================
    # この設定は，サーバ側の RobustFedAvg が毎ラウンド手紙に入れて配っているもの
    # （robust_strategy.py の _fill_common_config）。pyproject.toml を直接読んでいるわけではない。
    #   msg.content                 : 手紙の中身（名前 -> 入れ物 の対応表）。
    #   msg.content["config"]       : "config" という名前の中身（設定）を取り出す。
    #   dict(...)                   : それを普通の辞書に変換する。
    #   cfg["名前"]                  : 辞書から値を取り出す。サーバが毎回すべての項目を入れてくるので，
    #                                 予備の値は置かない（項目が欠けていたら，別の値で黙って動く代わりにエラーで止まる）。
    #   float(...) / int(...)       : 小数 / 整数 にそろえる。
    #   set(...)                    : 集合に変換する。「この番号は含まれているか」を素早く調べられる。
    cfg = dict(msg.content["config"])

    # --- データ分割用の設定 ---
    data_split = cfg["data_split"]
    alpha = float(cfg["alpha"])
    trial = int(cfg["trial"])  # Fang et al. (2020)の"trial"に対応：データ分割の再現性用シード兼繰り返し試行番号

    # --- 攻撃用設定 ---
    attack_type = cfg["attack_type"]
    malicious_ids = set(cfg["malicious_ids"])

    # --- round番号 ---
    server_round = cfg["server_round"]

    # ============================================================
    # 1) モデル初期化（サーバから受け取った重みをロード）
    #    ★攻撃を「更新量ベース」で行うため、学習前のグローバル重みを
    #      global_state_dict_cpu として退避しておく（Δ = local - global の計算に使う）。
    # ============================================================
    #   msg.content["arrays"].to_torch_state_dict()
    #       : 手紙から重みを取り出し，PyTorch の形（層の名前 -> Tensor の辞書）に直す。
    #   { k: ... for k, v in 辞書.items() }
    #       : 辞書内包表記。辞書の（名前 k, 値 v）を1組ずつ取り出し，新しい辞書を作る書き方。
    #   v.detach().cpu().clone()
    #       : 勾配の記録から切り離し，CPU 上に置き，コピーを作る。
    #         コピーにしておくことで，このあと学習でモデルが変わっても「学習前の重み」が残る。
    #   model = Net()                 : 空のモデルを作る。
    #   model.load_state_dict(...)    : サーバから届いた重みをモデルに読み込ませる。
    #   resolve_device(...)           : 設定（pyproject.toml の device）から，計算に使う装置（CPU か GPU）を決める。
    #   model.to(device)              : モデルをその装置に載せる。
    global_state_dict_cpu = {
        k: v.detach().cpu().clone()
        for k, v in msg.content["arrays"].to_torch_state_dict().items()
    }

    model = Net()
    model.load_state_dict(global_state_dict_cpu)

    device = resolve_device(context.run_config["device"])
    model.to(device)

    # ============================================================
    # 2) クライアントID / データ読み込み
    # ============================================================
    #   context.node_config : このクライアント固有の情報（Flower が用意する）。
    #       "partition-id"   : 自分の番号（0 から始まる）。20台なら 0〜19。
    #       "num-partitions" : クライアントの総数（データを何個に分けるか）。
    #   context.run_config  : pyproject.toml の [tool.flwr.app.config] の値（サーバと同じものが見える）。
    #   load_data(...)      : task.py の関数。全体の学習データを num_partitions 個に分け，
    #                         そのうち自分の番号（partition_id）のぶんだけを読み込む。
    #                         分け方（data_split, alpha, samples_per_client）と seed（trial）が全員同じなので，
    #                         各クライアントは互いに重ならない自分の担当ぶんを受け取る。
    #   samples_per_client  : クライアント1台に配る枚数（pyproject.toml の samples-per-client）。
    #                         全員が同じ枚数を持つので，サーバへ返す "num-examples"（FedAvg の重み）も全員同じになる。
    #   trainloader, _ = ... : load_data は（学習用, 評価用）の2つを返す。
    #                          _ は「受け取るが使わない」ことを示す慣習的な名前（ここでは評価用を捨てている）。
    #   partition_id in malicious_ids : 自分の番号が悪性の一覧に入っていれば True，なければ False。
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    samples_per_client = int(context.run_config["samples-per-client"])

    # データ分割の設定（data_split / alpha / trial / samples_per_client）を反映して読み込む（重要）
    trainloader, _ = load_data(
        partition_id,
        num_partitions,
        batch_size,
        data_split=data_split,
        alpha=alpha,
        trial=trial,
        samples_per_client=samples_per_client,
    )

    # 悪性クライアント判定
    is_malicious = partition_id in malicious_ids

    # ============================================================
    # 3) ローカル学習
    # ============================================================
    # ここが，クライアントが実際に学習する場所（task.py の train を train_fn という名前で呼んでいる）。
    #   渡しているもの : モデル，自分のデータ，手元のデータを何周するか（local-epochs），学習率，装置。
    #   中で起きること : 素朴な SGD でモデルの重みを更新する。model の中身が書き換わる。
    #   train_loss     : 学習中の損失の平均が返ってくる（小さいほどよく合っている）。
    #   cfg["lr"]      : 学習率。pyproject.toml の learning-rate を，サーバが毎ラウンド config に入れて配っている。
    #
    # 乱数の種を固定する（再現性のため）。
    #   学習では，手元のデータを毎回ランダムな順番に並べ替えて使う（シャッフル）。
    #   種を決めておかないと，同じ条件で回し直しても並び順が変わり，結果が一致しなくなる。
    #   trial・ラウンド番号・クライアント番号から種を作るので，
    #   「同じ条件の同じラウンドの同じクライアント」は毎回同じ並び順になり，
    #   ラウンドやクライアントが違えば別の並び順になる。
    #   % は割った余り（種として使える範囲の整数に収めている）。
    torch.manual_seed((trial * 1_000_003 + server_round * 1_009 + partition_id) % (2**31 - 1))
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        float(cfg["lr"]),
        device,
    )

    # ============================================================
    # 4) 画面表示（学習）
    # ============================================================
    #   print_round_summary(...)  : 画面に表示する（ファイルには記録しない）。
    print_round_summary(
        round_=server_round,
        train_loss=train_loss,
    )

    # ============================================================
    # 5) 学習後のモデル重みを取得（送信直前はCPUへ）
    # ============================================================
    #   model.state_dict() : 学習後のモデルの全ての重み（層の名前 -> Tensor の辞書）。
    #   1) で退避した global_state_dict_cpu が「学習前」，これが「学習後」。
    local_state_dict_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # ============================================================
    # 6) 攻撃（更新量ベース Δ = local − global）
    #    ここが重要：学習後・送信直前に、Δ = local - global に対して攻撃を適用し、
    #    global + Δ_malicious として送信する。
    #    ※ ALIE 攻撃は他クライアントの更新を要するためサーバ側
    #      （robust_strategy.py の aggregate_train）で集約直前に差し替える。
    #      ここ（クライアント側）では何もせず、honestな更新のまま送信する。
    # ============================================================
    #   delta = { k: 学習後[k] - 学習前[k] ... }
    #       : 層ごとに「学習でどれだけ変わったか」（更新量 Δ）を計算する。
    #         これが，サーバ側で検知の対象になる「更新ベクトル」のもと。
    #   if 条件A and 条件B:   : 両方が成り立つときだけ実行する。
    #       悪性 かつ sign_flip : Δ の符号を反転する（学習で進んだ向きと逆向きにする）。
    #       それ以外            : 何もしない。良性クライアントと，ALIE のときの悪性クライアントは
    #                             ここでは正直な更新のまま送り，悪性ぶんはサーバ側で差し替えられる。
    #   state_dict_to_send = { k: 学習前[k] + delta[k] ... }
    #       : 学習前の重みに（攻撃後の）Δ を足して，サーバへ送る重みを作る。
    #         攻撃しなかった場合は，学習後の重みと同じになる。
    delta = {k: local_state_dict_cpu[k] - global_state_dict_cpu[k] for k in local_state_dict_cpu}

    if is_malicious and attack_type == "sign_flip":
        delta = sign_flip_delta(delta)

    state_dict_to_send = {k: global_state_dict_cpu[k] + delta[k] for k in delta}

    # ============================================================
    # 7) サーバへ返送
    # ============================================================
    #   ArrayRecord(...)   : 送る重みを Flower の入れ物に入れる。
    #   metrics = {...}    : 一緒に送る数値をまとめた辞書。
    #       "num-examples" : 学習データの枚数。FedAvg が「データの多いクライアントほど重く」平均するのに使う。
    #       "partition-id" : 自分の番号。サーバが「どの更新が悪性クライアントのものか」を知るのに使う。
    #   MetricRecord(...)  : その辞書を Flower の入れ物に入れる。
    #   RecordDict({...})  : 重みと数値を，名前を付けて1つの箱にまとめる。
    #   Message(content=content, reply_to=msg)
    #                      : 届いた手紙（msg）への返信として，新しい手紙を作る。
    #   return ...         : その手紙を返すと，Flower がサーバへ届けてくれる。
    model_record = ArrayRecord(state_dict_to_send)

    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
        # サーバ側で「どの更新が悪性クライアントのものか」を特定するために送る
        # （ALIE 攻撃での差し替えと，検知ログの正解ラベル付けに使う）。
        "partition-id": partition_id,
    }

    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


# 【この部分の説明】
#   サーバから「採点して」という依頼が来るたびに実行される関数。
#   集約後のグローバルモデルを，クライアントの手元の評価用データで採点する。
#   学習も攻撃もしない（モデルは変えずに，採点結果だけを返す）。
#   毎ラウンド，全クライアントではなく一部だけが選ばれて呼ばれる
#   （pyproject.toml の fraction-evaluate。0.2 なら 20台中 4台）。
#   train() と同じ書き方が多いので，説明は違う部分だけにしている。
#
#   ※ 論文で使う精度は，これではなくサーバ側の評価（server_app.py の global_evaluate，
#      テストデータ全体での採点）のほう。こちらは各クライアントの手元データでの参考値。
@app.evaluate()
def evaluate(msg: Message, context: Context):
    # ============================================================
    # 0) 設定（config）取得：データ分割の設定を evaluate にも反映
    #    ★修正：ConfigRecord/TypedDict 等でも安全に扱えるよう dict 化
    # ============================================================
    cfg = dict(msg.content["config"])

    data_split = cfg["data_split"]
    alpha = float(cfg["alpha"])
    trial = int(cfg["trial"])  # Fang et al. (2020)の"trial"に対応：データ分割の再現性用シード兼繰り返し試行番号
    server_round = cfg["server_round"]

    # --- 攻撃設定（evaluateでは攻撃を適用しないが、表示・整合性確認用に読む）---
    attack_type = cfg["attack_type"]
    malicious_ids = set(cfg["malicious_ids"])

    # ============================================================
    # 1) モデル復元
    # ============================================================
    # train() と違い，学習前の重みを退避する必要がない（Δ を計算しない）ので，そのまま読み込むだけ。
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    device = resolve_device(context.run_config["device"])
    model.to(device)

    # ============================================================
    # 2) データ読み込み（評価も学習と同じ分割を使う）
    # ============================================================
    #   _, valloader = ... : train() とは逆に，学習用を捨てて評価用だけを受け取る。
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]

    _, valloader = load_data(
        partition_id,
        num_partitions,
        batch_size,
        data_split=data_split,
        alpha=alpha,
        trial=trial,
        samples_per_client=int(context.run_config["samples-per-client"]),
    )

    # 悪性クライアント判定（表示用）
    is_malicious = partition_id in malicious_ids

    # ============================================================
    # 3) 評価
    # ============================================================
    #   test_fn(model, データ, 装置) : task.py の test を呼ぶ。損失と精度の2つを返す。
    #   eval_loss, eval_acc = ...    : 返ってきた2つの値を，左から順に2つの変数で受け取る。
    eval_loss, eval_acc = test_fn(model, valloader, device)

    # ============================================================
    # 4) 画面表示（評価）
    # ============================================================
    # 画面に表示したあと，採点結果だけをサーバへ返す。
    # train() と違い，重み（"arrays"）は返さない（モデルを変えていないため）。
    print(
        f"[Client {partition_id}] round={server_round} split={data_split} alpha={alpha} "
        f"attack={attack_type} malicious={is_malicious}"
    )

    print_round_summary(
        round_=server_round,
        eval_loss=eval_loss,
        eval_acc=eval_acc,
    )

    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }

    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
