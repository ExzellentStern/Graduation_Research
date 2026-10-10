"""
robust_strategy.py

Flower の ServerApp Strategy。毎ラウンドの「設定の配布」「攻撃の差し込み」「検知スコアの記録」「集約」を担当する。

【役割分担】
- aggregation.py    : 集約の中核ロジック（純粋関数、numpyベース、Flower非依存）
- attack.py         : 攻撃（ALIE はここから呼ぶ。符号反転は client_app.py 側）
- detection.py      : 検知器スコア・分布構造の診断指標・検知性能指標
- robust_strategy.py: それらを Flower の Strategy インターフェースに接続する「配線」部分
- server_app.py     : 設定を読み，この Strategy を組み立てて開始する

FedAvg を継承し、以下の2点だけをオーバーライドする：
1. configure_train / configure_evaluate
   毎ラウンドのconfig（lr, data_split, alpha, attack_type, malicious_ids, trial）を
   config（ConfigRecord）に直接書き込んで渡す、公式ドキュメント記載のシンプルな方法。
   ※ client_app.py が "server_round"（アンダースコア）を参照しているため、
     Flower組み込みが自動セットする "server-round"（ハイフン）とは別に
     互換性のため "server_round" も明示的にセットする。
2. aggregate_train
   クライアントから届いた ArrayRecord を state_dict → ベクトル化し、
   （ALIE なら悪性分を差し替え →）検知スコアと検知性能指標を記録 →
   aggregation.py の aggregate() で集約してから ArrayRecord に戻す。
   MetricRecord（train_loss等のログ用メトリクス）は従来通り重み付き平均のまま。

【現在は使っていない機能】
Adaptive 攻撃・Robust Aggregation（Median / Trimmed Mean / Krum / Bulyan）への対応は外してある。
戻し方は unused_features/README.md を参照。
"""

# 【import の説明】（import 文の読み方は server_app.py の冒頭を参照）
from typing import Iterable, Optional  # 型ヒント用の名前。Iterable は「順に取り出せるもの」，Optional[X] は「X か None」

import numpy as np  # 数値計算のライブラリ。「as np」で np という短い名前を付けている

from flwr.app import ArrayRecord, ConfigRecord   # 重みの入れ物 / 設定の入れ物
from flwr.serverapp import Grid                  # クライアント群との通信窓口
from flwr.serverapp.strategy import FedAvg       # Flower 標準の FedAvg 戦略（下のクラスの土台）
from flwr.app import Message, MetricRecord       # 手紙 / 数値の入れ物

from fl_detection.aggregation import (
    aggregate,              # 集約手法の名前を渡すと，その手法で集約する関数（現在は fedavg だけ）
    stack_updates,          # クライアントごとのベクトルを，1つの表（行=クライアント）にまとめる関数
    state_dict_to_vector,   # モデルの重み（層ごとの辞書）を，1本の長いベクトルに並べ直す関数
    vector_to_state_dict,   # その逆。1本のベクトルを，層ごとの辞書に戻す関数
)
from fl_detection.attack import alie_attack  # ALIE 攻撃（悪性クライアントのベクトルを差し替える関数）
from fl_detection.detection import (
    DETECTOR_KEYS,          # 検知器の名前の一覧（"d1_norm" など4つ）
    STRUCTURE_KEYS,         # 分布構造の診断指標の名前の一覧（6つ）
    all_detector_scores,    # 4つの検知器のスコアをまとめて計算する関数
    evaluate_scores,        # スコアと正解ラベルから，ROC-AUC などの検知性能指標を計算する関数
    filtered_mean_distance, # スコア上位を仮に取り除いたとき，残りの平均が良性の重心からどれだけ離れているかを計算する関数
    structure_diagnostics,  # 分布構造の診断指標を計算する関数
)
from fl_detection.log_output import (
    log_detection_csv,          # 検知スコアを detection_updates.csv に追記する関数（1クライアント1行）
    log_detection_metrics_csv,  # 検知性能指標を detection_metrics.csv に追記する関数（1検知器1行）
)


# 【この部分の説明】
#   「毎ラウンド何をするか」を担当するクラス。この研究の中心になる処理はここにある。
#
#   クラスとは
#       データ（設定など）と，それを使う関数をひとまとめにした設計図。
#       server_app.py が RobustFedAvg(...) と書いて実体を1つ作り，strategy という名前で使っている。
#       クラスの中に書いた関数を「メソッド」と呼ぶ。
#
#   class RobustFedAvg(FedAvg):
#       カッコの中の FedAvg は「継承」。Flower 標準の FedAvg の機能を丸ごと引き継いだうえで，
#       必要な部分だけを自分で書き直す，という意味。
#       ここで書き直しているのは次の3つで，それ以外（ラウンドを回す，クライアントに送る，
#       評価結果をまとめる，など）は FedAvg のものがそのまま使われる。
#           configure_train    : 学習の依頼を出す前に呼ばれる   -> 設定を配る＋グローバルモデルを控える
#           configure_evaluate : 評価の依頼を出す前に呼ばれる   -> 設定を配る
#           aggregate_train    : 学習結果が全員ぶん届いたら呼ばれる -> 攻撃の差し込み・検知ログ・集約
#       これらを呼ぶのは Flower（server_app.py の strategy.start(...) の中）で，
#       このファイルの中に呼び出しは書かれていない。
#
#   1ラウンドの流れ
#       configure_train -> （各クライアントが学習）-> aggregate_train
#       -> configure_evaluate -> （一部のクライアントが評価）-> （server_app.py の global_evaluate）
#
#   self とは
#       メソッドの最初の引数。「このクラスから作った実体そのもの」を指す。
#       self.名前 と書くと，実体が持っているデータを読み書きできる。
#       メソッドをまたいで値を持ち回るために使う（例: __init__ で保存した設定を aggregate_train で読む）。
class RobustFedAvg(FedAvg):
    """
    攻撃の差し込みと検知ログの記録を行う FedAvg 拡張 Strategy。

    引数はすべて server_app.py が pyproject.toml の設定から渡す（ここには既定値を置かない）。
        lr / data_split / alpha / attack_type / malicious_ids / trial:
            毎ラウンド，クライアントへ配布する config
        alie_z            : ALIE の摂動強度 z
        num_clients       : クライアントの総数（設定の num-clients）。届いた返信の数と照合する
        aggregation_method: 集約手法（現在は "fedavg" だけ）
        malicious_ratio / run_id: ログに付ける実験条件の識別子
    """

    # 【__init__ の説明】
    #   実体を作るときに1回だけ実行されるメソッド（初期化）。
    #   server_app.py の RobustFedAvg(lr=..., alpha=..., ...) で渡された値を受け取り，
    #   self.名前 = 値 の形で実体の中に保存する。保存した値は，ほかのメソッドから self.名前 で読める。
    #
    #   引数の並びにある * : これより後ろの引数は，必ず 名前=値 の形で渡さなければならない，という印。
    #                        順番の取り違えを防ぐ。
    #   **kwargs           : ここに書いていない 名前=値 の引数を，まとめて受け取る書き方。
    #                        受け取ったものは，そのまま親の FedAvg に渡している。
    #   super().__init__(...) : 親（FedAvg）の初期化を先に実行する。super() は「親クラス」を指す。
    #   list(malicious_ids)   : リストに変換して保存する（もとのデータを後から変えられても影響を受けないよう，写しを持つ）。
    #   aggregation_method.lower() : 文字列を小文字にそろえる（"FedAvg" と書かれても "fedavg" として扱える）。
    #   名前の先頭の _（_last_global_vector）: 「クラスの中だけで使う」という慣習上の印。
    def __init__(
        self,
        *,
        fraction_evaluate: float,
        lr: float,
        data_split: str,
        alpha: float,
        attack_type: str,
        malicious_ids,
        trial: int,
        alie_z: float,
        num_clients: int,
        aggregation_method: str,
        malicious_ratio: float,
        run_id: str,
        **kwargs,
    ):
        super().__init__(fraction_evaluate=fraction_evaluate, **kwargs)

        # --- 毎ラウンド配布するconfig ---
        self.lr = lr
        self.data_split = data_split
        self.alpha = alpha
        self.attack_type = attack_type
        self.malicious_ids = list(malicious_ids)
        self.trial = trial
        self.alie_z = alie_z

        # --- 設定上のクライアント総数（aggregate_train で，届いた返信の数と照合する） ---
        self.num_clients = num_clients

        # --- ログに付ける実験条件の識別子 ---
        self.malicious_ratio = malicious_ratio
        self.run_id = run_id

        self.aggregation_method = aggregation_method.lower()

        # --- このラウンド開始時点のグローバルモデルベクトル（configure_trainで退避）。
        #     検知は更新ベクトル g_i = w_i - w_global に対して行うので，その w_global として使う。
        self._last_global_vector = None

    # ============================================================
    # 毎ラウンドのconfig配布（公式ドキュメントのシンプルな方法）
    # ============================================================
    # 【この部分の説明】
    #   クライアントへ送る手紙に，設定を書き込む部分。
    #   client_app.py が cfg["data_split"] などで読んでいる値は，ここで入れたもの。
    #
    #   _fill_common_config(self, server_round, config)
    #       config は設定の入れ物（辞書のように 名前 -> 値 で書き込める）。
    #       config["名前"] = 値 で書き込む。学習の依頼と評価の依頼で同じ内容を入れるので，
    #       1つのメソッドにまとめて，下の2つのメソッドから呼んでいる。
    #       -> None は「値を返さない」という型ヒント（config を直接書き換えるだけ）。
    #
    #   configure_train(self, server_round, arrays, config, grid)
    #       学習の依頼を出す直前に Flower が呼ぶ。
    #         arrays : これから配るグローバルモデルの重み
    #         config : これから配る設定（ここで書き足す）
    #       1. 設定を書き込む。
    #       2. 配る前のグローバルモデルを，1本のベクトルに並べ直して self に控えておく。
    #          あとで aggregate_train が「各クライアントの更新 = 届いた重み − 配った重み」を計算するのに使う。
    #          state_dict_to_vector は（ベクトル, 形の情報）の2つを返すので，形の情報は _ で受けて捨てている。
    #       3. return super().configure_train(...) : 残りの仕事（誰に送るかを決めて手紙を作る）は
    #          親の FedAvg に任せ，その結果をそのまま返す。
    #
    #   configure_evaluate(...)
    #       評価の依頼を出す直前に Flower が呼ぶ。設定を書き込んで，あとは親に任せるだけ。
    def _fill_common_config(self, server_round: int, config: ConfigRecord) -> None:
        # client_app.py が "server_round"（アンダースコア）を見ているため、
        # Flower組み込みが自動セットする "server-round"（ハイフン）とは別に明示的にセット
        config["server_round"] = server_round
        config["lr"] = self.lr
        config["data_split"] = self.data_split
        config["alpha"] = self.alpha
        config["attack_type"] = self.attack_type
        config["malicious_ids"] = self.malicious_ids
        config["trial"] = self.trial

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        self._fill_common_config(server_round, config)

        # 検知で使う「集約前のグローバルモデル」を退避しておく
        # （aggregate_train は arrays を受け取らないため、ここで保持するしかない）
        global_state_dict = arrays.to_torch_state_dict()
        self._last_global_vector, _ = state_dict_to_vector(global_state_dict)

        return super().configure_train(server_round, arrays, config, grid)

    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        self._fill_common_config(server_round, config)
        return super().configure_evaluate(server_round, arrays, config, grid)

    # ============================================================
    # 集約本体（攻撃の差し込み → 検知ログ → 集約）
    # ============================================================
    # 【補助メソッドの説明】
    #   クライアントから届いた手紙の中身（content）から，数値を1つ探して取り出す補助。
    #   中身は 名前 -> 入れ物 の対応表で，数値は "metrics" の入れ物に入っている
    #   （client_app.py が最後に送っている train_loss / num-examples / partition-id）。
    #
    #   _extract_weight(content)       : そのクライアントのデータ数（"num-examples"）を取り出す。
    #                                    self.weighted_by_key は FedAvg が持っている設定で，中身は "num-examples"。
    #   _extract_metric(content, key)  : 名前（key）を指定して値を取り出す。"partition-id" を取るのに使う。
    #
    #   for value in content.values(): : 入れ物を1つずつ順に調べる。
    #   try: ... except TypeError: continue
    #                                  : try の中でエラー（TypeError）が起きたら，止まらずに except の中を実行する。
    #                                    重みの入れ物のように「名前 in 入れ物」の調べ方ができないものは飛ばす，という意味。
    #                                    continue は「次の繰り返しへ進む」。
    #   return None                    : 最後まで見つからなければ None（値なし）を返す。
    def _extract_weight(self, content) -> Optional[float]:
        """reply の RecordDict から weighted_by_key（通常 'num-examples'）の値を探す。"""
        for value in content.values():
            try:
                if self.weighted_by_key in value:
                    return float(value[self.weighted_by_key])
            except TypeError:
                continue
        return None

    def _extract_metric(self, content, key: str):
        """reply の RecordDict から任意のメトリクスキー（例: 'partition-id'）の値を探す。"""
        for value in content.values():
            try:
                if key in value:
                    return value[key]
            except TypeError:
                continue
        return None

    # 【aggregate_train の説明】
    #   全クライアントの学習結果が届いたあと，Flower が毎ラウンド1回呼ぶ。この研究で一番大事なメソッド。
    #     server_round : 何ラウンド目か
    #     replies      : クライアントから届いた手紙の集まり
    #   返すもの       : （集約後の重み, 集約後の数値）の2つ
    #
    #   処理の流れ
    #       1. 届いた手紙を確認し（台数が設定と合っているかも確かめる），クライアント番号の順に並べる
    #       2. 各クライアントの重みを，1本のベクトルに並べ直して表にする
    #       3. どの行が悪性クライアントかを特定する
    #       4. ALIE 攻撃なら，悪性の行を攻撃用のベクトルに差し替える
    #       5. 良性の広がり（spread）と，悪性の押し出し具合（push_ratio）を測る
    #       6. 検知器 D1〜D4 のスコアを計算し，記録する          <- 研究の中心
    #       7. スコアと正解から ROC-AUC などを計算し，記録する  <- 研究の中心
    #       8. 重みを集約（FedAvg）して，新しいグローバルモデルを作る
    #   6 と 7 は「記録するだけ」で，8 の集約には影響しない（検知結果で誰かを除外したりはしていない）。
    #
    #   このメソッドで扱う表の形
    #       updates : 行 = クライアント（20台なら20行），列 = モデルの重み1つ1つ（約27万列）
    #       updates[3] と書くと，3番（0から数えて）の行，つまり1台ぶんのベクトルが取り出せる。
    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ):
        """ArrayRecordを集約し、MetricRecordは重み付き平均のまま返す。集約の直前に検知ログを記録する。"""
        # --- 1. 届いた手紙の確認 ---
        #   self._check_and_log_replies(...) : FedAvg が持っているメソッド。届いた手紙のうち，
        #                                      エラーなく返ってきたものだけを選び出す（残りは _ で捨てている）。
        #   [msg.content for msg in valid_replies]
        #                                    : リスト内包表記。手紙を1通ずつ取り出し，中身だけを集めたリストを作る。
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)

        # --- 台数の照合 ---
        # 設定の num-clients と，実際に学習結果を返してきたクライアントの数が一致しなければ，実行を止める。
        # 台数は2か所で別々に決まっている。
        #     num-clients（pyproject.toml）        : 悪性クライアントを何番から選ぶかに使う
        #     実際に起動しているクライアントの台数 : データを何個に分けるかに使う（Flower 側の num-supernodes）
        # この2つが食い違っても，以前はエラーにならなかった。たとえば 10 台しか起動していないのに
        # num-clients が 20 だと，悪性に選ばれた番号のうち実在しないものが黙って無視され，
        # 悪性の台数が設定より少ない，想定と違う実験がそのまま走ってしまう。
        # クライアントが途中でエラーになって返信が欠けた場合も，ここで検出できる。
        #   len(valid_replies) : 有効な返信の数。
        #   != は「等しくない」。raise ValueError(...) でエラーを起こして実行を止める。
        if len(valid_replies) != self.num_clients:
            raise ValueError(
                f"[RobustFedAvg] round={server_round}: 学習結果を返したクライアントは {len(valid_replies)} 台ですが，"
                f"設定の num-clients は {self.num_clients} です。"
                "起動しているクライアントの台数（Flower の num-supernodes）と pyproject.toml の num-clients が"
                "一致しているか，クライアント側でエラーが出ていないかを確認してください。"
            )

        reply_contents = [msg.content for msg in valid_replies]

        # クライアントからの返信は，届いた順（実行のたびに変わる）で並んでいる。
        # クライアント番号の順に並べ直しておく。こうしないと，合計や平均を取る順番が実行ごとに変わり，
        # 同じ条件で回し直したときに計算結果が末尾の桁でずれる（再現性のため）。
        #   リスト.sort(key=...)  : リストを並べ替える。key には「何を基準に並べるか」を返す関数を渡す。
        #   lambda c: ...         : その場で作る名前なしの小さな関数。「c を受け取って ... を返す」。
        #                           ここでは「手紙の中身 c から，クライアント番号を整数で取り出す」。
        reply_contents.sort(key=lambda c: int(self._extract_metric(c, "partition-id")))

        # --- 2. 各クライアントの state_dict をベクトル化 ---
        #   state_dict : モデルの重みを「層の名前 -> Tensor」で持つ辞書（層ごとに形がばらばら）。
        #   検知や集約では全部の重みを同列に扱いたいので，1本の長いベクトルに並べ直す。
        #
        #   content[self.arrayrecord_key] : 手紙の中身から，重みの入れ物を取り出す（arrayrecord_key の中身は "arrays"）。
        #   state_dict_to_vector(sd)      : （ベクトル, 形の情報）を返す。
        #                                   形の情報 shapes は，あとでベクトルを層ごとの辞書に戻すときに使う。
        #   vectors.append(vec)           : リストの末尾に追加する。
        #   if shapes_ref is None:        : 形の情報は全クライアントで同じなので，最初の1台ぶんだけ控える。
        #   stack_updates(vectors)        : ベクトルのリストを，1つの表（2次元配列）に積み重ねる。
        #   updates.shape[0]              : 表の行数（＝届いたクライアントの台数）。shape は（行数, 列数）。
        state_dicts = [
            content[self.arrayrecord_key].to_torch_state_dict()
            for content in reply_contents
        ]
        vectors = []
        shapes_ref = None
        for sd in state_dicts:
            vec, shapes = state_dict_to_vector(sd)
            vectors.append(vec)
            if shapes_ref is None:
                shapes_ref = shapes
        updates = stack_updates(vectors)
        n = updates.shape[0]

        # --- 3. 各クライアントがmalicious/honestのどちらかを特定する ---
        # 診断指標（benign_spread等）と検知ログは attack_type によらず計算するため、
        # malicious_ids との突合はここで毎回行う。
        #   partition_ids     : 表の各行が，何番のクライアントのものかを並べたリスト。
        #   malicious_ids_set : 悪性クライアントの番号の集合（「含まれるか」を調べやすい形）。
        #   malicious_pos     : 表の中で，悪性クライアントが「何行目にいるか」の一覧。
        #       [i for i, pid in enumerate(partition_ids) if 条件]
        #           : 条件つきのリスト内包表記。（行番号 i, クライアント番号 pid）を順に見て，
        #             条件が成り立つ行番号 i だけを集める。
        #       np.array([...], dtype=int) : それを numpy の整数の配列にする。
        #   benign_pos        : 同じく，良性クライアントが何行目にいるかの一覧（悪性でない行すべて）。
        #   .tolist()         : numpy の配列を，普通の Python のリストに戻す。
        #   ※ 正解（どれが悪性か）を知っているのは，実験として評価するため。
        #     検知器には渡しておらず，検知器はスコアを出すだけで，答え合わせは後で行う。
        partition_ids = [self._extract_metric(c, "partition-id") for c in reply_contents]
        malicious_ids_set = set(self.malicious_ids)
        malicious_pos = np.array(
            [
                i
                for i, pid in enumerate(partition_ids)
                if pid is not None and int(pid) in malicious_ids_set
            ],
            dtype=int,
        )
        malicious_pos_set = set(malicious_pos.tolist())
        benign_pos = np.array(
            [i for i in range(n) if i not in malicious_pos_set], dtype=int
        )

        # --- ALIE 攻撃（sotsuron_plan2.tex 4.3 A2）: サーバ側で正常更新の
        #     座標ごと平均・標準偏差から悪性更新をクラフトする aggregation-agnostic 攻撃。
        #     集約直前に，悪性クライアント分のベクトルを攻撃済みベクトルへ差し替える。
        #     符号反転（sign_flip）はクライアント側（client_app.py）で既に適用済みなので，ここでは何もしない。
        #   （ここが手順 4）
        #   if A and B and C:       : 3つの条件がすべて成り立つときだけ実行する
        #                             （攻撃が ALIE，悪性が1台以上，良性が1台以上）。
        #   alie_attack(...)        : attack.py の関数。良性の行だけから，重み1つ1つについて平均と標準偏差を求め，
        #                             「平均 − z × 標準偏差」というベクトルを作って，悪性の行すべてに入れる。
        #                             良性の集団のばらつきの中に収まるように作るので，見つかりにくい。
        #   updates = alie_attack(...) : 差し替え後の表で updates を置き換える。
        #                                悪性クライアントが正直に学習して送ってきた重みは，ここで捨てられる。
        if self.attack_type == "alie" and len(malicious_pos) > 0 and len(benign_pos) > 0:
            updates = alie_attack(
                updates,
                malicious_idx=malicious_pos,
                benign_idx=benign_pos,
                z=self.alie_z,
            )

        # --- honestクラスタの広がり（spread）と，maliciousの押し出し距離（push）の比 ---
        # 集約直前（攻撃適用後）のupdatesを使い、honest重心からの平均距離として定義する。
        # honestが存在しない/maliciousが存在しない場合はNoneのままログする。
        #   （ここが手順 5）
        #   updates[benign_pos]           : 表から良性の行だけを取り出す。
        #   .mean(axis=0)                 : 行の方向に平均する（＝重み1つ1つについて，クライアント間の平均を取る）。
        #                                   結果は1本のベクトルで，良性クライアントの「重心」（中心の位置）になる。
        #   benign_vectors - benign_center: 各良性クライアントの，重心からのずれ。
        #   np.linalg.norm(..., axis=1)   : 行ごとにベクトルの長さ（ノルム）を計算する（＝重心からの距離）。
        #   .mean()                       : その距離を平均する。float(...) で普通の小数にする。
        #   benign_spread : 良性クライアントが，重心からどのくらい離れて散らばっているか（平均距離）。
        #   push          : 悪性クライアントが，良性の重心からどのくらい離れているか（平均距離）。
        #   push_ratio    : push ÷ spread。1 より大きければ悪性は良性の集団の外側にいて，
        #                   1 より小さければ集団の内側に紛れ込んでいる。中心仮説で使う指標。
        #   if benign_spread > 1e-12:     : 0 で割るのを防ぐ（1e-12 は 0.000000000001）。
        benign_spread = None
        push_ratio = None
        if len(benign_pos) > 0:
            benign_vectors = updates[benign_pos]
            benign_center = benign_vectors.mean(axis=0)
            benign_spread = float(np.linalg.norm(benign_vectors - benign_center, axis=1).mean())
            if len(malicious_pos) > 0:
                malicious_vectors = updates[malicious_pos]
                malicious_push_dist = float(
                    np.linalg.norm(malicious_vectors - benign_center, axis=1).mean()
                )
                if benign_spread > 1e-12:
                    push_ratio = malicious_push_dist / benign_spread

        # --- sotsuron_plan2.tex 4.2--4.5 対応：更新ベクトルレベルの検知器スコアを
        #     1クライアント1行で logs/detection_updates.csv へ記録する。
        #     検知は更新ベクトル g_i = w_i - w_global に対して行うため、
        #     受信した（攻撃適用後の）モデルベクトルからグローバルモデルを引いて渡す。
        #     正解ラベル is_malicious は評価専用であり、検知器スコアの計算には使わない。
        #     この記録が本実験の目的なので，失敗したら握りつぶさずに実行ごと止める
        #     （止めないと final_model.pt だけが出来て，ログが欠けた条件が「完了」扱いになる）。
        #   （ここが手順 6）
        #   deltas = updates - self._last_global_vector[None, :]
        #       : 各行から「配った時点のグローバルモデル」を引いて，更新ベクトル（学習で動いた分）にする。
        #         [None, :] は，1本のベクトルを「1行だけの表」の形にする書き方。
        #         こうすると，表のすべての行から同じベクトルが引かれる（numpy のブロードキャスト）。
        #   all_detector_scores(deltas)  : 4つの検知器のスコアを計算する。
        #                                  返り値は辞書で，scores["d1_norm"][3] が「D1 の，3行目のクライアントのスコア」。
        #   structure_diagnostics(...)   : 分布構造の診断指標を計算する（こちらは正解を使う。考察用）。
        #   delta_norms                  : 各クライアントの更新ベクトルの長さ。
        #   for pos in range(n):         : 表の行を 0 行目から順に見て，1クライアントぶんの記録（辞書 row）を作る。
        #   (int(pid) if pid is not None else None) : 番号があれば整数にし，無ければ None のままにする。
        #   int(pos in malicious_pos_set): 悪性の行なら 1，良性の行なら 0（True/False を 1/0 にしている）。
        #   for key in DETECTOR_KEYS:    : 検知器の名前を順に取り出し，row[名前] = スコア と書き足す。
        #   detection_rows.append(row)   : 出来た1行ぶんを，リストに追加する。
        #   log_detection_csv(...)       : 全クライアントぶんをまとめて detection_updates.csv に追記する。
        deltas = updates - self._last_global_vector[None, :]
        scores = all_detector_scores(deltas)
        structure = structure_diagnostics(deltas, benign_pos, malicious_pos)
        delta_norms = np.linalg.norm(deltas, axis=1)
        detection_rows = []
        for pos in range(n):
            pid = partition_ids[pos]
            row = {
                "run_id": self.run_id,
                "round": server_round,
                "aggregation_method": self.aggregation_method,
                "data_split": self.data_split,
                "alpha": self.alpha,
                "attack_type": self.attack_type,
                "malicious_ratio": self.malicious_ratio,
                "trial": self.trial,
                "client_pos": pos,
                "partition_id": (int(pid) if pid is not None else None),
                "is_malicious": int(pos in malicious_pos_set),
                "delta_norm": float(delta_norms[pos]),
                "benign_spread": benign_spread,
                "push_ratio": push_ratio,
                "alie_z": (self.alie_z if self.attack_type == "alie" else None),
            }
            for key in DETECTOR_KEYS:
                row[key] = float(scores[key][pos])
            for key in STRUCTURE_KEYS:
                row[key] = structure[key]
            detection_rows.append(row)
        log_detection_csv(detection_rows)

        # --- 検知性能指標（ROC-AUC など）もこの場で計算して記録する。
        #     解析側（analyze_detection.py）は読み込みと集計だけにして，
        #     指標の計算式を detection.py の1か所に保つため。
        #     正解ラベルは評価にだけ使い，検知器スコアや集約には戻さない。
        #   （ここが手順 7）
        #   y_true      : 正解のリスト（各クライアントが悪性なら 1，良性なら 0）。
        #   n_malicious : 悪性クライアントの台数（1 の個数＝合計）。
        #   metric_base : どの検知器の行にも共通で入れる項目（実験条件と，spread などの指標）。
        #   evaluate_scores(y_true, スコア, k=n_malicious)
        #               : 1つの検知器について，ROC-AUC などをまとめて計算する。
        #                 k は「スコアの上位 k 台を悪性と判定したら，どれだけ当たるか」を見るときの k。
        #   {**metric_base, "detector": key, ...}
        #               : 共通の項目をすべて写したうえで，その検知器の結果を書き足した新しい辞書を作る。
        #   検知器が4つあるので，1ラウンドにつき4行が detection_metrics.csv に追記される。
        y_true = [row["is_malicious"] for row in detection_rows]
        n_malicious = int(sum(y_true))
        metric_base = {
            "run_id": self.run_id,
            "round": server_round,
            "aggregation_method": self.aggregation_method,
            "data_split": self.data_split,
            "alpha": self.alpha,
            "attack_type": self.attack_type,
            "alie_z": (self.alie_z if self.attack_type == "alie" else None),
            "malicious_ratio": self.malicious_ratio,
            "trial": self.trial,
            "n_clients": n,
            "n_malicious": n_malicious,
            "benign_spread": benign_spread,
            "push_ratio": push_ratio,
        }
        for key in STRUCTURE_KEYS:
            metric_base[key] = structure[key]

        # --- 参考の値（記録するだけ。卒業論文では扱わない。経緯は detection.filtered_mean_distance のメモを参照）---
        # 「検知器が疑った上位 f 台を仮に取り除いたら，残りの平均は良性の重心にどれだけ近づくか」を残しておく。
        # スイープの後からは計算できない（更新ベクトルを保存していない）ので，ここで計算する。集約には使わない。
        #   mean_dist_unfiltered : 何も取り除かなかった場合の値（比較の基準。同じラウンドの4行で同じ値）。
        #   mean_dist_filtered   : その検知器のスコア上位 f 台を取り除いた場合の値（下の for の中で検知器ごとに計算）。
        metric_base["mean_dist_unfiltered"] = filtered_mean_distance(deltas, scores[DETECTOR_KEYS[0]], benign_pos, k=0)
        metric_rows = []
        for key in DETECTOR_KEYS:
            m = evaluate_scores(y_true, scores[key], k=n_malicious)
            metric_rows.append({
                **metric_base,
                "detector": key,
                "auc": m["auc"],
                "ap": m["ap"],
                "recall_at_f": m["recall_at_k"],
                "precision_at_f": m["precision_at_k"],
                "mean_dist_filtered": filtered_mean_distance(deltas, scores[key], benign_pos, k=n_malicious),
            })
        log_detection_metrics_csv(metric_rows)

        # --- 8. 集約 ---
        # --- fedavg は num-examples（各クライアントのデータ数）による重み付けを行う ---
        #   weights : 各クライアントのデータ数を並べた配列。データの多いクライアントほど，平均への影響が大きくなる。
        #       A or 1.0 : A が取れなかった（None だった）ときは，代わりに 1.0 を使う。
        #   aggregate(...) : 表（全クライアントのベクトル）を，重み付きで平均して1本のベクトルにする。
        #                    ALIE のときは，差し替え後の悪性ベクトルも一緒に平均される（攻撃がモデルに効く仕組み）。
        #   vector_to_state_dict(...) : 1本のベクトルを，層ごとの辞書（state_dict）に戻す。
        #                               state_dicts[0] は，各層のデータ型を合わせるための見本として渡している。
        #   ArrayRecord(...)          : それを Flower の入れ物に入れる。これが次のラウンドのグローバルモデルになる。
        weights = np.array(
            [self._extract_weight(c) or 1.0 for c in reply_contents],
            dtype=np.float64,
        )

        aggregated_vector = aggregate(self.aggregation_method, updates, weights=weights)

        new_state_dict = vector_to_state_dict(aggregated_vector, shapes_ref, state_dicts[0])
        arrays = ArrayRecord(new_state_dict)

        # --- メトリクス（train_loss等）は従来通り重み付き平均で集計 ---
        #   self.train_metrics_aggr_fn : FedAvg が持っている集計用の関数。クライアントが送ってきた数値を，
        #                                データ数で重み付けして平均する。
        #   return arrays, metrics     : 集約後の重みと数値の2つを Flower に返して終わる。
        metrics = self.train_metrics_aggr_fn(reply_contents, self.weighted_by_key)

        return arrays, metrics
