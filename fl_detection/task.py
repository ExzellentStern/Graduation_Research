"""
このファイルの役割
- モデル(Net)の定義（MNIST 用）
- Federated Learning 用データ分割（Dirichlet による Non-IID）
- 学習(train) / 評価(test) の基本ループ
- サーバ側の共通テストデータローダ作成(load_centralized_dataset)

※ 注意：FLの実験では「再現性」が重要。
  - データ分割の trial（Fang et al. (2020)の"trial"に対応する繰り返し試行番号）を固定
  - 前処理（Normalizeなど）はサーバ・クライアントで揃える

【変更履歴】
  - CIFAR-10+CNN 版から MNIST+DNN 版へ変更（実験計画書 7.1 対応）
    - データセットを uoft-cs/cifar10 → ylecun/mnist に変更
    - Normalize をグレースケール1chの値に変更
    - Net を CNN → 全結合DNN（入力784 → 128 → 64 → 10）に変更
    - forward 内で画像を flatten する処理を追加
  - Fang et al. (2020) との条件整合（対応関係の検証レポート対応）
    - Net を全結合DNN → CNN（Fang Table 7a準拠）に変更（旧版は NetFC として残す）
    - Non-IID分割を Dirichlet → Fang方式（ラベルlを確率pで対応グループへ割当）に変更
      （旧Dirichlet実装は data_split="dirichlet" として参考用に残す）
    - optimizer を AdamW → SGD に変更（Fangの式2.1 w<-w-η∇ℓ(w;b) に合わせた素朴なSGD）
    - batch_size を 25 → 32 に変更（Fangのデフォルトに合わせた）
  - 検知評価（sotsuron_plan2.tex）への切替
    - Non-IID分割を Dirichlet に戻した（Non-IID度を alpha で連続的に操作するため）
    - 使わなくなった旧モデル NetFC と IID / Fang方式の分割は
      unused_features/task_extra.py へ移した（戻し方は unused_features/README.md）
  - データ数を全クライアントで同じにした分割 "dirichlet_equal" を追加し，実験ではこちらを使う（2026-10-11）
    - 従来の "dirichlet" は，alpha を小さくするとラベルの偏りと同時にデータ数の偏りも強まるため，
      検知性能の変化がどちらの偏りによるものかを分けられなかった
      （alpha=0.1 では 41〜15,083 枚まで開き，更新ノルムの大小がデータ数でほぼ決まっていた）。
    - "dirichlet_equal" は枚数を固定し，ラベルの比率だけを alpha で偏らせる（ラベル分布の偏りのみ）。
    - 従来の "dirichlet" は，以前の結果を再現するために残してある。
"""

# 【このファイルの読み方】
#   「何を学習するか」をまとめたファイル。連合学習の仕組み（Flower）とは関係のない，
#   ふつうの機械学習の部品が入っている。5つの部分に分かれている。
#       1) モデル         : Net（画像を受け取って，0〜9 のどの数字かを答える CNN）
#       2) 前処理         : 画像を，モデルに入れられる数値の形に直す
#       3) データ分割     : 学習用データ 60,000 枚を，クライアントの台数ぶんに分ける（Non-IID を作る場所）
#       4) テストデータ   : サーバが採点に使う 10,000 枚を用意する
#       5) 学習と評価     : モデルを学習させる train()，採点する test()
#
#       関数                       呼び出す側
#       Net                        server_app.py / client_app.py
#       load_data                  client_app.py（自分の担当ぶんのデータを読む）
#       load_centralized_dataset   server_app.py（毎ラウンドの採点）
#       train                      client_app.py（train_fn という名前で呼んでいる）
#       test                       server_app.py / client_app.py（test_fn）
#
# 【import の説明】
import torch                      # PyTorch 本体
import torch.nn as nn             # ニューラルネットワークの部品（層など）。nn という短い名前を付けている
import numpy as np                # 数値計算のライブラリ（データ分割で使う）
from datasets import load_dataset # Hugging Face のデータセットを読み込む関数（MNIST をここから取得する）
from torch.utils.data import DataLoader  # データを，決まった枚数（バッチ）ずつ取り出せるようにする道具
from torchvision.transforms import Compose, Normalize, ToTensor  # 画像の前処理の部品


# ============================================================
# 1) モデル定義（MNIST 用 CNN）
#    Fang et al. (2020) Table 7a に準拠：
#    Conv(3x3x30)+ReLU -> MaxPool(2x2) -> Conv(3x3x50)+ReLU -> MaxPool(2x2)
#    -> FC(200)+ReLU -> FC(10)（Softmaxは CrossEntropyLoss 内で暗黙的に適用）
#    28x28 入力 -> conv1:26x26x30 -> pool:13x13x30
#              -> conv2:11x11x50 -> pool:5x5x50(=1250) -> FC(1250,200) -> FC(200,10)
# ============================================================
# 【この部分の説明】
#   手書き数字の画像（縦28 × 横28，白黒）を受け取り，10個の数値（0〜9 それぞれの「らしさ」の点数）を返すモデル。
#   点数が一番高い数字が，モデルの答えになる。
#
#   class Net(nn.Module):
#       nn.Module は PyTorch の「モデルの土台」。これを継承すると，重みの管理や保存の機能が付いてくる。
#   def __init__(self):
#       モデルを作るときに1回だけ実行される。使う「層」をここで用意する。
#       層は，学習で値が変わる重みを中に持っている。重みは全部で 266,060 個。
#       super(Net, self).__init__() : 親（nn.Module）の初期化を先に実行する決まり文句。
#
#   用意している層
#     nn.Conv2d(1, 30, kernel_size=3)  : 畳み込み層。3×3 の小さな窓を画像の上で滑らせ，線や角などの模様を拾う。
#                                        入力1枚（白黒）から，30種類の模様の地図を作る。
#     nn.Conv2d(30, 50, kernel_size=3) : 2つ目の畳み込み層。30枚の地図から，より複雑な50種類の模様の地図を作る。
#     nn.MaxPool2d(kernel_size=2, stride=2)
#                                      : プーリング層。2×2 の範囲ごとに最大値だけを残し，縦横を半分に縮める。
#                                        重みは持たない。位置の細かいずれに強くなる。
#     nn.Linear(50 * 5 * 5, 200)       : 全結合層。1250 個の数値を受け取り，200 個の数値にまとめる。
#     nn.Linear(200, 10)               : 最後の全結合層。200 個から，10 個（0〜9 の点数）を出す。
#     nn.ReLU()                        : 活性化関数。マイナスの値を 0 にし，プラスの値はそのまま通す。
#                                        これを挟まないと，層を重ねても単純な計算しかできない。
#
#   def forward(self, x):
#       データがモデルの中を流れる順番を書く。model(画像) と呼ぶと，この forward が実行される。
#       x の形の変化（1枚あたり）
#           入力                  : 1 × 28 × 28
#           conv1 -> relu         : 30 × 26 × 26   （3×3 の窓なので，縦横が2ずつ減る）
#           pool                  : 30 × 13 × 13   （縦横が半分）
#           conv2 -> relu         : 50 × 11 × 11
#           pool                  : 50 × 5 × 5     （= 1250 個の数値）
#           view                  : 1250           （1列に並べ直す）
#           fc1 -> relu           : 200
#           fc2                   : 10             （0〜9 の点数）
#       self.pool(self.relu(self.conv1(x))) : 内側から順に実行される（conv1 -> relu -> pool）。
#       x.view(x.size(0), -1)  : 形を並べ直す。x.size(0) は一度に入れた画像の枚数。
#                                -1 は「残りは自動で計算して」の意味（ここでは 1250 になる）。
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 30, kernel_size=3)
        self.conv2 = nn.Conv2d(30, 50, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(50 * 5 * 5, 200)
        self.fc2 = nn.Linear(200, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


# ============================================================
# 2) データ前処理（PyTorchテンソル化 & 正規化）
#    MNIST はグレースケール1chなので mean/std も1要素にする
#    （MNIST全体の統計量: mean=0.1307, std=0.3081 を使用）
# ============================================================
# 【この部分の説明】
#   画像ファイルのままではモデルに入れられないので，数値の配列（Tensor）に直し，値の範囲を整える。
#     Compose([...])                    : 複数の前処理を，上から順に続けて行うようにまとめる。
#     ToTensor()                        : 画像を Tensor に変える。画素値は 0〜255 から 0〜1 になる。
#     Normalize((0.1307,), (0.3081,))   : （値 − 0.1307）÷ 0.3081 を計算する（正規化）。
#                                         MNIST 全体の平均が 0、ばらつきが 1 になるようにして，学習を安定させる。
#                                         (0.1307,) のカンマは「要素が1つのタプル」を表す書き方（白黒なので1つ）。
#
#   apply_transforms(batch)
#     データを取り出すたびに呼ばれ，前処理を適用する。batch は「列の名前 -> 値のリスト」の辞書。
#     MNIST では画像の列の名前が "image" だが，このプログラムのほかの部分は "img" という名前で
#     読んでいるので，前処理のあとで名前を "img" に付け替えている。
#       [pytorch_transforms(img) for img in batch[key]] : 画像1枚ずつに前処理をかける（リスト内包表記）。
#       batch.pop(key)           : 辞書からその名前の値を取り出し，同時に辞書から消す。
#       batch["img"] = batch.pop(key) : つまり，名前の付け替え。
pytorch_transforms = Compose([
    ToTensor(),
    Normalize((0.1307,), (0.3081,)),
])


def apply_transforms(batch):
    """HuggingFace datasets の batch に対して前処理を適用する。"""

    # MNIST(ylecun/mnist) の画像カラムは "image"。
    # CIFAR-10 は "img" だったため、両対応のままにしておく。
    if "img" in batch:
        key = "img"
    elif "image" in batch:
        key = "image"
    else:
        # 画像が無いなら変換するものが無いので、そのまま返す
        return batch

    batch[key] = [pytorch_transforms(img) for img in batch[key]]

    # 学習ループ側は batch["img"] を期待しているので、統一
    if key != "img":
        batch["img"] = batch.pop(key)

    return batch


# ============================================================
# 3) FL用データ分割（Dirichlet による Non-IID）
# ============================================================
# 方針：
# - MNIST(train) を 1回だけ読み込み、各クライアント用 index を作って再利用する
# - Non-IID の作り方は2通りある（pyproject.toml の data-split で選ぶ）。どちらも alpha が大きいとほぼ IID，小さいほど偏りが強い。
#     "dirichlet_equal"（実験で使う）: クライアントごとに Dirichlet 分布で「クラスの比率」を作る。枚数は全員同じ。
#                                      偏るのはラベルの比率だけ（ラベル分布の偏り）。
#     "dirichlet"（以前の方式）      : クラスごとに Dirichlet 分布で「各クライアントへの割当比率」を作る。
#                                      ラベルの比率と一緒に，枚数もクライアントによって大きく変わる
#                                      （ラベル分布の偏り + データ数の偏り）。
#
# 注意：
# - data_split / alpha / num_partitions が変わったら index を作り直す必要があるため、キー付きキャッシュにする
# - transforms は Dataset.with_transform(apply_transforms) で遅延適用（メモリ節約）
#
# 【この部分の説明】
#   Non-IID（クライアントごとにデータが偏っている状態）を作っているのは，ここ。
#   研究の横軸である「Non-IID 度」は，下の _build_dirichlet_equal_partitions の alpha で決まる。
#
#   【Non-IID の種類について】
#     Non-IID には種類がある（NIID-Bench, Li et al., 2022 の整理）。
#       ラベル分布の偏り : クライアントごとに，持っているクラス（数字）の比率が違う。
#       特徴分布の偏り   : 同じクラスでも，見た目（筆跡など）がクライアントごとに違う。
#       データ数の偏り   : クライアントごとに，持っている枚数が違う。
#     この研究で調べたいのは「ラベル分布の偏り」が検知に与える影響なので，
#     枚数は全員同じに固定し，ラベルの比率だけを変える "dirichlet_equal" を使う。
#
#   3つの変数（ファイルの中だけで使う。名前がすべて大文字なのは「ファイル全体で共有する値」の慣習）
#     _TRAIN_DS        : 読み込んだ学習用データをしまっておく入れ物。最初は None（空）。
#     _PARTITION_CACHE : 一度作った分割をしまっておく辞書。同じ条件なら作り直さずに使い回す（キャッシュ）。
#     _DATASET_NAME    : データセットの名前。
#
#   _load_train_dataset()
#     学習用データ（60,000 枚）を読み込む。2回目以降は，しまっておいたものをそのまま返す。
#       global _TRAIN_DS              : 関数の中から，外にある変数を書き換えるための宣言。
#       load_dataset(名前, split="train") : 学習用の部分を読み込む（初回だけインターネットから取得し，以後は手元の控えを使う）。
#       .with_transform(apply_transforms) : 「取り出すときに前処理をかける」と予約する（この時点ではまだ変換しない）。
_TRAIN_DS = None
_PARTITION_CACHE = {}  # key -> List[List[int]]

# データセット名（MNIST）。切り替えが必要な場合はここだけ変更すればよい。
_DATASET_NAME = "ylecun/mnist"


def _load_train_dataset():
    """MNIST train split をロードしてキャッシュする（1プロセスで1回）。"""
    global _TRAIN_DS
    if _TRAIN_DS is None:
        _TRAIN_DS = load_dataset(_DATASET_NAME, split="train")
        _TRAIN_DS = _TRAIN_DS.with_transform(apply_transforms)
    return _TRAIN_DS


# 【_build_dirichlet_partitions の説明】（以前の分割。data-split = "dirichlet" のときだけ使う。現在の実験では使っていない）
#   学習用データを，クライアントの台数ぶんに分ける。返すのは「クライアントごとの，担当する画像の番号の一覧」。
#
#   考え方（数字 0〜9 のそれぞれについて，別々に同じことをする）
#     1. その数字の画像（約 6,000 枚）を集めて，順番をかき混ぜる。
#     2. 「どのクライアントに何割ずつ配るか」の割合を，乱数で決める（ディリクレ分布）。
#     3. その割合どおりに，画像を各クライアントへ配る。
#   割合を数字ごとに引き直すので，「数字 3 はクライアント A にほとんど，数字 7 はクライアント B にほとんど」
#   というように，クライアントごとに持っている数字が偏る。
#
#   alpha の意味（ディリクレ分布のつまみ）
#     alpha が大きい（例: 100）: 割合がほぼ均等になる。どのクライアントも，全部の数字を同じくらい持つ（ほぼ IID）。
#     alpha が小さい（例: 0.1）: 割合が極端に偏る。1つの数字が，ほぼ1〜2台のクライアントに集中する。
#   割合は数字ごとに決まるので，「どの数字を持つか」だけでなく，合計の枚数もクライアントによって大きく変わる
#   （alpha=0.1 では，最少 48 枚から最多 11,117 枚まで開いた条件がある）。
#
#   コードの読み方
#     rng = np.random.default_rng(trial) : trial 番号を種にした乱数の発生器。同じ trial なら毎回同じ分け方になる。
#     num_classes = int(labels.max()) + 1 : クラスの数。ラベルの最大値が 9 なので 10。
#     client_indices = [[] for _ in range(num_partitions)]
#         : 空のリストを，クライアントの台数ぶん用意する。ここに画像の番号をためていく。
#           _ は「使わない変数」の慣習的な名前（回数ぶん繰り返したいだけ）。
#     np.where(labels == c)[0] : ラベルが c である画像の番号（何枚目か）の一覧。
#     rng.shuffle(idx_c)       : その一覧の順番をかき混ぜる。
#     rng.dirichlet(alpha=np.full(num_partitions, alpha))
#         : クライアントの台数ぶんの割合を，ディリクレ分布から引く。合計は 1 になる。
#           np.full(個数, 値) は，同じ値を個数ぶん並べた配列。
#     counts = (proportions * len(idx_c)).astype(int)
#         : 割合 × 枚数 で，各クライアントに配る枚数を出す。.astype(int) で小数点以下を切り捨てる。
#     端数調整 : 切り捨てたぶん，合計が実際の枚数より少なくなる。その差（diff）を，
#                割合の大きいクライアントから順に1枚ずつ足して，合計を合わせている。
#                np.argsort(-proportions) は，割合の大きい順に並べたときのクライアントの番号。
#     cursor                   : 「ここまで配った」位置。クライアントごとに take 枚ずつ切り出して進める。
#     idx_c[cursor:cursor + take] : cursor 番目から take 枚ぶんを取り出す（スライス）。
#     リスト.extend(別のリスト)   : リストの末尾に，別のリストの中身を全部つなげる。
#     最後に，各クライアントの中で画像の順番をかき混ぜ（数字 0 から順に並んでいるのを崩す），配列にして返す。
def _build_dirichlet_partitions(labels, num_partitions: int, alpha: float, trial: int):
    """
    Dirichlet Non-IID:
    - 各クラス c のサンプル集合を、Dirichlet(alpha) でクライアントへ配分する。
    - alpha が小さいほど「偏りが強い」(例: 0.1 は激偏り、10 はほぼIID)
    """
    rng = np.random.default_rng(trial)
    labels = np.asarray(labels)
    num_classes = int(labels.max()) + 1

    client_indices = [[] for _ in range(num_partitions)]

    for c in range(num_classes):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)

        proportions = rng.dirichlet(alpha=np.full(num_partitions, alpha))
        counts = (proportions * len(idx_c)).astype(int)

        # 端数調整（合計が一致するように）
        diff = len(idx_c) - counts.sum()
        if diff > 0:
            for k in np.argsort(-proportions)[:diff]:
                counts[k] += 1
        elif diff < 0:
            for k in np.argsort(-proportions):
                if diff == 0:
                    break
                if counts[k] > 0:
                    counts[k] -= 1
                    diff += 1

        cursor = 0
        for k in range(num_partitions):
            take = int(counts[k])
            if take <= 0:
                continue
            client_indices[k].extend(idx_c[cursor:cursor + take].tolist())
            cursor += take

    # クライアント内シャッフル
    for k in range(num_partitions):
        rng.shuffle(client_indices[k])

    return [np.asarray(v, dtype=int) for v in client_indices]


# 【_build_dirichlet_equal_partitions の説明】（実験で使っている分割。data-split = "dirichlet_equal"）
#   学習用データを，クライアントの台数ぶんに分ける。返すのは「クライアントごとの，担当する画像の番号の一覧」。
#   上の _build_dirichlet_partitions との違いは，全クライアントの枚数が同じになること。
#
#   なぜ枚数をそろえるのか
#     上の方式では，alpha を小さくすると「持っている数字の偏り」と「枚数の偏り」が同時に強まる。
#     枚数が多いクライアントは，1ラウンドの学習で重みを更新する回数が多いので，更新ベクトルが大きくなる。
#     すると，検知のしやすさが変わったときに，それが数字の偏りのせいなのか，枚数の偏りのせいなのかを区別できない。
#     また，FedAvg は枚数で重み付けして平均するので，悪性クライアントの影響力が「台数の割合」からずれてしまう
#     （20台中4台が悪性でも，枚数の少ない4台なら影響力は 0.2 より小さい）。
#     枚数をそろえると，変わるのは数字の比率だけになり，悪性の影響力も台数の割合（0.2）に一致する。
#
#   考え方
#     1. クライアント1台ごとに，「0〜9 の数字をどんな比率で持つか」を乱数で決める（ディリクレ分布）。
#          例: クライアント A は「3 が 70%，7 が 25%，残りが少し」，クライアント B は「0 が 90%，…」。
#     2. 全クライアントに，1枚ずつ順番に画像を配っていく（A に1枚，B に1枚，…，最後の台まで配ったらまた A に戻る）。
#        配るたびに，そのクライアントの比率に従って「どの数字を配るか」をくじで決める。
#     3. これを samples_per_client 回くり返す。全員がちょうど samples_per_client 枚ずつ持つことになる。
#   1つの数字の画像は約 6,000 枚しかないので，同じ数字を欲しがるクライアントが多いと，途中で在庫がなくなる。
#   在庫のなくなった数字はくじから外し，残っている数字だけで比率を作り直して配る（下の weights の計算）。
#   1台ずつまとめて配るのではなく1枚ずつ順番に配るのは，在庫切れの影響が，後に配られるクライアントだけに
#   集中しないようにするため。
#
#   samples_per_client（1台あたりの枚数）について
#     pyproject.toml の samples-per-client で決める。現在は 1500（20台で 30,000 枚。全 60,000 枚の半分）。
#     全部の画像を配りきる設定（20台なら 3000）にすると，在庫切れが頻繁に起き，決めた比率どおりに配れなくなる
#     （alpha=1 では，平均で約2割の画像が別の数字に差し替わった）。
#     1500 なら，差し替えはほとんど起きない（くじ引きによる自然なずれと同じ程度）。
#
#   alpha の意味（ディリクレ分布のつまみ）
#     alpha が大きい（例: 1000）: どのクライアントの比率も，データ全体の比率とほぼ同じになる（ほぼ IID）。
#     alpha が小さい（例: 1）   : 比率が極端に偏る。1台のクライアントが，1〜2種類の数字ばかりを持つ。
#   ディリクレ分布に渡す値は alpha * prior（prior はデータ全体での各数字の割合，約 0.1 ずつ）。
#   Hsu et al. (2019) の式 q ~ Dir(alpha * p) をそのまま使っている。数字1つあたりの値は約 alpha / 10 になる。
#   ※ 以前の方式 "dirichlet" は，数字1つあたりの値が alpha そのもの（alpha / 10 ではない）。
#     2つの方式のあいだで alpha の値を比べるときは，この違いに注意すること。
#
#   コードの読み方
#     rng = np.random.default_rng(trial) : trial 番号を種にした乱数の発生器。同じ trial なら毎回同じ分け方になる。
#     if samples_per_client * num_partitions > len(labels): raise ValueError(...)
#         : 配る枚数の合計が，手持ちの画像の枚数を超えていたら，エラーにして止める。
#     pools           : 数字ごとの「在庫」。pools[c] は，数字 c の画像の番号を，かき混ぜて並べた一覧。
#     remaining       : 数字ごとの，在庫の残り枚数。配るたびに1ずつ減らす。
#     prior           : データ全体での，各数字の割合（合計は 1）。
#     rng.dirichlet(値の並び, size=num_partitions)
#         : クライアントの台数ぶん，比率を引く。class_ratios[k] が，クライアント k の比率（10個の値で，合計は 1）。
#     weights = class_ratios[k] * (remaining > 0)
#         : (remaining > 0) は「在庫があれば True（1），なければ False（0）」の並び。
#           掛け算すると，在庫のない数字の比率だけが 0 になる。
#     if weights.sum() <= 0: weights = (remaining > 0).astype(float)
#         : そのクライアントが欲しい数字がすべて在庫切れのときは，在庫のある数字から均等に選ぶ。
#     rng.choice(num_classes, p=weights / weights.sum())
#         : 0〜9 のどれかを，p の確率で1つ選ぶ。weights を合計で割って，合計が 1 になるようにしている。
#     remaining[c] -= 1 ; pools[c][remaining[c]]
#         : 在庫を1減らし，在庫の一覧の後ろから1枚取り出す。同じ画像が2回配られることはない。
def _build_dirichlet_equal_partitions(
    labels, num_partitions: int, alpha: float, trial: int, samples_per_client: int
):
    """
    Dirichlet Non-IID（データ数は全クライアントで同じ）:
    - クライアントごとに「クラスの比率」を Dirichlet(alpha * prior) から引き，その比率で samples_per_client 枚を配る。
    - 偏るのはラベルの比率だけで，枚数は全員同じ（ラベル分布の偏りのみ。Hsu et al. (2019) の考え方）。
    """
    rng = np.random.default_rng(trial)
    labels = np.asarray(labels)
    num_classes = int(labels.max()) + 1

    if samples_per_client * num_partitions > len(labels):
        raise ValueError(
            f"samples-per-client × クライアント数 = {samples_per_client * num_partitions} 枚が，"
            f"学習用データの枚数（{len(labels)} 枚）を超えています。samples-per-client を小さくしてください。"
        )

    # 数字ごとの在庫（画像の番号をかき混ぜたもの）と，その残り枚数
    pools = []
    for c in range(num_classes):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        pools.append(idx_c)
    remaining = np.array([len(pool) for pool in pools])
    prior = remaining / remaining.sum()

    # クライアントごとのクラス比率（行 = クライアント，列 = クラス）
    class_ratios = rng.dirichlet(alpha * prior, size=num_partitions)

    # 全クライアントに1枚ずつ順番に配る
    client_indices = [[] for _ in range(num_partitions)]
    for _ in range(samples_per_client):
        for k in range(num_partitions):
            weights = class_ratios[k] * (remaining > 0)  # 在庫切れのクラスは選ばない
            if weights.sum() <= 0:
                weights = (remaining > 0).astype(float)
            c = int(rng.choice(num_classes, p=weights / weights.sum()))
            remaining[c] -= 1
            client_indices[k].append(int(pools[c][remaining[c]]))

    return [np.asarray(v, dtype=int) for v in client_indices]


# 【_get_partitions の説明】
#   分割を取得する窓口。同じ条件の分割が作成済みならそれを返し，なければ作って控えを取る。
#     key = (num_partitions, data_split, float(alpha), trial, samples_per_client)
#         : 分割の条件をひとまとめにしたもの（タプル）。これを辞書の「名前」として使う。
#     if key in _PARTITION_CACHE: return _PARTITION_CACHE[key]
#         : 同じ条件で作ったことがあれば，しまってあるものを返して終わる。
#     分割は毎ラウンド・毎クライアントで必要になるが，作るのに時間がかかるので，1回作ったら使い回している。
#     条件に trial が入っているので，trial が違えば別の分割になる。
#   全クライアントが同じ条件（台数・分け方・alpha・trial・1台あたりの枚数）でこの関数を呼ぶので，
#   全員が同じ分け方を再現でき，担当が重なることはない。
#   data_split が "dirichlet_equal" なら枚数をそろえる分割，"dirichlet" なら以前の分割を使う。
#   samples_per_client（1台あたりの枚数）を使うのは "dirichlet_equal" だけ。
def _get_partitions(
    num_partitions: int, data_split: str, alpha: float, trial: int, samples_per_client: int
):
    """設定に応じた partitions を作成/取得する。

    data_split:
        "dirichlet_equal" : クライアントごとにクラス比率を Dirichlet(alpha * prior) から引く。枚数は全員 samples_per_client 枚
                            （ラベル分布の偏りのみ。実験で使う）。
        "dirichlet"       : クラスごとに Dirichlet(alpha) で各クライアントへ配分する。枚数も偏る（以前の方式）。
                            samples_per_client は使わない。
        "iid" / "non_iid"（Fang方式）は unused_features/task_extra.py に移してある。
    """
    train_ds = _load_train_dataset()
    labels = train_ds["label"]

    key = (num_partitions, data_split, float(alpha), trial, samples_per_client)
    if key in _PARTITION_CACHE:
        return _PARTITION_CACHE[key]

    if data_split == "dirichlet_equal":
        parts = _build_dirichlet_equal_partitions(
            labels,
            num_partitions=num_partitions,
            alpha=float(alpha),
            trial=trial,
            samples_per_client=int(samples_per_client),
        )
    elif data_split == "dirichlet":
        parts = _build_dirichlet_partitions(labels, num_partitions=num_partitions, alpha=float(alpha), trial=trial)
    else:
        raise ValueError(
            f"Unknown data_split: {data_split}（使えるのは 'dirichlet_equal' / 'dirichlet'。"
            "'iid' / 'non_iid' は unused_features/task_extra.py に移してあります）"
        )

    _PARTITION_CACHE[key] = parts
    return parts


# 【load_data の説明】
#   1台のクライアントが使うデータを用意する。client_app.py が毎ラウンド呼んでいる。
#   返すのは（学習用, 評価用）の2つの DataLoader。
#
#   引数の並びにある * : これより後ろの引数は，名前=値 の形で渡す決まり（順番の取り違えを防ぐ）。
#
#   処理
#     1. 学習用データ全体と，分割（誰がどの画像を担当するか）を用意する。
#     2. if partition_id < 0 or partition_id >= num_partitions:
#            クライアント番号が範囲外ならエラーにする。
#     3. parts[partition_id]              : 自分の担当ぶんの画像の番号の一覧。
#        train_ds.select(番号のリスト)    : 全体から，その番号の画像だけを取り出す。
#     4. partition.train_test_split(test_size=0.2, seed=trial)
#            自分の担当ぶんを，さらに 学習用 8割 / 評価用 2割 に分ける。seed を決めているので毎回同じ分け方。
#            クライアントが学習に使うのは，この8割のほう。
#     5. DataLoader(データ, batch_size=..., shuffle=...)
#            データを batch_size 枚ずつ取り出せるようにする。
#            shuffle=True  : 取り出す順番を毎回かき混ぜる（学習用。同じ順番ばかりだと学習が偏るため）。
#            shuffle=False : 順番を変えない（評価用。採点なので順番は関係ない）。
def load_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    *,
    data_split: str,
    alpha: float,
    trial: int,
    samples_per_client: int,
):
    """
    各クライアントが使うデータローダを作る。

    Args:
        partition_id: 何番目のクライアントか（0..num_partitions-1）
        num_partitions: クライアント総数
        batch_size: ミニバッチサイズ
        data_split: データ分割方式（"dirichlet_equal" または "dirichlet"）
        alpha: Dirichlet の濃度パラメータ（小さいほど偏りが強い）
        samples_per_client: クライアント1台に配る枚数（"dirichlet_equal" のときだけ使う）
        trial: 再現性用シード（データ分割と train/test split に使用）

    Returns:
        trainloader, testloader:
            このクライアントに割り当てられたデータを 80% train / 20% test に分けた DataLoader
    """
    train_ds = _load_train_dataset()
    parts = _get_partitions(
        num_partitions=num_partitions,
        data_split=data_split,
        alpha=alpha,
        trial=trial,
        samples_per_client=samples_per_client,
    )

    if partition_id < 0 or partition_id >= num_partitions:
        raise ValueError(f"partition_id out of range: {partition_id}")

    indices = parts[partition_id]
    partition = train_ds.select(indices.tolist())

    partition_train_test = partition.train_test_split(test_size=0.2, seed=trial)

    trainloader = DataLoader(
        partition_train_test["train"],
        batch_size=batch_size,
        shuffle=True,
    )
    testloader = DataLoader(
        partition_train_test["test"],
        batch_size=batch_size,
        shuffle=False,
    )

    return trainloader, testloader


# ============================================================
# 4) サーバ側で使用する「全体テストデータ」
# ============================================================
# 【この部分の説明】
#   サーバが毎ラウンドの採点に使うテストデータ（10,000 枚）を用意する。
#   学習用データとは別に用意されている部分（split="test"）で，分割はしない。
#   どのクライアントも学習に使っていないデータなので，モデルの実力を公平に測れる。
#   論文で使う精度は，このデータでの採点結果（server_app.py の global_evaluate）。
#     .with_format("torch")  : 取り出した値を PyTorch の Tensor の形にする。
#     batch_size=128         : 採点のときに一度に処理する枚数。まとめて処理するための単位で，結果には影響しない。
def load_centralized_dataset():
    test_dataset = load_dataset(_DATASET_NAME, split="test")
    dataset = test_dataset.with_format("torch").with_transform(apply_transforms)
    return DataLoader(dataset, batch_size=128)


# ============================================================
# 5) 計算に使う装置（CPU / GPU）の決定
# ============================================================
# 【この部分の説明】
#   モデルの計算を CPU でするか GPU でするかを，設定（pyproject.toml の device）から決める。
#   サーバ（server_app.py）とクライアント（client_app.py）の両方が，この関数を使う。
#
#   GPU には2種類ある。
#       cuda : NVIDIA の GPU（Windows / Linux）。
#       mps  : Apple Silicon の Mac に内蔵されている GPU。
#   設定に書ける値
#       "auto" : 使える中で一番速いものを自動で選ぶ（cuda -> mps -> cpu の順に探す）。
#       "cuda" / "mps" / "cpu" : その装置を必ず使う。使えない環境ではエラーにする。
#
#   CPU と GPU では，計算の丸め方がわずかに違うので，結果の数値も末尾の桁で変わる。
#   同じ実験の中では装置を混ぜないこと（どの装置で回したかは run_config.txt に記録される）。
#
#   torch.cuda.is_available()         : NVIDIA の GPU が使えれば True。
#   torch.backends.mps.is_available() : Mac の GPU が使えれば True。
#   torch.device("名前")              : 「どの装置で計算するか」を表すもの。model.to(device) のように渡して使う。
DEVICE_CHOICES = ("auto", "cuda", "mps", "cpu")


def resolve_device(name: str) -> torch.device:
    """設定の device（"auto" / "cuda" / "mps" / "cpu"）から，実際に使う装置を決める。"""
    available = {
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
        "cpu": True,
    }
    if name == "auto":
        name = next(candidate for candidate in ("cuda", "mps", "cpu") if available[candidate])
    if name not in available:
        raise ValueError(f"未対応の device です: {name!r}（対応: {DEVICE_CHOICES}）")
    if not available[name]:
        raise ValueError(
            f"device=\"{name}\" が指定されていますが，この環境では使えません。"
            "pyproject.toml の device を \"auto\" か \"cpu\" にするか，GPU を使える環境で実行してください"
            "（NVIDIA の GPU をクライアントで使うには，[tool.sweep] の client-num-gpus の設定も必要です）。"
        )
    if name == "cuda":
        # NVIDIA の GPU では，速さを優先して計算のしかたを実行のたびに変える機能がある。
        # 同じ条件で回し直したときに結果が変わりにくくなるよう，それを止めておく。
        #   deterministic = True : 毎回同じ計算のしかたを使う。
        #   benchmark = False    : 一番速い計算のしかたを実行時に探す機能を止める。
        # ※ これで完全に一致するかどうかは，NVIDIA の GPU のある環境では未確認（manual.tex の「再現性」を参照）。
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return torch.device(name)


# ============================================================
# 6) 学習・評価
# ============================================================
# 【train の説明】
#   モデルを学習させる。クライアントが，自分の手元のデータで実行する。
#     net         : 学習させるモデル（中身の重みが書き換わる）
#     trainloader : 学習用データ
#     epochs      : 手元のデータを何周するか（pyproject.toml の local-epochs）
#     lr          : 学習率。1回の更新で重みをどれだけ動かすかの倍率（pyproject.toml の learning-rate）
#     device      : 計算に使う装置（CPU か GPU）
#   返すのは，学習中の損失の平均。
#
#   準備
#     criterion = torch.nn.CrossEntropyLoss() : 損失関数。モデルの答えが正解からどれだけ外れているかを1つの数値にする。
#                                               小さいほどよい。分類の問題で標準的に使われるもの。
#     optimizer = torch.optim.SGD(net.parameters(), lr=lr)
#         : 重みを更新する係。SGD（確率的勾配降下法）は「損失が減る向きに，学習率のぶんだけ重みを動かす」
#           という一番基本的な方法。net.parameters() は，モデルの全部の重み。
#     net.train() : モデルを学習モードにする。
#
#   学習の繰り返し（バッチ1つごとに，次の4つを行う）
#     optimizer.zero_grad()                  : 前回計算した勾配を消す（消さないと足し込まれてしまう）。
#     loss = criterion(net(images), labels)  : 画像をモデルに入れて答えを出し，正解と比べて損失を計算する。
#     loss.backward()                        : 損失を減らすには各重みをどちらにどれだけ動かせばよいか（勾配）を計算する。
#     optimizer.step()                       : その向きに，重みを実際に動かす。ここでモデルが変わる。
#   これを全バッチぶん，epochs 周くり返す。
#   データが多いクライアントほどバッチの数が多く，重みを動かす回数も多くなる
#   （48 枚のクライアントは2回，11,117 枚のクライアントは278回）。
#
#     batch["img"].to(device) : バッチの画像を，計算に使う装置に載せる。
#     loss.item()             : Tensor になっている損失を，普通の数値として取り出す。
#     running_loss / (len(trainloader) * epochs) : 損失の合計を，バッチの総数で割って平均にする。
#                                                  len(trainloader) は1周あたりのバッチの数。
def train(net, trainloader, epochs, lr, device):
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)  # Fangの式(1) w<-w-η∇ℓ(w;b) に合わせた素朴なSGD

    net.train()
    running_loss = 0.0

    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    avg_trainloss = running_loss / (len(trainloader) * epochs)
    return avg_trainloss


# 【test の説明】
#   モデルを採点する。学習はしない（重みは変えない）。返すのは（損失, 精度）の2つ。
#     net.eval()            : モデルを評価モードにする。
#     with torch.no_grad(): : この中では勾配を計算しない（採点だけなので不要。速く・軽くなる）。
#     outputs = net(images) : 各画像について，0〜9 の点数（10個の数値）が返ってくる。
#     torch.max(outputs.data, 1)[1]
#         : 画像ごとに，点数が一番高い位置（＝モデルが答えた数字）を取り出す。
#           torch.max(表, 1) は（最大値, その位置）の2つを返すので，[1] で位置のほうを取っている。
#     (予測 == labels).sum().item() : 予測と正解が一致した枚数を数える。
#     accuracy = correct / len(testloader.dataset) : 正解した枚数 ÷ 全体の枚数 ＝ 精度（0〜1）。
#     loss = loss / len(testloader)                : バッチごとの損失の平均を，さらにバッチの数で平均したもの。
def test(net, testloader, device):
    net.to(device)
    net.eval()

    criterion = torch.nn.CrossEntropyLoss()
    correct = 0
    loss = 0.0

    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy