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
# - Non-IID(Dirichlet): クラスごとに Dirichlet 分布で各クライアントへの割当比率を作り、偏りを導入
#   （alpha が大きいとほぼ IID，小さいほど偏りが強い）
#
# 注意：
# - data_split / alpha / num_partitions が変わったら index を作り直す必要があるため、キー付きキャッシュにする
# - transforms は Dataset.with_transform(apply_transforms) で遅延適用（メモリ節約）
#
# 【この部分の説明】
#   Non-IID（クライアントごとにデータが偏っている状態）を作っているのは，ここ。
#   研究の横軸である「Non-IID 度」は，下の _build_dirichlet_partitions の alpha で決まる。
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


# 【_build_dirichlet_partitions の説明】
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


# 【_get_partitions の説明】
#   分割を取得する窓口。同じ条件の分割が作成済みならそれを返し，なければ作って控えを取る。
#     key = (num_partitions, data_split, float(alpha), trial)
#         : 分割の条件をひとまとめにしたもの（タプル）。これを辞書の「名前」として使う。
#     if key in _PARTITION_CACHE: return _PARTITION_CACHE[key]
#         : 同じ条件で作ったことがあれば，しまってあるものを返して終わる。
#     分割は毎ラウンド・毎クライアントで必要になるが，作るのに時間がかかるので，1回作ったら使い回している。
#     条件に trial が入っているので，trial が違えば別の分割になる。
#   全クライアントが同じ条件（台数・alpha・trial）でこの関数を呼ぶので，全員が同じ分け方を再現でき，
#   担当が重なったり抜けたりしない。
def _get_partitions(num_partitions: int, data_split: str, alpha: float, trial: int):
    """設定に応じた partitions を作成/取得する。

    data_split:
        "dirichlet" : Dirichlet(alpha) によるクラスごとの配分（現在はこれだけ）。
                      "iid" / "non_iid"（Fang方式）は unused_features/task_extra.py に移してある。
    """
    train_ds = _load_train_dataset()
    labels = train_ds["label"]

    key = (num_partitions, data_split, float(alpha), trial)
    if key in _PARTITION_CACHE:
        return _PARTITION_CACHE[key]

    if data_split == "dirichlet":
        parts = _build_dirichlet_partitions(labels, num_partitions=num_partitions, alpha=float(alpha), trial=trial)
    else:
        raise ValueError(
            f"Unknown data_split: {data_split}（現在使えるのは 'dirichlet' だけ。"
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
):
    """
    各クライアントが使うデータローダを作る。

    Args:
        partition_id: 何番目のクライアントか（0..num_partitions-1）
        num_partitions: クライアント総数
        batch_size: ミニバッチサイズ
        data_split: データ分割方式（現在は "dirichlet" だけ）
        alpha: Dirichlet の濃度パラメータ（小さいほど偏りが強い）
        trial: 再現性用シード（データ分割と train/test split に使用）

    Returns:
        trainloader, testloader:
            このクライアントに割り当てられたデータを 80% train / 20% test に分けた DataLoader
    """
    train_ds = _load_train_dataset()
    parts = _get_partitions(num_partitions=num_partitions, data_split=data_split, alpha=alpha, trial=trial)

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