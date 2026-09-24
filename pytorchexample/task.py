"""
このファイルの役割
- モデル(Net)の定義（MNIST 用）
- Federated Learning 用データ分割（IID / Non-IID 切替）
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
"""

import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor


# ============================================================
# 1) モデル定義（MNIST 用 CNN）
#    Fang et al. (2020) Table 7a に準拠：
#    Conv(3x3x30)+ReLU -> MaxPool(2x2) -> Conv(3x3x50)+ReLU -> MaxPool(2x2)
#    -> FC(200)+ReLU -> FC(10)（Softmaxは CrossEntropyLoss 内で暗黙的に適用）
#    28x28 入力 -> conv1:26x26x30 -> pool:13x13x30
#              -> conv2:11x11x50 -> pool:5x5x50(=1250) -> FC(1250,200) -> FC(200,10)
# ============================================================
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
# 1b) (旧版・参考用) 全結合DNN
#     Fangの``DNN''は実際にはCNN（Table 7a）だったため Net は CNN に置き換えた。
#     卒論7章までの実験で使用していたモデルはこちら。実験では使用しない。
# ============================================================
class NetFC(nn.Module):
    def __init__(self):
        super(NetFC, self).__init__()

        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


# ============================================================
# 2) データ前処理（PyTorchテンソル化 & 正規化）
#    MNIST はグレースケール1chなので mean/std も1要素にする
#    （MNIST全体の統計量: mean=0.1307, std=0.3081 を使用）
# ============================================================
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
# 3) FL用データ分割（IID / Non-IID 切替）
# ============================================================
# 方針：
# - MNIST(train) を 1回だけ読み込み、各クライアント用 index を作って再利用する
# - IID: 全データをシャッフルして均等分割
# - Non-IID(Dirichlet): クラスごとに Dirichlet 分布で各クライアントへの割当比率を作り、偏りを導入
#
# 注意：
# - data_split / alpha / num_partitions が変わったら index を作り直す必要があるため、キー付きキャッシュにする
# - transforms は Dataset.with_transform(apply_transforms) で遅延適用（メモリ節約）
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


def _build_iid_partitions(labels, num_partitions: int, trial: int):
    rng = np.random.default_rng(trial)
    all_idx = np.arange(len(labels))
    rng.shuffle(all_idx)
    return np.array_split(all_idx, num_partitions)


def _build_fang_probabilistic_partitions(labels, num_partitions: int, p: float, trial: int):
    """
    Fang et al. (2020) 4.1節の Non-IID 生成方式。
    - クライアントをクラス数 L 個のグループへ均等分割する
    - ラベル l の学習データは，対応するグループ l へ確率 p で割り当て，
      残り確率 (1-p) は他の L-1 グループへ均等に振り分ける
    - グループ内では，割り当てられたデータをグループ内の各クライアントへ均等に配分する

    p が大きいほど「そのラベルは対応グループに集中する」＝Non-IID度が高くなる。
    Fangのデフォルトは p=0.5。
    """
    rng = np.random.default_rng(trial)
    labels = np.asarray(labels)
    num_classes = int(labels.max()) + 1
    L = num_classes

    # クライアントを L グループへ均等分割（m=100, L=10 なら1グループ10クライアント）
    client_ids = np.arange(num_partitions)
    groups = np.array_split(client_ids, L)

    # 各ラベルのサンプルを，確率pで対応グループへ・残りを他グループへ均等に割当
    group_indices = [[] for _ in range(L)]
    for l in range(L):
        idx_l = np.where(labels == l)[0]
        rng.shuffle(idx_l)

        if L > 1:
            probs = np.full(L, (1.0 - p) / (L - 1))
            probs[l] = p
        else:
            probs = np.array([1.0])
        probs = probs / probs.sum()

        assigned_groups = rng.choice(L, size=len(idx_l), p=probs)
        for g in range(L):
            group_indices[g].extend(idx_l[assigned_groups == g].tolist())

    # 各グループ内で，割り当てられたサンプルをグループ内クライアントへ均等配分
    client_indices = [[] for _ in range(num_partitions)]
    for g in range(L):
        idx_g = np.asarray(group_indices[g], dtype=int)
        rng.shuffle(idx_g)
        members = groups[g]
        if len(members) == 0:
            continue
        splits = np.array_split(idx_g, len(members))
        for member_id, split in zip(members, splits):
            client_indices[member_id].extend(split.tolist())

    for k in range(num_partitions):
        rng.shuffle(client_indices[k])

    return [np.asarray(v, dtype=int) for v in client_indices]


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


def _get_partitions(num_partitions: int, data_split: str, alpha: float, p: float, trial: int):
    """設定に応じた partitions を作成/取得する。

    data_split:
        "iid"       : 均等分割
        "non_iid"   : Fang et al. (2020) 方式（ラベルlを確率pで対応グループへ割当）。デフォルト。
        "dirichlet" : (旧・参考用) Dirichlet(alpha) によるクラスごとの配分。
    """
    train_ds = _load_train_dataset()
    labels = train_ds["label"]

    key = (num_partitions, data_split, float(alpha), float(p), trial)
    if key in _PARTITION_CACHE:
        return _PARTITION_CACHE[key]

    if data_split == "iid":
        parts = _build_iid_partitions(labels, num_partitions=num_partitions, trial=trial)
        parts = [np.asarray(part, dtype=int) for part in parts]
    elif data_split == "non_iid":
        parts = _build_fang_probabilistic_partitions(labels, num_partitions=num_partitions, p=float(p), trial=trial)
    elif data_split == "dirichlet":
        parts = _build_dirichlet_partitions(labels, num_partitions=num_partitions, alpha=float(alpha), trial=trial)
    else:
        raise ValueError(f"Unknown data_split: {data_split} (expected: 'iid' / 'non_iid' / 'dirichlet')")

    _PARTITION_CACHE[key] = parts
    return parts


def load_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    *,
    data_split: str = "iid",
    alpha: float = 0.5,
    p: float = 0.5,
    trial: int = 1,
):
    """
    各クライアントが使うデータローダを作る（IID / Non-IID 切替対応）。

    Args:
        partition_id: 何番目のクライアントか（0..num_partitions-1）
        num_partitions: クライアント総数
        batch_size: ミニバッチサイズ
        data_split: "iid" / "non_iid"（Fang方式，デフォルト） / "dirichlet"（旧方式，参考用）
        alpha: Dirichlet の濃度パラメータ（data_split="dirichlet"の時のみ使用。小さいほど偏りが強い）
        p: ラベルlを対応クライアント群へ割り当てる確率（data_split="non_iid"の時のみ使用。
           Fang et al. (2020) のデフォルトはp=0.5）
        trial: 再現性用シード（データ分割と train/test split に使用）

    Returns:
        trainloader, testloader:
            このクライアントに割り当てられたデータを 80% train / 20% test に分けた DataLoader
    """
    train_ds = _load_train_dataset()
    parts = _get_partitions(num_partitions=num_partitions, data_split=data_split, alpha=alpha, p=p, trial=trial)

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
def load_centralized_dataset():
    test_dataset = load_dataset(_DATASET_NAME, split="test")
    dataset = test_dataset.with_format("torch").with_transform(apply_transforms)
    return DataLoader(dataset, batch_size=128)


# ============================================================
# 5) 学習・評価
# ============================================================
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