"""
このファイルは以下を担当する：
- CNNモデルの定義
- Federated Learning 用データ分割
- 学習・評価ループ
"""

import torch
import torch.nn as nn
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor


# CNNモデルをつくる（CIFAR-10 用）
#アプリケーション6の授業資料に記載あり.
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        #畳み込み層1.Conv2dが畳み込み層を生成するAPI.
        #nn.Conv2d(画像のチャンネル数,カーネル数,カーネルのサイズ)
        #RGB画像を入力として、5×5×3(ch) のカーネルを 6 種類使い、画像の局所的特徴を 6 通りの見方で抽出する層を作る.
        #in_channnelが3,out_channnelが6,カーネルサイズが5
        self.conv1 = nn.Conv2d(3, 6, 5)

        # プーリング層(nn.MaxPool2d(領域サイズ,ストライド数))
        #MaxPool2dは分割した領域の最大値を採用する方法.
        #つまり領域サイズ2で2だけストライドする(ピクセルを移動する)ということ
        self.pool = nn.MaxPool2d(2, 2)

        #in_channelが6,out_channelが16,カーネルサイズが5.
        #self.conv1でout_channelが6だったため，チャンネルを合わせるためにin_channelが6
        self.conv2 = nn.Conv2d(6, 16, 5)

        # 全結合層
        #nn.Linear(入力データの次元,出力データの次元)
        #（conv2 と pool などを経て）最終的に特徴マップが 16ch・5×5 になり,それを120次元に落としている.
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        #120次元を84次元に落としている.
        self.fc2 = nn.Linear(120, 84)

        #84次元を10次元(10クラス)に落としている.
        self.fc3 = nn.Linear(84, 10)

        # 活性化関数(ReLU)とドロップアウト
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    #ここで各層を連結してCNNを構築する.(アプリケーション開発6資料8-2)
    def forward(self, x):
        #self.conv1で局所特徴を抽出し,それをReLU関数を用いて非線形化し,pool処理でサイズを圧縮している.
        #資料ではここを3行で分けて書いている.
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        # 畳み込み層で抽出した空間的特徴をすべて並べて、全結合層が扱える 1 次元の特徴ベクトルに変換している。
        x = x.view(-1, 16 * 5 * 5)

        # 全結合層
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))

        # 最終出力
        return self.fc3(x)


#グローバル変数
#FederatedDataset を毎回新しく作ると処理が重いので、一度作ったものを保存しておき、次からはそれを再利用するためにグローバル変数 fds を用意している。
fds = None



# 画像前処理（PyTorch用）
# PyTorch のテンソル形式に変換し、画素値のスケールを学習しやすい範囲（だいたい -1〜1）にそろえる前処理を定義している。
pytorch_transforms = Compose([
    ToTensor(), 
    Normalize(#正規化
        (0.5, 0.5, 0.5),  # 平均
        (0.5, 0.5, 0.5)   # 標準偏差
    )  # → おおよそ [-1, 1] に正規化
])


def apply_transforms(batch):#バッチを受け取る関数を定義
    """
    batch[img]に入っている画像リストをfor img in batch["img"]で一枚ずつ取り出し,その取り出したものを
    pytorch_transform(img)で画像を[-1,1]に正規化し,その結果を元のbatch["img"]に上書きをするという処理.
    つまり,与えらた画像をPyTorchが学習に使える形に整形する処理をここでしている.
    """
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch




# クライアント1人分の学習環境を用意する関数
#Federated Learningではデータを分割して分散学習をする必要がある.その処理.
def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """
    partition_id:何番目のクライアントか.
    num_partitions:クライアントの総数
    ミニバッチサイズ
    これらを定義する.
    """

    

    global fds

    # もしFederatedDatasetを持っていなかったら,
    #IIDになるようにnum_partition(参加者数)で分割する.
    #先行研究では非IIDデータの取り扱いが問題として提起されていた.

    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)

        fds = FederatedDataset(#ここでデータセットを作っている.
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )

    # 指定クライアントのデータを取得する.
    #全体を分割したデータからpartition_id(クライアントの各番号)が割り当てられたミニデータ(ミニバッチ)を取得し,partitionに入れる.
    partition = fds.load_partition(partition_id)

    # クライアント内で 80% train / 20% test に分割
    #そしてそのクライアントに渡すようのデータ（partion）をテスト用(20%)と訓練用(80%)に分割する.
    partition_train_test = partition.train_test_split(
        test_size=0.2,
        seed=42#ここで乱数の固定をしている.これがないと,毎回違うデータで分割されるため,結果が変わってくる.(重要)
    )

    # 前処理を適用(ここあとで)
    partition_train_test = partition_train_test.with_transform(apply_transforms)

    # DataLoader を用いてテストデータを読み込む(第3回の授業資料に記載あり.)
    #バッチサイズは授業同様25にしてみる.(とりあえず)
    trainloader = DataLoader(#train(訓練用)
        partition_train_test["train"],
        batch_size=25,
        shuffle=True
    )

    testloader = DataLoader(#test(検証用)
        partition_train_test["test"],
        batch_size=25,
        shuffle=False
    )

    return trainloader, testloader









"""
Federated Learning では、
各クライアントが自分に割り当てられたローカルデータのみを用いてモデルの学習を行う。
一方でサーバは、クライアントから集約したモデルを共通の基準で評価する役割を担い、
そのために全体の test データを用いる場合がある。クライアントとサーバは同じモデル構造や前処理を用いるが、
目的は学習と集約・評価で明確に分かれている。
それを踏まえて以下でサーバ側で使うテストデータを作成する.
"""
# サーバ側で使用する「全体テストデータ」(グローバルモデルの評価用)
def load_centralized_dataset():
    test_dataset = load_dataset("uoft-cs/cifar10", split="test")#cifar10をすべて読み込む

    dataset = (
        test_dataset
        .with_format("torch")
        .with_transform(apply_transforms)
    )#※クライアント側はFL用のFederatedDatasetだが,サーバー側は生のdatasets.Datasetなので,PyTorch で扱える形式（Tensor）として取り出せるように指定し,
    #apply_transformsで正規化を別途行う必要がある.
    #バッチサイズ128で順番に取り出す.
    return DataLoader(dataset, batch_size=128)



# 学習処理（クライアント側）
# この関数は、1つのクライアントが自分のローカルデータだけを使ってCNNモデルを指定されたエポック数だけ学習し、その学習中に得られた平均損失を返す処理である
def train(net, trainloader, epochs, lr, device):
    """
    train の Docstring
    
    :param net: 学習するCNNモデル
    :param trainloader: このクライアント専用の学習データ
    :param epochs:データ全体を何周学習するか
    :param lr: 学習率
    """
    net.to(device)
    #多クラス分類用の損失関数（交差エントロピー）を定義
    criterion = torch.nn.CrossEntropyLoss().to(device)#criterionは評価基準の意
    #確率的勾配降下法（SGD）を用いた最適化手法.SGDからあとで変えてもいい.(1/30:SGDからAdamWに変更)
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=lr,
        #momentum=0.9
    )
    #学習モードに切り替える.
    net.train()
    running_loss = 0.0#損失を蓄積させる変数の初期化

    for _ in range(epochs):#エポックのループ
        for batch in trainloader:#trainloaderから1バッチずつデータを取り出す.
            images = batch["img"].to(device)
            labels = batch["label"].to(device)



            #勾配のリセット
            optimizer.zero_grad()
            #予測結果と正解ラベルから損失を計算
            loss = criterion(net(images), labels)
            #誤差逆伝播により、各パラメータの勾配を計算
            loss.backward()
            optimizer.step()

            # 全エポック分のlossを累積
            running_loss += loss.item()

    #損失の和(running_loss)をtrainloader*epochで割り,1エポックあたりの平均学習損失を求めている.
    avg_trainloss = running_loss / (len(trainloader) * epochs)

    return avg_trainloss



#モデルの性能評価(テストデータでモデル性能を評価する関数)
def test(net, testloader, device):
    

    net.to(device)
    #1/30:net.eval()を追加する.Pytorchには2つのモデルがあり,net.trainだとdropoutが有効,net.evalだとdropoutが無効になる.
    net.eval()
    #多クラス分類用の損失関数（交差エントロピー）を定義
    criterion = torch.nn.CrossEntropyLoss()
    #変数の初期化
    correct = 0
    loss = 0.0


    with torch.no_grad():#評価時には勾配計算を行わないようにしている.
        for batch in testloader:
            #画像データと正解ラベルを計算デバイスに転送する
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            #モデルに画像を入力し、各クラスに対する予測値（スコア）を得る。
            outputs = net(images)
            #損失計算,各データについて,最も高いクラスを予測ラベルとして選び,正解ラベルと一致した数をカウントする.
            loss += criterion(outputs, labels).item()
            correct += (
                torch.max(outputs.data, 1)[1] == labels
                ).sum().item()

    # 正解率
    accuracy = correct / len(testloader.dataset)

    # バッチ平均lossを平均を取る
    loss = loss / len(testloader)

    return loss, accuracy

#ログとかファイル出力を行うプログラムを加えたいところ.(1/29)
#1/29,ログのcsvファイルの作成プログラムは成功.
#用いた画像の生成結果に関するファイル出力があった方がいいかも(1/29)