
import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from pytorchexample.log_output import log_client_metrics, print_round_summary
from pytorchexample.FL_assignment import Net, load_data
from pytorchexample.FL_assignment import test as test_fn
from pytorchexample.FL_assignment import train as train_fn

# クライアントアプリ
app = ClientApp()

#サーバから受信したグローバルモデルを用いてローカルデータで学習および評価を行い，その結果をサーバへ返却する
@app.train()
def train(msg: Message, context: Context):
   

    # Load the model and initialize it with the received weights
    model = Net()#CNNモデルの構造を作成
    #サーバーから受け取った重みarraysをPytorch形式に変換して作ったモデルに読み込む.
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    #クライアントの環境に応じて使用デバイスを変える.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #モデルをそのデバイスに移動させる.
    model.to(device)

    # データの読み込み
    #このクライアントが 何番目のクライアントか を取得する.分割自体はFL_assignment.pyでやっている.
    #割り当てられたidに対応するデータを用いる
    partition_id = context.node_config["partition-id"]
    #全クライアント数を取得する.
    num_partitions = context.node_config["num-partitions"]
    #学習時に使う ミニバッチサイズを取得する.サーバ側で決められたバッチサイズが渡される.
    batch_size = context.run_config["batch-size"]
    #idに対応するクライアントデータの読み込み.それをDataloaderに変換する.
    #FLでは学習をしたら終わりなので,戻り値は捨てる.(_)の状態にする.
    trainloader, _ = load_data(partition_id, num_partitions, batch_size)

    # Call the training function
    train_loss = train_fn(
        model,#サーバから受け取った 最新のグローバルモデル
        trainloader,#このクライアントに割り当てられた ローカル学習データ
        context.run_config["local-epochs"],#ローカルで何エポック学習するか
        msg.content["config"]["lr"],#サーバが決めて、クライアントに配布している値.サーバ側で決めている.
        device,#CPU,GPUとか
    )

    # ===== 追加：ログ出力（クライアント側 train）=====
    # server_round はサーバ側から config に載せて送られてくる想定。
    # もし無ければ -1 になり、あとでサーバ側を修正すれば正しいroundが入る。
    server_round = msg.content.get("config", {}).get("server_round", -1)

    log_client_metrics(
        round_=server_round,
        client_id=partition_id,
        train_loss=train_loss,
        num_examples=len(trainloader.dataset),
    )
    print_round_summary(
        round_=server_round,
        train_loss=train_loss,
    )
    #memo:各クライアントのラウンド数は把握できないのでこれ改善の余地あり.
    #今回はアプリケーション開発課題用なのでとりあえずサーバーの損失関数と正確性が読み取れれば大丈夫.
    # ===== 追加ここまで =====

    # ローカル学習の結果を、サーバに返すための処理をここで行う.
    #クライアントで学習し終わったモデルの重みstate_dictを取り出し,Flowerが通信に使える形式であるArrayRecordに変換する.
    #model_record = ArrayRecord(model.state_dict())
    # ===== サーバへ返す直前に CPU に戻す（重要）=====
    state_dict_cpu = {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
    }
    model_record = ArrayRecord(state_dict_cpu)
    # ===== ここまで =====

    #学習に関するメタデータなど.num-examplesはこのクライアントが学習に使ったデータ数.FedAvg では データ数に応じた重み付き平均を取るために使われる.
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    #上記のメタデータをFlowerが理解できる評価指標用の形式に変換する.
    metric_record = MetricRecord(metrics)
    #サーバーに返すデータを1つ(content)にまとめている.
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    #今回の学習結果をサーバーから送られてきたメッセージへの返信として返す.
    return Message(content=content, reply_to=msg)

'''
これにより,サーバは
各クライアントの重みを受信し,num-examples を考慮して集約(FedAvg)
学習状況をログとして保存が可能になる.
'''

#サーバから「今のモデルを、このクライアントの検証データで評価して結果だけ返してほしい」と依頼されたときに呼ばれる関数
@app.evaluate()
def evaluate(msg: Message, context: Context):
    

    #さっきと同じ.CNNモデル構造ののNet()を作成してサーバーから受け取った重みarraysをPytorch形式に変換して作ったモデルに読み込む
    #それを対応する機器処理させる.
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    #ここもデータをとってきているだけで説明は変わりないので省略
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    #def evaluateは評価用なので訓練用データは捨て,検証用であるvalloaderだけを受け取るようにする.ここがさっきと違う.
    _, valloader = load_data(partition_id, num_partitions, batch_size)

    # サーバから受け取ったグローバルモデルを使って,このクライアントの検証データで評価を行い、損失と正解率を計算している。
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    #ログ出力（クライアント側 evaluate）(追加部分)
    # train と同様に server_round を取得（無ければ -1）
    server_round = msg.content.get("config", {}).get("server_round", -1)

    log_client_metrics(
        round_=server_round,
        client_id=partition_id,
        eval_loss=eval_loss,
        eval_acc=eval_acc,
        num_examples=len(valloader.dataset),
    )
    print_round_summary(
        round_=server_round,
        eval_loss=eval_loss,
        eval_acc=eval_acc,
    )
    # メモ
    #メタ情報
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    #上記のメタデータをFlowerが理解できる評価指標用の形式に変換する.
    metric_record = MetricRecord(metrics)
    #サーバーに返すデータを1つ(content)にまとめている.
    content = RecordDict({"metrics": metric_record})

    
    #今回の学習結果をサーバーから送られてきたメッセージへの返信として返す.
    return Message(content=content, reply_to=msg)
