"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from pytorchexample.log_output import log_server_metrics, print_round_summary, log_run_config
from pytorchexample.FL_assignment import Net, load_centralized_dataset, test

app = ServerApp()

#サーバーのメイン処理
@app.main()
def main(grid: Grid, context: Context) -> None:
    """
    gridがクライアントとの通信や分散実行を管理する仕組み
    contextには実験設定（ラウンド数・学習率など）が入っている
    """

    #実験設定の読み込み
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
#ここから(変更箇所)
    local_epochs: int = context.run_config["local-epochs"]
    batch_size: int = context.run_config["batch-size"]

    print("=== Experiment Configuration ===")
    print("rounds         :", num_rounds)
    print("fraction_eval  :", fraction_evaluate)
    print("local_epochs   :", local_epochs)
    print("batch_size     :", batch_size)
    print("learning_rate  :", lr)
    print("optimizer      : SGD")
    print("data_split     : IID")
    print("seed           : 42")
    print("================================")

    # 既存：ファイル保存用ログ（JSON / CSV想定）
    log_run_config(
        rounds=num_rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=lr,
        optimizer="SGD",
        iid=True,
        seed=42,
    )
    #ここまで

    # グローバルモデルの初期化
    #CNNモデルNet()を生成,その重みであるstate_dictをArrayRecordに変換
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    #ここで FedAvg（Federated Averaging） を使用すると宣言
    """
    FedAvgの流れは以下↓
    各クライアントがローカル学習
    重みをサーバーに送信
    サーバーが平均を取る
    新しいグローバルモデルを配信
    """
    strategy = FedAvg(fraction_evaluate=fraction_evaluate)

    # ここがめっちゃ重要なFLの動作
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,#初期グローバルモデル
        train_config=ConfigRecord({"lr": lr}),#クライアントに渡す学習設定
        num_rounds=num_rounds,#サーバラウンド数
        evaluate_fn=global_evaluate,#各ラウンド後に呼ばれる評価関数
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    #Federated Learningの全ラウンド終了後に得られた最終的なグローバルモデルの重みであるresult.arraysを
    #to_torch_state_dictでFlower用の配列型式からPyTorch の state_dict 形式に変換する.
    #変換したものをfinal_model.ptに保存する.パラメータとかがファイルに入ってるが,人が読むようなファイルではない.
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")

#Federated Learning の各サーバラウンド終了後に呼ばれる「評価係」を定義している
def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:

    #CNNモデルを新しく生成
    model = Net()
    #Flower形式の重みarraysをPytorchのstate_dictに変換し,モデルに読み込む
    #これにより集約後のグローバルモデルができる
    model.load_state_dict(arrays.to_torch_state_dict())
    #評価を行う計算デバイスを設定
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # サーバが持っている 全体のテストデータを読み込む
    test_dataloader = load_centralized_dataset()

    # グローバルモデルを使ってテストデータに対する予測を行い平均損失(test_loss)と正解率(test_acc)を算出.
    test_loss, test_acc = test(model, test_dataloader, device)

    # ===== 追加：ログ出力（サーバ側 evaluate）=====
    log_server_metrics(server_round, test_acc, test_loss)
    print_round_summary(server_round, accuracy=test_acc, loss_func=test_loss)
    # ===== 追加ここまで =====

    #評価結果を Flower 用の形式で返す
    return MetricRecord({"accuracy": test_acc, "loss_func": test_loss})


