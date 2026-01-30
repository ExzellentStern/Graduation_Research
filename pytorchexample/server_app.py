"""pytorchexample: A Flower / PyTorch app."""
#以下,結果をわかりやすくするためにインポートするやつ
import os
import csv
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid


# ここまで
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from pytorchexample.log_output import log_server_metrics, print_round_summary, log_run_config
from pytorchexample.FL_assignment import Net, load_centralized_dataset, test

app = ServerApp()

# ここ各ラウンドごとのラベルづけされた画像をファイルに出力するためのプログラム

_FIXED_IMAGES_CPU = None  # shape: (N,3,32,32)  正規化済み
_FIXED_LABELS_CPU = None  # shape: (N,)

def _unnormalize(imgs: torch.Tensor) -> torch.Tensor:
    """Normalize((0.5),(0.5)) を逆変換して [0,1] に戻す（保存・可視化用）"""
    return torch.clamp(imgs * 0.5 + 0.5, 0.0, 1.0)

def _get_fixed_batch(n_images: int = 32):
    """
    テストデータローダの「先頭から n_images 枚」を固定セットとして確保する。
    1回目だけロードしてCPUにキャッシュ、以降は同じ画像を使う。
    """
    global _FIXED_IMAGES_CPU, _FIXED_LABELS_CPU
    if _FIXED_IMAGES_CPU is not None:
        return _FIXED_IMAGES_CPU, _FIXED_LABELS_CPU

    # load_centralized_dataset() は shuffle=False のはずなので、先頭バッチは毎回同じ
    dl = load_centralized_dataset()
    batch = next(iter(dl))
    images = batch["img"][:n_images].detach().cpu()
    labels = batch["label"][:n_images].detach().cpu()

    _FIXED_IMAGES_CPU = images
    _FIXED_LABELS_CPU = labels
    return _FIXED_IMAGES_CPU, _FIXED_LABELS_CPU

def _save_round_fixed_eval(
    server_round: int,
    images: torch.Tensor,
    labels: torch.Tensor,
    preds: torch.Tensor,
    confs: torch.Tensor,
    out_root: str = "logs/fixed_eval",
):
    """
    ラウンドごとの固定セット推論結果を保存する。
    - preds.csv: index/true/pred/conf
    - grid.png: 画像一覧
    - mis/*.png: 誤分類画像だけ、ファイル名に true/pred/conf を埋め込む（ラベル付き画像）
    """
    out_dir = os.path.join(out_root, f"round_{server_round:03d}")
    os.makedirs(out_dir, exist_ok=True)

    # CSV 保存
    csv_path = os.path.join(out_dir, "preds.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "true_label", "pred_label", "confidence"])
        for i in range(labels.size(0)):
            w.writerow([
                i,
                int(labels[i].item()),
                int(preds[i].item()),
                float(confs[i].item()),
            ])

    # 画像grid保存（固定画像なので毎ラウンド同じだが、比較の基準として置いておくと便利）
    vis = _unnormalize(images.detach().cpu())
    grid = make_grid(vis, nrow=8)
    save_image(grid, os.path.join(out_dir, "grid.png"))

    # 誤分類画像のみ、ラベル入りファイル名で保存（←これが“ラベルづけされた画像”）
    mis_dir = os.path.join(out_dir, "mis")
    os.makedirs(mis_dir, exist_ok=True)

    labels_cpu = labels.detach().cpu()
    preds_cpu = preds.detach().cpu()
    confs_cpu = confs.detach().cpu()

    for i in range(labels_cpu.size(0)):
        t = int(labels_cpu[i].item())
        p = int(preds_cpu[i].item())
        if p != t:
            c = float(confs_cpu[i].item())
            filename = f"{i:03d}_true{t}_pred{p}_conf{c:.3f}.png"
            save_image(vis[i], os.path.join(mis_dir, filename))

#ここまで
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
    print("optimizer      : AdamW")
    print("data_split     : IID")#非IIDは研究内容
    print("seed           : 42")
    print("================================")

    # 既存：ファイル保存用ログ（JSON / CSV想定）
    log_run_config(
        rounds=num_rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=lr,
        optimizer="AdamW",
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
    # CNNモデルを新しく生成
    model = Net()

    # Flower形式の重みarraysをPyTorchのstate_dictに変換し，モデルに読み込む
    # これにより集約後のグローバルモデルができる
    model.load_state_dict(arrays.to_torch_state_dict())

    # 評価を行う計算デバイスを設定
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ★評価モード（Dropoutなどを無効化してブレを減らす）
    model.eval()

    # サーバが持っている 全体のテストデータを読み込む
    test_dataloader = load_centralized_dataset()

    # グローバルモデルを使ってテストデータに対する予測を行い
    # 平均損失(test_loss)と正解率(test_acc)を算出
    test_loss, test_acc = test(model, test_dataloader, device)

    # ここから追加：固定画像セットの予測をラウンドごとに保存
    # 固定画像セット（例：32枚）を毎回同じものとして使い、
    # その予測ラベル/真のラベル/自信度をラウンド別に保存する
    fixed_images_cpu, fixed_labels_cpu = _get_fixed_batch(n_images=32)
    fixed_images = fixed_images_cpu.to(device)
    fixed_labels = fixed_labels_cpu.to(device)

    with torch.no_grad():
        logits = model(fixed_images)
        probs = F.softmax(logits, dim=1)
        confs, preds = torch.max(probs, dim=1)

    _save_round_fixed_eval(
        server_round=server_round,
        images=fixed_images,
        labels=fixed_labels,
        preds=preds,
        confs=confs,
        out_root="logs/fixed_eval",
    )
    # ===== 追加ここまで =====

    #ログ出力（サーバ側 evaluate）
    log_server_metrics(server_round, test_acc, test_loss)
    print_round_summary(server_round, accuracy=test_acc, loss_func=test_loss)
    

    # 評価結果を Flower 用の形式で返す
    return MetricRecord({"accuracy": test_acc, "loss_func": test_loss})

