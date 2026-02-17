"""
このファイルの役割（Server側）：

- Federated Learning の全体制御
- グローバルモデルの初期化
- クライアントへの学習設定配布（train_config）
- FedAvg による集約
- 各ラウンド後のグローバル評価
- 固定画像による可視化ログ出力
- 最終モデル保存

※ 攻撃設定（attack_type 等）や IID/non-IID 設定はここからクライアントへ配布される。
"""

# ============================================================
# 可視化・ログ用ライブラリ
# ============================================================
import os
import csv
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid

# ============================================================
# Flower / PyTorch 関連
# ============================================================
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from pytorchexample.log_output import (
    log_server_metrics,
    print_round_summary,
    log_run_config,
)
from pytorchexample.task import Net, load_centralized_dataset, test


app = ServerApp()

# ============================================================
# 固定評価用データ（毎ラウンド同じ入力を使う）
# ============================================================
_FIXED_IMAGES_CPU = None
_FIXED_LABELS_CPU = None


def _unnormalize(imgs: torch.Tensor) -> torch.Tensor:
    """[-1,1] 正規化を [0,1] に戻す（画像保存用）"""
    return torch.clamp(imgs * 0.5 + 0.5, 0.0, 1.0)


def _get_fixed_batch(n_images: int = 32):
    """
    テストデータの先頭から n_images 枚を固定評価セットとして使用。
    1度だけ読み込み、以降はキャッシュを使う。
    """
    global _FIXED_IMAGES_CPU, _FIXED_LABELS_CPU

    if _FIXED_IMAGES_CPU is not None:
        return _FIXED_IMAGES_CPU, _FIXED_LABELS_CPU

    dl = load_centralized_dataset()
    batch = next(iter(dl))

    images = batch["img"][:n_images].detach().cpu()
    labels = batch["label"][:n_images].detach().cpu()

    _FIXED_IMAGES_CPU = images
    _FIXED_LABELS_CPU = labels

    return _FIXED_IMAGES_CPU, _FIXED_LABELS_CPU


# ============================================================
# 固定評価結果の保存
# ============================================================
def _save_round_fixed_eval(
    server_round: int,
    images: torch.Tensor,
    labels: torch.Tensor,
    preds: torch.Tensor,
    confs: torch.Tensor,
    out_root: str = "logs/fixed_eval",
):
    """
    各ラウンドでの固定画像に対する予測結果を保存する。

    保存内容：
    - preds.csv（予測結果一覧）
    - grid.png（画像一覧）
    - mis/*.png（誤分類画像のみ）
    """
    out_dir = os.path.join(out_root, f"round_{server_round:03d}")
    os.makedirs(out_dir, exist_ok=True)

    # ===================== CSV保存 =====================
    csv_path = os.path.join(out_dir, "preds.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "true_label", "pred_label", "confidence"])
        for i in range(labels.size(0)):
            writer.writerow([i, int(labels[i]), int(preds[i]), float(confs[i])])

    # ===================== グリッド画像保存 =====================
    vis = _unnormalize(images.detach().cpu())
    grid = make_grid(vis, nrow=8)
    save_image(grid, os.path.join(out_dir, "grid.png"))

    # ===================== 誤分類画像保存 =====================
    mis_dir = os.path.join(out_dir, "mis")
    os.makedirs(mis_dir, exist_ok=True)

    labels_cpu = labels.detach().cpu()
    preds_cpu = preds.detach().cpu()
    confs_cpu = confs.detach().cpu()

    for i in range(labels_cpu.size(0)):
        t = int(labels_cpu[i])
        p = int(preds_cpu[i])
        if p != t:
            c = float(confs_cpu[i])
            filename = f"{i:03d}_true{t}_pred{p}_conf{c:.3f}.png"
            save_image(vis[i], os.path.join(mis_dir, filename))


# ============================================================
# 追加：毎ラウンド config を train/eval 両方に配る FedAvg
# ============================================================
class FedAvgWithPerRoundConfig(FedAvg):
    """
    - configure_train / configure_evaluate が呼ばれるたびに、
      クライアントへ渡す instructions に config を「そのラウンド番号付き」で注入する。
    - Flowerのバージョン差（Message/RecordDict/FitInsなど）を吸収して、
      ConfigRecord が空にならないようにする。
    """

    def __init__(
        self,
        *,
        fraction_evaluate: float,
        lr: float,
        data_split: str,
        alpha: float,
        attack_type: str,
        malicious_ids,
        poison_rate: float,
    ):
        super().__init__(fraction_evaluate=fraction_evaluate)
        self._base_cfg = {
            "lr": lr,
            "data_split": data_split,
            "alpha": alpha,
            "attack_type": attack_type,
            "malicious_ids": malicious_ids,
            "poison_rate": poison_rate,
        }

    def _cfg(self, server_round: int) -> ConfigRecord:
        d = dict(self._base_cfg)
        d["server_round"] = server_round
        return ConfigRecord(d)

    def _inject_config_anywhere(self, ins, cfg: ConfigRecord) -> bool:
        """Flowerのバージョン差を吸収して、configをとにかく届く形で入れる。"""

        # Case A: ins が Message で ins.content が RecordDict/TypedDict など
        if hasattr(ins, "content"):
            try:
                ins.content["config"] = cfg
                return True
            except Exception:
                pass

        # Case B: ins が (client_proxy, something) のタプル
        if isinstance(ins, tuple) and len(ins) == 2:
            obj = ins[1]
            if hasattr(obj, "content"):
                try:
                    obj.content["config"] = cfg
                    return True
                except Exception:
                    pass
            if hasattr(obj, "config"):
                try:
                    obj.config = cfg  # FitIns/EvaluateIns系
                    return True
                except Exception:
                    pass

        # Case C: FitIns/EvaluateIns っぽい（.config を持つ）
        if hasattr(ins, "config"):
            try:
                ins.config = cfg
                return True
            except Exception:
                pass

        return False

    def configure_train(self, server_round: int, *args, **kwargs):
        instructions = super().configure_train(server_round, *args, **kwargs)
        cfg = self._cfg(server_round)

        for ins in instructions:
            self._inject_config_anywhere(ins, cfg)

        return instructions

    def configure_evaluate(self, server_round: int, *args, **kwargs):
        instructions = super().configure_evaluate(server_round, *args, **kwargs)
        cfg = self._cfg(server_round)

        for ins in instructions:
            self._inject_config_anywhere(ins, cfg)

        return instructions


# ============================================================
# メイン処理（FL全体制御）
# ============================================================
@app.main()
def main(grid: Grid, context: Context) -> None:
    # ===================== 実験設定読み込み（pyproject.toml） =====================
    fraction_evaluate = context.run_config["fraction-evaluate"]
    num_rounds = context.run_config["num-server-rounds"]
    lr = context.run_config["learning-rate"]
    local_epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]

    # ---- 追加：データ分割 / 攻撃 設定（無ければデフォルト） ----
    data_split = context.run_config.get("data-split", "iid")                 # "iid" / "non_iid" / "dirichlet"
    alpha = float(context.run_config.get("dirichlet-alpha", 0.5))            # Dirichlet 用
    attack_type = context.run_config.get("attack-type", "none")              # "none" / "sign_flip" / "scale" / ...
    malicious_ids = context.run_config.get("malicious-ids", [])              # 例: [0,1]
    poison_rate = float(context.run_config.get("poison-rate", 0.0))          # label_flip 等で利用する想定

    # 表示（確認用）
    print("=== Experiment Configuration ===")
    print("rounds        :", num_rounds)
    print("fraction_eval :", fraction_evaluate)
    print("local_epochs  :", local_epochs)
    print("batch_size    :", batch_size)
    print("learning_rate :", lr)
    print("optimizer     : AdamW")
    print("data_split    :", data_split, f"(alpha={alpha})")
    print("attack_type   :", attack_type)
    print("malicious_ids :", malicious_ids)
    print("poison_rate   :", poison_rate)
    print("seed          : 42")
    print("================================")

    # 実験ログ保存（あなたの log_run_config 仕様に合わせて iid を bool で渡す）
    log_run_config(
        rounds=num_rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=lr,
        optimizer="AdamW",
        iid=(data_split == "iid"),
        seed=42,
    )

    # ===================== グローバルモデル初期化 =====================
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # ===================== 集約戦略（毎ラウンド config を注入する版） =====================
    strategy = FedAvgWithPerRoundConfig(
        fraction_evaluate=fraction_evaluate,
        lr=lr,
        data_split=data_split,
        alpha=alpha,
        attack_type=attack_type,
        malicious_ids=malicious_ids,
        poison_rate=poison_rate,
    )

    # ===================== FL開始 =====================
    # この Flower 版では start() が train_config_fn を受け取らないので、
    # strategy 側（configure_train/evaluate）で毎ラウンド注入する。
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    # ===================== 最終モデル保存 =====================
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")


# ============================================================
# 各ラウンド後のグローバル評価
# ============================================================
def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # ===================== 全体テスト評価 =====================
    test_dataloader = load_centralized_dataset()
    test_loss, test_acc = test(model, test_dataloader, device)

    # ===================== 固定画像評価（可視化用） =====================
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

    # ===================== ログ出力 =====================
    log_server_metrics(server_round, test_acc, test_loss)
    print_round_summary(server_round, accuracy=test_acc, loss_func=test_loss)

    return MetricRecord({"accuracy": test_acc, "loss_func": test_loss})
