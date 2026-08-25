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
from datetime import datetime
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid

# ============================================================
# Flower / PyTorch 関連
# ============================================================
import torch
from flwr.app import ArrayRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

from pytorchexample.log_output import (
    configure_log_dir,
    log_server_metrics,
    print_round_summary,
    log_run_config,
    log_results_csv,
    log_summary_csv,
)
from pytorchexample.task import Net, load_centralized_dataset, test
from pytorchexample.robust_strategy import RobustFedAvg
from malicious import resolve_malicious_ids


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
# メイン処理（FL全体制御）
#
# 【変更履歴】毎ラウンドconfig配布 + Robust Aggregation の実体は
# pytorchexample/robust_strategy.py の RobustFedAvg に統合した
# （実験計画書 7.2 の aggregation.py と server_app.py の連携）。
# 旧 FedAvgWithPerRoundConfig（_inject_config_anywhereによる回避策）は削除し、
# 公式ドキュメント記載のシンプルな config 直接書き込み方式に統一した。
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
    attack_type = context.run_config.get("attack-type", "none")              # "none" / "sign_flip" / "scale" / "adaptive"
    poison_rate = float(context.run_config.get("poison-rate", 0.0))          # label_flip 等で利用する想定
    seed = int(context.run_config.get("seed", 42))                           # 実験計画書7.5：データ分割の再現性用シード

    # ---- 追加：実験計画書7.6：実験ごとに出力先を分けるためのディレクトリ ----
    # run_experiments.py が実験条件ごとに別ディレクトリを渡すことで、
    # final_model.pt / fixed_eval の画像が実験間で上書きされないようにする。
    output_dir = str(context.run_config.get("output-dir", "."))
    os.makedirs(output_dir, exist_ok=True)

    # ---- 追加：results.csv/summary.csv/diagnostics.csv の出力先を output_dir に紐付ける ----
    # log_output.py のデフォルト（logs/、プロジェクト直下固定）のままだと、
    # run_experiments.py で条件ごとに output_dir を分けても集計ログは全条件・全実行で
    # 1つのファイルに混ざってしまい、あとから解析する際に条件の切り分けができない。
    # output_dir が指定されている（"." 以外の）場合は、そのスイープの親ディレクトリ
    # （例: experiments_phase1/<条件>/ -> experiments_phase1/_logs/）に集約する。
    if output_dir not in (".", ""):
        sweep_root = os.path.dirname(os.path.normpath(output_dir)) or "."
        configure_log_dir(os.path.join(sweep_root, "_logs"))

    # ---- 追加：悪性クライアント生成（実験計画書 7.3 対応） ----
    # malicious-ids に手動指定（例: "0,1"）があればそちらを優先し、
    # 空なら num-clients / malicious-ratio / malicious-seed から自動生成する。
    num_clients = int(context.run_config.get("num-clients", 100))
    malicious_ratio = float(context.run_config.get("malicious-ratio", 0.0))
    malicious_seed = int(context.run_config.get("malicious-seed", 42))
    malicious_ids_raw = str(context.run_config.get("malicious-ids", ""))
    malicious_ids = resolve_malicious_ids(
        num_clients=num_clients,
        malicious_ratio=malicious_ratio,
        seed=malicious_seed,
        manual_ids_raw=malicious_ids_raw,
    )

    # ---- 追加：Robust Aggregation 設定（実験計画書 7.2 対応） ----
    # "fedavg" / "median" / "trimmed_mean" / "krum" / "bulyan"
    aggregation_method = context.run_config.get("aggregation-method", "fedavg")
    # 防御側が想定する悪性クライアント数の上限 f。
    # -1（pyproject.tomlのデフォルト＝未指定）なら、簡易的に実際の malicious_ids の数をそのまま使う
    # （攻撃側と防御側の想定を分けて実験したい場合は run_config で 0 以上の値を明示的に指定すること）。
    # ★注意：0 は「防御側は悪性クライアントがいないと想定する」という有効な設定であり、
    #   「未指定」とは区別する必要があるため、番兵値として -1 を使っている。
    num_malicious_assumed_raw = int(context.run_config.get("num-malicious-assumed", -1))
    num_malicious_assumed = (
        len(malicious_ids) if num_malicious_assumed_raw < 0 else num_malicious_assumed_raw
    )

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
    print("num_clients   :", num_clients)
    print("malicious_ratio:", malicious_ratio, f"(seed={malicious_seed}, manual={malicious_ids_raw!r})")
    print("malicious_ids :", malicious_ids)
    print("poison_rate   :", poison_rate)
    print("aggregation   :", aggregation_method, f"(f_assumed={num_malicious_assumed})")
    print("seed          :", seed)
    print("================================")

    # 実験ログ保存（あなたの log_run_config 仕様に合わせて iid を bool で渡す）
    log_run_config(
        rounds=num_rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=lr,
        optimizer="AdamW",
        iid=(data_split == "iid"),
        seed=seed,
    )

    # ---- 追加：実験計画書7.5：results.csv / summary.csv 用の実行識別子 ----
    run_id = (
        f"{aggregation_method}_{data_split}_ratio{malicious_ratio}"
        f"_seed{seed}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    # 最終ラウンドの結果を summary.csv に書くために覚えておく入れ物
    last_eval = {}

    # ===================== グローバルモデル初期化 =====================
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # ===================== 集約戦略（毎ラウンド config 配布 + Robust Aggregation） =====================
    strategy = RobustFedAvg(
        fraction_evaluate=fraction_evaluate,
        lr=lr,
        data_split=data_split,
        alpha=alpha,
        attack_type=attack_type,
        malicious_ids=malicious_ids,
        poison_rate=poison_rate,
        seed=seed,
        aggregation_method=aggregation_method,
        num_malicious_assumed=num_malicious_assumed,
        malicious_ratio=malicious_ratio,
        run_id=run_id,
    )

    # ===================== 各ラウンド後のグローバル評価 =====================
    # 実験計画書7.5：results.csv に条件つきで書き込むため、main() 内のクロージャにして
    # run_id / aggregation_method / data_split / malicious_ratio / attack_type / seed を
    # そのまま参照できるようにしている。最終ラウンドの値は last_eval に残しておき、
    # strategy.start() 完了後に summary.csv へ1行だけ書き込むのに使う。
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
            out_root=os.path.join(output_dir, "fixed_eval"),
        )

        # ===================== ログ出力 =====================
        log_server_metrics(server_round, test_acc, test_loss)
        print_round_summary(server_round, accuracy=test_acc, loss_func=test_loss)

        log_results_csv(
            run_id=run_id,
            round_=server_round,
            accuracy=test_acc,
            loss_func=test_loss,
            aggregation_method=aggregation_method,
            data_split=data_split,
            malicious_ratio=malicious_ratio,
            attack_type=attack_type,
            seed=seed,
        )
        last_eval["round"] = server_round
        last_eval["accuracy"] = test_acc
        last_eval["loss_func"] = test_loss

        return MetricRecord({"accuracy": test_acc, "loss_func": test_loss})

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
    model_path = os.path.join(output_dir, "final_model.pt")
    print(f"\nSaving final model to {model_path} ...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, model_path)

    # ===================== 実験計画書7.5：summary.csv（最終ラウンドの結果を1行） =====================
    if last_eval:
        log_summary_csv(
            run_id=run_id,
            final_round=last_eval["round"],
            final_accuracy=last_eval["accuracy"],
            final_loss=last_eval["loss_func"],
            aggregation_method=aggregation_method,
            data_split=data_split,
            malicious_ratio=malicious_ratio,
            attack_type=attack_type,
            seed=seed,
        )