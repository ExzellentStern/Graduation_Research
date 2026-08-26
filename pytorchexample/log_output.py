
import os
import csv
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

#ログの保存先
LOG_DIR = "logs"
CLIENT_CSV = os.path.join(LOG_DIR, "client_metrics.csv")
SERVER_CSV = os.path.join(LOG_DIR, "server_metrics.csv")
CONFIG_TXT = os.path.join(LOG_DIR, "run_config.txt")
PLOT_DIR = os.path.join(LOG_DIR, "plots")

# 実験計画書 7.5 対応：実験条件を区別できる形のCSV
# （server_metrics.csv は round/accuracy/loss_func のみで、複数条件・複数実行の
#   結果を一つのファイルに貯めていく run_experiments.py（7.6）では条件を区別できない）
RESULTS_CSV = os.path.join(LOG_DIR, "results.csv")
SUMMARY_CSV = os.path.join(LOG_DIR, "summary.csv")


# =========================
# 内部ユーティリティ
# =========================
def _ensure_dir():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)


# =========================
# 実験設定（ハイパーパラメータ）記録
# =========================
def log_run_config(
    rounds: int,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    optimizer: str,
    iid: bool,
    trial: int,
):
    _ensure_dir()
    with open(CONFIG_TXT, "a", encoding="utf-8") as f:
        f.write("=================================\n")
        f.write(f"timestamp     : {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"rounds        : {rounds}\n")
        f.write(f"local_epochs  : {local_epochs}\n")
        f.write(f"batch_size    : {batch_size}\n")
        f.write(f"learning_rate : {learning_rate}\n")
        f.write(f"optimizer     : {optimizer}\n")
        f.write(f"IID           : {iid}\n")
        f.write(f"trial         : {trial}\n")
        f.write("=================================\n\n")


# =========================
# クライアント側ログ
# =========================
def log_client_metrics(
    round_: int,
    client_id: int,
    train_loss: float | None = None,
    eval_loss: float | None = None,
    eval_acc: float | None = None,
    num_examples: int | None = None,
):
    _ensure_dir()
    file_exists = os.path.exists(CLIENT_CSV)
    with open(CLIENT_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow([
                "round",
                "client_id",
                "train_loss",
                "eval_loss",
                "eval_acc",
                "num_examples",
            ])
        w.writerow([round_, client_id, train_loss, eval_loss, eval_acc, num_examples])


# =========================
# サーバ側ログ
# =========================
def log_server_metrics(
    round_: int,
    accuracy: float,
    loss_func: float,
):
    _ensure_dir()
    file_exists = os.path.exists(SERVER_CSV)
    with open(SERVER_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["round", "accuracy", "loss_func"])
        w.writerow([round_, accuracy, loss_func])


# =========================
# 実験結果ログ（実験計画書 7.5 対応）
# =========================
_RESULTS_COLUMNS = [
    "run_id",
    "round",
    "accuracy",
    "loss_func",
    "aggregation_method",
    "data_split",
    "malicious_ratio",
    "attack_type",
    "trial",
]

_SUMMARY_COLUMNS = [
    "run_id",
    "final_round",
    "final_accuracy",
    "final_loss",
    "aggregation_method",
    "data_split",
    "malicious_ratio",
    "attack_type",
    "trial",
]


def log_results_csv(
    run_id: str,
    round_: int,
    accuracy: float,
    loss_func: float,
    aggregation_method: str,
    data_split: str,
    malicious_ratio: float,
    attack_type: str,
    trial: int,
):
    """ラウンドごとの Accuracy/Loss を、実験条件つきで logs/results.csv に追記する。"""
    _ensure_dir()
    file_exists = os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(_RESULTS_COLUMNS)
        w.writerow([
            run_id,
            round_,
            accuracy,
            loss_func,
            aggregation_method,
            data_split,
            malicious_ratio,
            attack_type,
            trial,
        ])


# =========================
# メカニズム分析用ログ（report/mechanism_analysis_plan.tex 対応）
# =========================
DIAGNOSTICS_CSV = os.path.join(LOG_DIR, "diagnostics.csv")

_DIAGNOSTICS_COLUMNS = [
    "run_id",
    "round",
    "aggregation_method",
    "data_split",
    "malicious_ratio",
    "trial",
    "n_malicious",
    "n_benign",
    "benign_spread",
    "malicious_push_dist",
    "push_ratio",
    "frac_selected_malicious",
]


def log_diagnostics_csv(
    run_id: str,
    round_: int,
    aggregation_method: str,
    data_split: str,
    malicious_ratio: float,
    trial: int,
    n_malicious: int,
    n_benign: int,
    benign_spread: float | None,
    malicious_push_dist: float | None,
    push_ratio: float | None,
    frac_selected_malicious: float | None,
):
    """
    honestクラスタの分散・maliciousの押し出し距離・Krum/Bulyanの選択結果を、
    ラウンドごとに logs/diagnostics.csv へ追記する。

    - benign_spread: honest更新の重心からの平均距離
    - malicious_push_dist: malicious（攻撃適用後）更新のhonest重心からの平均距離
    - push_ratio: malicious_push_dist / benign_spread
    - frac_selected_malicious: krum/bulyanで選択されたクライアントのうちmaliciousの割合
      （fedavg/median/trimmed_meanはNone）
    malicious_push_dist / push_ratio / frac_selected_malicious は該当しない条件（n_malicious=0など）
    ではNoneのまま書き込む。
    """
    _ensure_dir()
    file_exists = os.path.exists(DIAGNOSTICS_CSV)
    with open(DIAGNOSTICS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(_DIAGNOSTICS_COLUMNS)
        w.writerow([
            run_id,
            round_,
            aggregation_method,
            data_split,
            malicious_ratio,
            trial,
            n_malicious,
            n_benign,
            benign_spread,
            malicious_push_dist,
            push_ratio,
            frac_selected_malicious,
        ])


def log_summary_csv(
    run_id: str,
    final_round: int,
    final_accuracy: float,
    final_loss: float,
    aggregation_method: str,
    data_split: str,
    malicious_ratio: float,
    attack_type: str,
    trial: int,
):
    """1回の実行につき1行、最終ラウンドの Accuracy/Loss を logs/summary.csv に追記する。"""
    _ensure_dir()
    file_exists = os.path.exists(SUMMARY_CSV)
    with open(SUMMARY_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(_SUMMARY_COLUMNS)
        w.writerow([
            run_id,
            final_round,
            final_accuracy,
            final_loss,
            aggregation_method,
            data_split,
            malicious_ratio,
            attack_type,
            trial,
        ])


# =========================
# ターミナル用 簡易表示
# =========================
def print_round_summary(
    round_: int,
    train_loss: float | None = None,
    eval_loss: float | None = None,
    eval_acc: float | None = None,
    accuracy: float | None = None,
    loss_func: float | None = None,
):
    parts = [f"[ROUND {round_}]"]
    if train_loss is not None:
        parts.append(f"train_loss={train_loss:.4f}")
    if eval_loss is not None:
        parts.append(f"eval_loss={eval_loss:.4f}")
    if eval_acc is not None:
        parts.append(f"eval_acc={eval_acc:.4f}")
    if accuracy is not None:
        parts.append(f"accuracy={accuracy:.4f}")
    if loss_func is not None:
        parts.append(f"loss_func={loss_func:.4f}")
    print(" | ".join(parts))


# =========================
# ===== グラフ生成 =====
# =========================
def plot_server_metrics():
    """
    サーバ側の accuracy / loss_func の推移を描画
    """
    if not os.path.exists(SERVER_CSV):
        print("server_metrics.csv not found")
        return

    df = pd.read_csv(SERVER_CSV)

    plt.figure()
    plt.plot(df["round"], df["accuracy"], marker="o", label="accuracy")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("Server Accuracy per Round")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(PLOT_DIR, "server_accuracy.png"))
    plt.close()

    plt.figure()
    plt.plot(df["round"], df["loss_func"], marker="o", label="loss")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.title("Server Loss per Round")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(PLOT_DIR, "server_loss.png"))
    plt.close()


def plot_client_metrics():
    """
    クライアント側の train_loss / eval_loss / eval_acc（平均）を描画
    """
    if not os.path.exists(CLIENT_CSV):
        print("client_metrics.csv not found")
        return

    df = pd.read_csv(CLIENT_CSV)

    # round ごとに平均
    grouped = df.groupby("round").mean(numeric_only=True).reset_index()

    if "train_loss" in grouped:
        plt.figure()
        plt.plot(grouped["round"], grouped["train_loss"], marker="o")
        plt.xlabel("Round")
        plt.ylabel("Train Loss")
        plt.title("Client Train Loss (mean)")
        plt.grid(True)
        plt.savefig(os.path.join(PLOT_DIR, "client_train_loss.png"))
        plt.close()

    if "eval_loss" in grouped:
        plt.figure()
        plt.plot(grouped["round"], grouped["eval_loss"], marker="o")
        plt.xlabel("Round")
        plt.ylabel("Eval Loss")
        plt.title("Client Eval Loss (mean)")
        plt.grid(True)
        plt.savefig(os.path.join(PLOT_DIR, "client_eval_loss.png"))
        plt.close()

    if "eval_acc" in grouped:
        plt.figure()
        plt.plot(grouped["round"], grouped["eval_acc"], marker="o")
        plt.xlabel("Round")
        plt.ylabel("Eval Accuracy")
        plt.title("Client Eval Accuracy (mean)")
        plt.grid(True)
        plt.savefig(os.path.join(PLOT_DIR, "client_eval_acc.png"))
        plt.close()


def plot_all():
    """
    実験終了後に一括で呼ぶ用
    """
    plot_server_metrics()
    plot_client_metrics()
    print("Saved plots to logs/plots/")
