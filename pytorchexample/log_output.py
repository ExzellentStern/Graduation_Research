# log_output.py
import os
import csv
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

# =========================
# ログ保存先
# =========================
LOG_DIR = "logs"
CLIENT_CSV = os.path.join(LOG_DIR, "client_metrics.csv")
SERVER_CSV = os.path.join(LOG_DIR, "server_metrics.csv")
CONFIG_TXT = os.path.join(LOG_DIR, "run_config.txt")
PLOT_DIR = os.path.join(LOG_DIR, "plots")


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
    seed: int,
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
        f.write(f"seed          : {seed}\n")
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
