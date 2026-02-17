"""
このファイルは一応FederatedLearning関連なので提出するが,
機械学習以外の操作も結構ある.課題評価としては概ねFL_assignmentを用いれば良いと思う.
"""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pytorchexample.log_output import log_client_metrics, print_round_summary
from pytorchexample.task import Net, load_data
from pytorchexample.task import test as test_fn
from pytorchexample.task import train as train_fn

# 追加：ラベル分布の簡易サマリ用
from collections import Counter
import numpy as np


def summarize(loader, n_batches=10):
    """
    loader から最初の n_batches だけ取り出してラベル分布を返す。
    Non-IID かどうかの“実データ”確認用。
    """
    c = Counter()
    seen = 0
    for i, batch in enumerate(loader):
        y = batch["label"].cpu().numpy().tolist()
        c.update(y)
        seen += len(y)
        if i + 1 >= n_batches:
            break
    return seen, dict(sorted(c.items()))


# クライアントアプリ
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    # ============================================================
    # 0) サーバから受け取った設定（config）を一度だけ取り出す
    #    ★修正：ConfigRecord/TypedDict 等でも安全に扱えるよう dict 化
    # ============================================================
    cfg_rec = msg.content.get("config", None)
    cfg = dict(cfg_rec) if cfg_rec is not None else {}

    # --- IID / non-IID 用設定（データ分割）---
    data_split = cfg.get("data_split", "iid")
    alpha = float(cfg.get("alpha", 0.5))

    # --- 攻撃用設定（保持：将来の拡張のため消さない）---
    attack_type = cfg.get("attack_type", "none")
    malicious_ids = set(cfg.get("malicious_ids", []))
    poison_rate = float(cfg.get("poison_rate", 0.0))  # ※ label_flip 等で使う想定

    # --- round番号（無ければ -1）---
    server_round = cfg.get("server_round", -1)

    # ============================================================
    # 1) モデル初期化（サーバから受け取った重みをロード）
    # ============================================================
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ============================================================
    # 2) クライアントID / データ読み込み
    # ============================================================
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]

    # IID/non-IID を反映して読み込む（重要）
    trainloader, _ = load_data(
        partition_id,
        num_partitions,
        batch_size,
        data_split=data_split,
        alpha=alpha,
    )

    # 悪性クライアント判定
    is_malicious = partition_id in malicious_ids

    # ============================================================
    # 追加：ラベル分布チェック（Non-IID確認用）
    # - ログがうるさくなるので「最初の1回だけ」出す
    # - ここでは server_round が 0 のときのみ表示
    #   （あなたの設定だと train の server_round は 0,1,2... で来る想定）
    # ============================================================
    if server_round == 0:
        seen, dist = summarize(trainloader, n_batches=20)
        print(
            f"[Client {partition_id}] (train) round={server_round} split={data_split} alpha={alpha} "
            f"seen={seen} label_dist={dist}"
        )

    # ============================================================
    # ★ 修正：lr が config に無い場合でも落ちないようにフォールバック
    # - まず cfg["lr"] を触らない（KeyError対策）
    # - context.run_config から拾えるなら拾う
    # - 最後に安全なデフォルト 1e-3
    # ============================================================
    lr = float(cfg.get("lr", context.run_config.get("learning-rate", 1e-3)))
    if "lr" not in cfg:
        print(
            f"[Client {partition_id}] WARNING: 'lr' missing in cfg. "
            f"Fallback to learning-rate={context.run_config.get('learning-rate', None)} -> lr={lr}"
        )

    # ============================================================
    # 3) ローカル学習
    # ============================================================
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        lr,
        device,
    )

    # ============================================================
    # 4) ログ出力（学習）
    # ============================================================
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

    # ============================================================
    # 5) 学習後のモデル重みを取得（送信直前はCPUへ）
    # ============================================================
    state_dict_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # ============================================================
    # 6) 攻撃（ここが重要：学習後・送信直前に適用）
    #    ※ sign_flip / scale は維持
    #    ※ poison_rate は将来の label_flip 等で使う想定なので保持
    # ============================================================
    if is_malicious and attack_type == "sign_flip":
        for k in state_dict_cpu:
            state_dict_cpu[k] = -state_dict_cpu[k]

    elif is_malicious and attack_type == "scale":
        scale_factor = 10.0
        for k in state_dict_cpu:
            state_dict_cpu[k] = scale_factor * state_dict_cpu[k]

    # ============================================================
    # 7) サーバへ返送
    # ============================================================
    model_record = ArrayRecord(state_dict_cpu)

    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }

    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    # ============================================================
    # 0) 設定（config）取得：IID/non-IID を evaluate にも反映
    #    ★修正：ConfigRecord/TypedDict 等でも安全に扱えるよう dict 化
    # ============================================================
    cfg_rec = msg.content.get("config", None)
    cfg = dict(cfg_rec) if cfg_rec is not None else {}

    data_split = cfg.get("data_split", "iid")
    alpha = float(cfg.get("alpha", 0.5))
    server_round = cfg.get("server_round", -1)

    # --- 攻撃設定（evaluateでは攻撃を適用しないが、表示・整合性確認用に読む）---
    attack_type = cfg.get("attack_type", "none")
    malicious_ids = set(cfg.get("malicious_ids", []))
    poison_rate = float(cfg.get("poison_rate", 0.0))

    # ============================================================
    # 1) モデル復元
    # ============================================================
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ============================================================
    # 2) データ読み込み（評価もIID/non-IIDを揃える）
    # ============================================================
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]

    _, valloader = load_data(
        partition_id,
        num_partitions,
        batch_size,
        data_split=data_split,
        alpha=alpha,
    )

    # 悪性クライアント判定（表示用）
    is_malicious = partition_id in malicious_ids

    # ============================================================
    # 追加：評価データ側のラベル分布も確認したいなら（最初の1回だけ）
    # - evaluate は round=-1 で呼ばれることがあるので、
    #   ここでは server_round <= 0 のときだけ出すようにしておく
    # ============================================================
    if server_round <= 0:
        seen, dist = summarize(valloader, n_batches=20)
        print(
            f"[Client {partition_id}] (eval) round={server_round} split={data_split} alpha={alpha} "
            f"seen={seen} label_dist={dist}"
        )

    # ============================================================
    # 3) 評価
    # ============================================================
    eval_loss, eval_acc = test_fn(model, valloader, device)

    # ============================================================
    # 4) ログ出力（評価）
    # ============================================================
    print(
        f"[Client {partition_id}] round={server_round} split={data_split} alpha={alpha} "
        f"attack={attack_type} malicious={is_malicious} poison={poison_rate}"
    )

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

    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }

    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
