"""
robust_strategy.py

aggregation.py（実験計画書 7.2）を Flower の ServerApp Strategy に接続するモジュール。

【この設計での役割分担】
- aggregation.py    : 集約の中核ロジック（純粋関数、numpyベース、Flower非依存）
- robust_strategy.py: Flowerの Strategy インターフェースに接続する「配線」部分
- server_app.py      : どの aggregation_method / f を使うかを決めて Strategy を組み立てる

FedAvg を継承し、以下の2点だけをオーバーライドする：
1. configure_train / configure_evaluate
   毎ラウンドのconfig（lr, data_split, alpha, attack_type, malicious_ids, poison_rate）を
   config（ConfigRecord）に直接書き込んで渡す、公式ドキュメント記載のシンプルな方法。
   ※ client_app.py が "server_round"（アンダースコア）を参照しているため、
     Flower組み込みが自動セットする "server-round"（ハイフン）とは別に
     互換性のため "server_round" も明示的にセットする。
2. aggregate_train
   クライアントから届いた ArrayRecord を state_dict → ベクトル化し、
   aggregation.py の aggregate() で頑健集約してから ArrayRecord に戻す。
   MetricRecord（train_loss等のログ用メトリクス）は従来通り重み付き平均のまま。
"""

from typing import Iterable, Optional

import numpy as np

from flwr.app import ArrayRecord, ConfigRecord
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg
from flwr.app import Message, MetricRecord

from aggregation import (
    aggregate,
    is_bulyan_feasible,
    stack_updates,
    state_dict_to_vector,
    vector_to_state_dict,
)
from attack import adaptive_attack


class RobustFedAvg(FedAvg):
    """
    Robust Aggregation 対応の FedAvg 拡張 Strategy。

    追加パラメータ:
        aggregation_method: "fedavg" | "median" | "trimmed_mean" | "krum" | "bulyan"
        num_malicious_assumed: 防御側が想定する悪性クライアント数の上限 f
            （攻撃側の実際の malicious_ids とは独立。防御は f だけを知っている前提）
        trimmed_mean_beta: 指定があれば trimmed_mean の beta として f より優先
        krum_multi_k: krum で平均に使う候補数（1なら標準Krum）

        lr / data_split / alpha / attack_type / malicious_ids / poison_rate:
            既存の FedAvgWithPerRoundConfig と同じ、毎ラウンド配布するconfig
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
        seed: int = 42,
        aggregation_method: str = "fedavg",
        num_malicious_assumed: int = 0,
        trimmed_mean_beta: Optional[int] = None,
        krum_multi_k: int = 1,
        **kwargs,
    ):
        super().__init__(fraction_evaluate=fraction_evaluate, **kwargs)

        # --- 毎ラウンド配布するconfig（既存のFedAvgWithPerRoundConfigと同じ項目） ---
        self.lr = lr
        self.data_split = data_split
        self.alpha = alpha
        self.attack_type = attack_type
        self.malicious_ids = list(malicious_ids)
        self.poison_rate = poison_rate
        self.seed = seed

        # --- Robust Aggregation 設定 ---
        self.aggregation_method = aggregation_method.lower()
        self.num_malicious_assumed = num_malicious_assumed
        self.trimmed_mean_beta = trimmed_mean_beta
        self.krum_multi_k = krum_multi_k

        # --- Adaptive Model Poisoning Attack（実験計画書7.4）用：
        #     このラウンド開始時点のグローバルモデルベクトル（configure_trainで退避）
        self._last_global_vector = None

    # ============================================================
    # 毎ラウンドのconfig配布（公式ドキュメントのシンプルな方法）
    # ============================================================
    def _fill_common_config(self, server_round: int, config: ConfigRecord) -> None:
        # client_app.py が "server_round"（アンダースコア）を見ているため、
        # Flower組み込みが自動セットする "server-round"（ハイフン）とは別に明示的にセット
        config["server_round"] = server_round
        config["lr"] = self.lr
        config["data_split"] = self.data_split
        config["alpha"] = self.alpha
        config["attack_type"] = self.attack_type
        config["malicious_ids"] = self.malicious_ids
        config["poison_rate"] = self.poison_rate
        config["seed"] = self.seed

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        self._fill_common_config(server_round, config)

        # 実験計画書7.4：Adaptive攻撃で使う「集約前のグローバルモデル」を退避しておく
        # （aggregate_train は arrays を受け取らないため、ここで保持するしかない）
        global_state_dict = arrays.to_torch_state_dict()
        self._last_global_vector, _ = state_dict_to_vector(global_state_dict)

        return super().configure_train(server_round, arrays, config, grid)

    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        self._fill_common_config(server_round, config)
        return super().configure_evaluate(server_round, arrays, config, grid)

    # ============================================================
    # Robust Aggregation 本体
    # ============================================================
    def _extract_weight(self, content) -> Optional[float]:
        """reply の RecordDict から weighted_by_key（通常 'num-examples'）の値を探す。"""
        for value in content.values():
            try:
                if self.weighted_by_key in value:
                    return float(value[self.weighted_by_key])
            except TypeError:
                continue
        return None

    def _extract_metric(self, content, key: str):
        """reply の RecordDict から任意のメトリクスキー（例: 'partition-id'）の値を探す。"""
        for value in content.values():
            try:
                if key in value:
                    return value[key]
            except TypeError:
                continue
        return None

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ):
        """ArrayRecordをRobust Aggregationで集約し、MetricRecordは重み付き平均のまま返す。"""
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)
        if not valid_replies:
            return None, None

        reply_contents = [msg.content for msg in valid_replies]

        # --- 各クライアントの state_dict をベクトル化 ---
        state_dicts = [
            content[self.arrayrecord_key].to_torch_state_dict()
            for content in reply_contents
        ]
        vectors = []
        shapes_ref = None
        for sd in state_dicts:
            vec, shapes = state_dict_to_vector(sd)
            vectors.append(vec)
            if shapes_ref is None:
                shapes_ref = shapes
        updates = stack_updates(vectors)
        n = updates.shape[0]

        # --- Adaptive Model Poisoning Attack（実験計画書7.4） ---
        # 攻撃者はサーバ視点で「全クライアントの更新」と「使用中の集約手法」を
        # 把握できる（論文でいう full-knowledge 攻撃者）という前提のもと、
        # 集約直前に悪性クライアント分のベクトルをクラフトした攻撃済みベクトルへ差し替える。
        # sign_flip / scale はクライアント側（client_app.py）で既に適用済みのため、
        # ここで扱うのは attack_type == "adaptive" のときだけ。
        if self.attack_type == "adaptive" and self._last_global_vector is not None and self.malicious_ids:
            partition_ids = [self._extract_metric(c, "partition-id") for c in reply_contents]
            malicious_ids_set = set(self.malicious_ids)
            malicious_pos = [
                i
                for i, pid in enumerate(partition_ids)
                if pid is not None and int(pid) in malicious_ids_set
            ]
            if malicious_pos:
                updates = adaptive_attack(
                    updates,
                    malicious_idx=malicious_pos,
                    global_vector=self._last_global_vector,
                    aggregation_method=self.aggregation_method,
                    f=self.num_malicious_assumed,
                )

        # --- fedavg のみ num-examples による重み付けを行う ---
        weights = None
        if self.aggregation_method == "fedavg":
            weights = np.array(
                [self._extract_weight(c) or 1.0 for c in reply_contents],
                dtype=np.float64,
            )

        # --- Bulyanの事前条件チェック（不成立なら理由を明示して例外を出す） ---
        if self.aggregation_method == "bulyan" and not is_bulyan_feasible(
            n, self.num_malicious_assumed
        ):
            raise ValueError(
                f"[RobustFedAvg] round={server_round}: Bulyan は n={n}, "
                f"f={self.num_malicious_assumed} では数学的に不成立です（n >= 4f+3 が必要）。"
                " run_experiments.py 側でこの例外を捕捉し、Bulyanの限界として記録してください。"
            )

        aggregated_vector = aggregate(
            self.aggregation_method,
            updates,
            weights=weights,
            f=self.num_malicious_assumed,
            trimmed_mean_beta=self.trimmed_mean_beta,
            krum_multi_k=self.krum_multi_k,
        )

        new_state_dict = vector_to_state_dict(aggregated_vector, shapes_ref, state_dicts[0])
        arrays = ArrayRecord(new_state_dict)

        # --- メトリクス（train_loss等）は従来通り重み付き平均で集計 ---
        metrics = self.train_metrics_aggr_fn(reply_contents, self.weighted_by_key)

        return arrays, metrics
