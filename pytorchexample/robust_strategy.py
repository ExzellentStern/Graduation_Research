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
from attack import adaptive_attack, alie_attack
from detection import DETECTOR_KEYS, all_detector_scores
from pytorchexample.log_output import log_detection_csv, log_diagnostics_csv


class RobustFedAvg(FedAvg):
    """
    Robust Aggregation 対応の FedAvg 拡張 Strategy。

    追加パラメータ:
        aggregation_method: "fedavg" | "median" | "trimmed_mean" | "krum" | "bulyan"
        num_malicious_assumed: 防御側が想定する悪性クライアント数の上限 f
            （攻撃側の実際の malicious_ids とは独立。防御は f だけを知っている前提）
        trimmed_mean_beta: 指定があれば trimmed_mean の beta として f より優先
        krum_multi_k: krum で平均に使う候補数（1なら標準Krum）

        lr / data_split / alpha / p / attack_type / malicious_ids / poison_rate:
            既存の FedAvgWithPerRoundConfig と同じ、毎ラウンド配布するconfig
            （alpha は data_split="dirichlet" 用，p は data_split="non_iid"（Fang方式）用）
    """

    def __init__(
        self,
        *,
        fraction_evaluate: float,
        lr: float,
        data_split: str,
        alpha: float,
        p: float = 0.5,
        attack_type: str = "none",
        malicious_ids,
        poison_rate: float,
        trial: int = 1,
        alie_z: float = 1.0,
        aggregation_method: str = "fedavg",
        num_malicious_assumed: int = 0,
        trimmed_mean_beta: Optional[int] = None,
        krum_multi_k: int = 1,
        malicious_ratio: float = 0.0,
        run_id: str = "unknown",
        **kwargs,
    ):
        super().__init__(fraction_evaluate=fraction_evaluate, **kwargs)

        # --- 毎ラウンド配布するconfig（既存のFedAvgWithPerRoundConfigと同じ項目） ---
        self.lr = lr
        self.data_split = data_split
        self.alpha = alpha
        self.p = p
        self.attack_type = attack_type
        self.malicious_ids = list(malicious_ids)
        self.poison_rate = poison_rate
        self.trial = trial
        self.alie_z = alie_z

        # --- report/mechanism_analysis_plan.tex 対応：診断ログ用の実験条件識別子 ---
        self.malicious_ratio = malicious_ratio
        self.run_id = run_id

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
        config["p"] = self.p
        config["attack_type"] = self.attack_type
        config["malicious_ids"] = self.malicious_ids
        config["poison_rate"] = self.poison_rate
        config["trial"] = self.trial

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

        # --- 各クライアントがmalicious/honestのどちらかを特定する ---
        # report/mechanism_analysis_plan.tex の診断指標（benign_spread等）は
        # attack_type によらず計算するため、malicious_ids との突合はここで毎回行う。
        partition_ids = [self._extract_metric(c, "partition-id") for c in reply_contents]
        malicious_ids_set = set(self.malicious_ids)
        malicious_pos = np.array(
            [
                i
                for i, pid in enumerate(partition_ids)
                if pid is not None and int(pid) in malicious_ids_set
            ],
            dtype=int,
        )
        malicious_pos_set = set(malicious_pos.tolist())
        benign_pos = np.array(
            [i for i in range(n) if i not in malicious_pos_set], dtype=int
        )

        # --- Adaptive Model Poisoning Attack（実験計画書7.4） ---
        # 攻撃者はサーバ視点で「全クライアントの更新」と「使用中の集約手法」を
        # 把握できる（論文でいう full-knowledge 攻撃者）という前提のもと、
        # 集約直前に悪性クライアント分のベクトルをクラフトした攻撃済みベクトルへ差し替える。
        # sign_flip / scale はクライアント側（client_app.py）で既に適用済みのため、
        # ここで扱うのは attack_type == "adaptive" のときだけ。
        if self.attack_type == "adaptive" and self._last_global_vector is not None and len(malicious_pos) > 0:
            updates = adaptive_attack(
                updates,
                malicious_idx=malicious_pos,
                global_vector=self._last_global_vector,
                aggregation_method=self.aggregation_method,
                f=self.num_malicious_assumed,
                # Krum/Bulyan攻撃のε-ball摂動用：trial番号とラウンド番号から決定的に再現可能にする
                seed=(self.trial * 1_000_003 + server_round) % (2**32 - 1),
            )
        # --- ALIE 攻撃（sotsuron_plan2.tex 4.3 A2）: サーバ側で正常更新の
        #     座標ごと平均・標準偏差から悪性更新をクラフトする aggregation-agnostic 攻撃。
        elif self.attack_type == "alie" and len(malicious_pos) > 0 and len(benign_pos) > 0:
            updates = alie_attack(
                updates,
                malicious_idx=malicious_pos,
                benign_idx=benign_pos,
                z=self.alie_z,
            )

        # --- report/mechanism_analysis_plan.tex 対応：honestクラスタの分散とmaliciousの押し出し距離 ---
        # 集約直前（攻撃適用後）のupdatesを使い、honest重心からの平均距離として定義する。
        # honestが存在しない/maliciousが存在しない場合はNoneのままログする。
        benign_spread = None
        malicious_push_dist = None
        push_ratio = None
        if len(benign_pos) > 0:
            benign_vectors = updates[benign_pos]
            benign_center = benign_vectors.mean(axis=0)
            benign_spread = float(np.linalg.norm(benign_vectors - benign_center, axis=1).mean())
            if len(malicious_pos) > 0:
                malicious_vectors = updates[malicious_pos]
                malicious_push_dist = float(
                    np.linalg.norm(malicious_vectors - benign_center, axis=1).mean()
                )
                if benign_spread > 1e-12:
                    push_ratio = malicious_push_dist / benign_spread

        # --- sotsuron_plan2.tex 4.2--4.5 対応：更新ベクトルレベルの検知器スコアを
        #     1クライアント1行で logs/detection_updates.csv へ記録する。
        #     検知は更新ベクトル g_i = w_i - w_global に対して行うため、
        #     受信した（攻撃適用後の）モデルベクトルからグローバルモデルを引いて渡す。
        #     正解ラベル is_malicious は評価専用であり、検知器スコアの計算には使わない。
        if self._last_global_vector is not None:
            deltas = updates - self._last_global_vector[None, :]
        else:
            deltas = updates - updates.mean(axis=0, keepdims=True)
        try:
            scores = all_detector_scores(deltas)
            delta_norms = np.linalg.norm(deltas, axis=1)
            detection_rows = []
            for pos in range(n):
                pid = partition_ids[pos]
                row = {
                    "run_id": self.run_id,
                    "round": server_round,
                    "aggregation_method": self.aggregation_method,
                    "data_split": self.data_split,
                    "alpha": self.alpha,
                    "attack_type": self.attack_type,
                    "malicious_ratio": self.malicious_ratio,
                    "trial": self.trial,
                    "client_pos": pos,
                    "partition_id": (int(pid) if pid is not None else None),
                    "is_malicious": int(pos in malicious_pos_set),
                    "delta_norm": float(delta_norms[pos]),
                    "benign_spread": benign_spread,
                    "push_ratio": push_ratio,
                }
                for key in DETECTOR_KEYS:
                    row[key] = float(scores[key][pos])
                detection_rows.append(row)
            log_detection_csv(detection_rows)
        except Exception as exc:  # 検知ログの失敗で学習自体は止めない
            print(f"[RobustFedAvg] round={server_round}: 検知ログの記録に失敗しました: {exc}")

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

        aggregated_vector, agg_info = aggregate(
            self.aggregation_method,
            updates,
            weights=weights,
            f=self.num_malicious_assumed,
            trimmed_mean_beta=self.trimmed_mean_beta,
            krum_multi_k=self.krum_multi_k,
            return_info=True,
        )

        # --- report/mechanism_analysis_plan.tex 対応：Krum/Bulyanの選択結果 ---
        # selected_idx のうちmaliciousが占める割合（krum標準=multi_k=1なら0か1になる）。
        # fedavg/median/trimmed_meanはselected_idxが無いためNoneのまま。
        frac_selected_malicious = None
        selected_idx = agg_info.get("selected_idx")
        if selected_idx is not None and len(selected_idx) > 0:
            frac_selected_malicious = float(
                sum(1 for idx in selected_idx if int(idx) in malicious_pos_set) / len(selected_idx)
            )

        log_diagnostics_csv(
            run_id=self.run_id,
            round_=server_round,
            aggregation_method=self.aggregation_method,
            data_split=self.data_split,
            malicious_ratio=self.malicious_ratio,
            trial=self.trial,
            n_malicious=len(malicious_pos),
            n_benign=len(benign_pos),
            benign_spread=benign_spread,
            malicious_push_dist=malicious_push_dist,
            push_ratio=push_ratio,
            frac_selected_malicious=frac_selected_malicious,
        )

        new_state_dict = vector_to_state_dict(aggregated_vector, shapes_ref, state_dicts[0])
        arrays = ArrayRecord(new_state_dict)

        # --- メトリクス（train_loss等）は従来通り重み付き平均で集計 ---
        metrics = self.train_metrics_aggr_fn(reply_contents, self.weighted_by_key)

        return arrays, metrics
