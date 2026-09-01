# 検知評価実験 — ファイル索引

更新ベクトルレベルの悪性更新検知の実験（`research_plan_v2.pdf` / `manual.pdf`）。

Flower アプリは `flwr run .` をリポジトリ直下で実行するため、ランタイムの
Python モジュールは直下に置く必要がある（サブフォルダに移すと import が壊れる）。
そのぶん、旧 Fang 再現実験の資産は `legacy_fang_reproduction/` に退避した。

## この実験で使うファイル（すべてリポジトリ直下）

### ランタイム（`flwr run` 実行中に読み込まれる）
| ファイル | 役割 |
|---|---|
| `pyproject.toml` | Flower アプリ定義・全 run-config の既定値 |
| `pytorchexample/server_app.py` | サーバ制御。`alie-z` 等を読む |
| `pytorchexample/robust_strategy.py` | 集約直前フック。ALIE 適用＋ D1–D4 スコア計算・記録 |
| `pytorchexample/client_app.py` | クライアント学習。`sign_flip` 攻撃はここ |
| `pytorchexample/task.py` | CNN モデル・データ分割（`dirichlet` を使用） |
| `pytorchexample/log_output.py` | `log_detection_csv`（1クライアント1行のログ） |
| `detection.py` | 検知器 D1–D4・検知性能指標（純 numpy、scikit-learn 不要） |
| `attack.py` | `sign_flip` / `alie`（`adaptive` は旧実験用、未使用） |
| `aggregation.py` | 集約手法（本実験は `fedavg` のみ） |
| `malicious.py` | 悪性クライアント ID の解決 |

### スイープ・解析・確認
| ファイル | 役割 | 実行 |
|---|---|---|
| `run_experiments.py` | スイープ実行 | `python3 run_experiments.py --detection` |
| `analyze_detection.py` | 検知ログ後処理（指標 CSV・図） | `python3 analyze_detection.py` |
| `tools/verify_dirichlet_partition.py` | Dirichlet 分割の目視確認 | `python3 tools/verify_dirichlet_partition.py --alpha 100 0.5 0.1 --clients 20 --trial 1` |

### テスト
`test_detection.py`（検知器）、`test_attack.py` / `test_aggregation.py` / `test_malicious.py`（既存）。
※ `test_attack.py::test_adaptive_attack_noop_for_fedavg` は本実験以前からの既知失敗。

### ドキュメント
`research_plan_v2.tex` / `.pdf` … 研究計画（目的・仮説・指標・スケジュール）
`manual.tex` / `.pdf` … 本コードの操作マニュアル

## 出力先
| パス | 内容 | git |
|---|---|---|
| `logs/detection_updates.csv` | 1クライアント1行の検知ログ（全条件が追記） | ignore |
| `logs/detection_metrics.csv` | 条件×ラウンド×検知器の指標（`analyze_detection.py`） | ignore |
| `experiments_detection/<条件名>/` | 条件ごとの `final_model.pt`（完了判定に使用） | ignore |
| `results_detection/*.png`, `*.csv` | 図・要約表 | ignore |

## 実行フロー
```
python3 test_detection.py                                  # 0. テスト
python3 tools/verify_dirichlet_partition.py --alpha 100 0.5 0.1 --clients 20 --trial 1  # 1. 分割確認
python3 run_experiments.py --detection                     # 2. スイープ（27条件, CPU 1–2h）
python3 analyze_detection.py                               # 3. 指標CSV＋図
```

## legacy_fang_reproduction/
旧 Fang 再現実験（卒論第7章）の資産。本実験のコードは一切参照しない。
- `report/` … 第7章の表・図生成物・`mechanism_analysis_plan.tex`・`fang_correspondence_audit.tex`
- `result/` … 第7章の図（`sotsuron.tex` はここを参照するようパス変更済み）
- `logs_experiments_sweep/` … 760 run の生データ
- `sotsuron_plan.*` … 旧研究計画（`research_plan_v2` に置換）
- `analyze_logs.py` / `visualize_results.py` … 旧スイープ用スクリプト
- `adaptive_model_poisoning_attack.pdf` … Fang 論文
- `sotsuron_fang_ch7_snapshot.pdf` … 移動前にビルドした卒論 PDF
