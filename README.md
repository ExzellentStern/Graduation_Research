# Federated Learning における悪性更新の検知可能性の評価

卒業研究（3CDA2216 笠原秀星）の実験コード・実験データ・卒業論文をまとめたブランチです。

- **卒業論文**：[`Memo/soturon3.pdf`](Memo/soturon3.pdf)
- **実験コードの解説書**：[`manual.pdf`](manual.pdf)

## 研究の概要

Federated Learning（FL）で悪性クライアントが送る更新を，集約の直前の更新ベクトルに対する二値分類として捉え，
4種類の検知器（ノルム偏差，参照方向へのコサイン距離，2クラスタリング，スペクトル残差）の検知性能を ROC-AUC で評価しました。
評価の軸は，データ分布の不均一性（Non-IID 度）と攻撃のステルス性の2つです。
符号反転攻撃と ALIE に対して，計300回の実行を行いました。結果と考察は卒業論文の第8〜10章にあります。

## ファイル構成

| パス | 内容 |
|---|---|
| `Memo/soturon3.pdf` | 卒業論文。`soturon3.tex` がソース，`fig/` が論文の図 |
| `manual.pdf` | 実験コードの解説書（ソースは `manual.tex`）。第I部 基礎知識，第II部 コードの全体像，第III部 使い方，第IV部 卒論の主張と根拠，第V部 リファレンス |
| `fl_detection/` | 実験コード（Flower アプリのパッケージ） |
| `pyproject.toml` | Flower アプリの定義と，実験の設定値すべて（共通の値と，スイープで変える軸） |
| `run_experiments.py` | スイープ（全条件の実行）を起動するスクリプト |
| `analyze_detection.py` | 実験ログを集計し，論文の図と表を作るスクリプト |
| `experiments_detection_v3/` | 本実験（300実行）のログと分析結果。**卒業論文の結果はすべてこのデータによる** |
| `experiments_detection_v2/` | 以前のデータ分割方式（データ数も偏る）で行った54実行のログ。分割方式を変えた根拠として残したもので，論文の結論には使っていない |

### `fl_detection/` の中身

| ファイル | 役割 |
|---|---|
| `server_app.py` | サーバの制御。設定の読み込み，乱数の種の固定，毎ラウンドの精度評価 |
| `client_app.py` | クライアントの学習。符号反転攻撃はここで行う |
| `robust_strategy.py` | 集約の直前の処理。ALIE の適用と，検知器 D1〜D4 のスコア・検知性能の計算と記録 |
| `detection.py` | 検知器 D1〜D4，検知性能の指標（ROC-AUC など），分布構造の診断指標 |
| `attack.py` | 攻撃（符号反転，ALIE） |
| `task.py` | CNN モデル，データ分割（Hsu ら（2019）の Dirichlet 分割。データ数は全クライアントで同じ） |
| `aggregation.py` | 集約（FedAvg）と，重みとベクトルの変換 |
| `malicious.py` | 悪性クライアントの選出 |
| `log_output.py` | ログの出力 |

### `experiments_detection_v3/_logs/` の中身

| ファイル | 内容 |
|---|---|
| `detection_metrics.csv` | 実行 × ラウンド × 検知器ごとの検知性能（ROC-AUC など） |
| `detection_updates.csv.gz` | ラウンドごと・クライアントごとのノルムと検知スコア（GitHub の容量制限のため gzip で圧縮） |
| `results.csv`，`summary.csv` | ラウンドごと・最終ラウンドのテスト精度 |
| `run_config.txt` | 実行ごとに実際に使った設定の全項目 |
| `analysis/` | `analyze_detection.py` が作った図と表（論文の図表の元） |

## 実験の条件

| 項目 | 値 |
|---|---|
| データセット | MNIST（学習データ 60,000 件のうち 30,000 件を使用） |
| クライアント数 | 20（毎ラウンド全員が参加），うち悪性 4（割合 0.2） |
| 学習 | 50 ラウンド，ローカルエポック 2，バッチサイズ 32，学習率 0.01（SGD），集約は FedAvg |
| Non-IID 度 | Dirichlet の $\alpha$ = 1000（ほぼ IID），5，1 |
| 攻撃 | なし，符号反転，ALIE（係数 $z$ = 0.385，0.5，0.75，1，1.25，1.5，2，3） |
| 試行 | 各条件 10 回，計 300 実行（CPU で実行） |

## 再現の手順

詳しくは `manual.pdf` の第III部「使い方」にあります。

```bash
pip install -e .                                   # 依存パッケージ（Flower, PyTorch など）のインストール
python3 run_experiments.py --dry-run               # 実行せずに，条件の一覧とコマンドを表示する
python3 run_experiments.py                         # スイープを実行する（300実行。CPU で約60時間）

# 分析（論文の図表を作り直す）
gunzip -k experiments_detection_v3/_logs/detection_updates.csv.gz
python3 analyze_detection.py experiments_detection_v3/_logs
```

## 注意

- コードのコメントには，`legacy_fang_reproduction/`（以前の Fang ら（2020）の再現実験）や `unused_features/`（現在使っていない攻撃・集約手法など）への言及がありますが，これらはこのブランチには含めていません。`legacy_fang_reproduction/` は `main` ブランチにあります。
