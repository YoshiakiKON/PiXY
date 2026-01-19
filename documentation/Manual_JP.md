# マニュアル — PiXY（日本語）

このマニュアルは機能の詳細、設定、ログ、ファイル形式説明を含みます。

## 目次
- 概要
- インストール
- 起動オプション
- UI の説明
- 参照点の扱い
- 出力ファイル（CSV, centroids_*.txt）
- 高度な設定（Config.py）
- ログとデバッグ

## インストール
1. Python 3.10+ を用意
2. 仮想環境を作成し、依存をインストール:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 起動オプション
- `--auto` : 前回画像を自動処理します。
- `--auto-exit` : 自動処理後に終了します。

## UI の主要構成
- メインウィンドウ: 画像ビュー、重心リスト、参照テーブル、ツールバー。
- 参照テーブル: 画像座標（`u`, `v`）と装置座標（`Stage X`, `Stage Y`, `Stage Z`）および推定座標、残差(residual)を表示します。

## 参照点の追加・編集
1. `Add Ref` をクリックして参照モードに入る。
2. 画像上で既知位置をクリックし観測点を追加。
3. テーブルで画像座標 (`u`, `v`) と装置座標 (`Stage X`, `Stage Y`, `Stage Z`) を確認・編集し、フィットを再計算します。

## 出力ファイル
- centroids_YYYYMMDD_HHMMSS.txt: 検出された重心リスト。
- エクスポートCSV: 変換後座標を CSV で保存できます。

## 設定ファイル
- `Config.py` を開き、既定パラメータ（表示設定や閾値）を調整できます。

## ログ
- デバッグ用ログは `debug_px2xy.log` を参照してください。

## スクリーンショット
- 重要画面のスクリーンショットを `documentation/images/` に保存してください。
