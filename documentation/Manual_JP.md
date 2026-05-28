# マニュアル — PiXY（日本語）

このマニュアルは、現行 v1.4 のワークフロー、主要操作、設定、ログ、ファイル形式を説明します。

## 目次
- 概要
- インストール
- 起動オプション
- 操作フロー（v1.4）
- UI の説明
- フィデューシャル点の扱い
- 出力ファイル（CSV, centroids_*.txt）
- 設定ファイル（Config.py）
- ログとデバッグ

## 概要
- PiXY は、画像中の候補点（重心）を抽出し、フィデューシャル点を使って画像座標（`u`, `v`）をステージ座標（`X`, `Y`, `Z`）へ変換します。

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

## 操作フロー（v1.4）
1. `New Project` で画像を読み込む。
2. `START Centroid Extraction` を押す。
3. 左パネルで抽出パラメータ（`Number of Groups`, `Boundary Offset`, `Neck Separation`, `Shape Complexity`, `Grain Size Threshold` ヒストグラム）を調整する。
4. `Add GroupN` で必要な点を中カラムへ追加する。
5. `Finish Centroid Extraction` で通常モードへ戻る。
6. `Add Fiducial Point` / `Update u, v` で参照点を整えて、`Export XYZ` またはクリップボード出力する。

## UI の主要構成
- 左パネル
	- `START/Finish Centroid Extraction` で抽出モードを切替
	- `Recalculation Trigger`（`Auto` / `Manual`）
	- グループカード（`Add GroupN`、`Show/Hide`）
- 中央パネル
	- 候補点テーブルと操作（`Export XYZ`, `Clipboard`, `Add Target`, `Update u, v`, `Clear`）
- 右パネル
	- 画像表示と座標/向き制御
	- `Boundary` と `Display Mode`（`Original`/`Posterized`）は抽出モード中のみ表示
	- 通常モードは `Original` + `Boundary OFF` 固定

## フィデューシャル点の追加・編集
1. `Add Fiducial Point` をクリックしてフィデューシャル点モードに入る。
2. 画像上で既知位置をクリックし観測点を追加。
3. テーブルで画像座標 (`u`, `v`) と装置座標 (`Stage X`, `Stage Y`, `Stage Z`) を確認・編集し、フィットを再計算します。

## 出力ファイル
- `centroids_YYYYMMDD_HHMMSS.txt`: 検出された重心リスト。
- エクスポートCSV: 変換後座標を CSV で保存できます。

## 設定ファイル
- `Config.py` を開き、既定パラメータ（表示設定や閾値）を調整できます。

## ログ
- デバッグ用ログは `debug_px2xy.log` を参照してください。

## スクリーンショット
- 重要画面のスクリーンショットを `documentation/images/` に保存してください。
