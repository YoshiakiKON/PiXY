# PiXY 操作マニュアル（日本語）

---

## 表紙情報

- ソフトウェア: PiXY — Pixel-to-Stage XY Coordinate Converter
- 著者: Yoshiaki KON
- リポジトリ: https://github.com/YoshiakiKON/PiXY
- Zenodo DOI: 10.5281/zenodo.18174474
- ライセンス: MIT
- 最終更新: 2026-08-26 (v1.5.5)

---

## バージョンノート (v1.5.5)

**v1.5.4 からの変更点:**
- エクスポート画像のマーカーが、表示中のオーバーレイ情報を使うように変更し、古い centroid セットが出力される問題を修正。
- マーカー色とラベルサイズを調整し、画面上の見た目に近く、保存画像で文字が重なりすぎないように改善。
- Online でのエクスポートは Image 座標系のままで維持し、元画像上の位置が正しく反映されるようにした。

**v1.5.4 からアップグレードする際の注意:**
- プロジェクト形式に互換性の破壊はないため、既存の `.pixy` はそのまま開ける。
- 表示された点と出力画像の点が、より一致するようになった。

---

## 概要

- 目的: 顕微鏡画像上の候補点と装置の fiducial を対応付け、画像座標を stage 座標へ変換して測定時間を短縮し、再現性を高める。
- 想定利用者: 微小領域分析のオペレーター、分析担当者、装置管理者
- 主な機能: 画像読込、粒子検出（K-means + connected components）、fiducial 入力、変換推定、残差可視化、CSV 出力 / クリップボードコピー
- 利点: 事前にターゲットを準備でき、装置側での手間が減り、出力画像の記録も一貫する

---

## はじめ方（インストールと起動）

- システム要件
  - OS: Windows を推奨（配布 EXE あり）。Linux / Mac ではソース実行可能。
  - Python: 3.8 以上（ソース実行時）
  - ディスク容量とメモリ: 画像サイズに依存する
- 依存ライブラリ: `requirements.txt` を参照（PySide6, OpenCV, NumPy など）
- ソースからのインストール

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python Main.py
```

- 配布 EXE を使う場合: ダウンロード後に実行すればよい。Windows によってブロックされる場合はプロパティから `Unblock` を実施。
- 注意: ソース実行時は仮想環境を有効化し、`requirements.txt` に合わせたバージョンを使う。

---

## クイックスタート（最小手順）

1. `PiXY_ver155.exe` を起動する。
2. `New Project` を押して画像を読み込む。
3. `START Centroid Extraction` を押す。
4. 画像上の粒子検出を行い、候補点を確認する。
5. `Add GroupN` から各グループの候補を中心リストに追加する。
6. `Finish Centroid Extraction` でオンライン整列に戻る。
7. `Add Fiducial Point` を押して fiducial を画像上で指定し、対応する stage 座標を入力する。
8. 3 点以上を登録して残差を確認する。
9. 問題がなければ `Export XYZ` または `Clipboard` で出力する。

---

## UI 概要

- メインボタン
  - 左上（プロジェクト操作）
    - `New Project`: 新しいプロジェクトを開始し画像を読み込む。
    - `Load Project`: 保存済み `.pixy` を読み込む。
    - `Save Project`: 現在の状態を `.pixy` で保存する。

  - 左中（fiducial 操作）
    - `Add Fiducial Point`: fiducial 登録モードに入り、画像上の点をクリックする。
    - `Update XY`: 選択した fiducial の stage 座標を更新する。
    - `Clear`: fiducial 行を削除またはクリアする。

  - 左下（粒子検出のパラメータ設定）
    - `START Centroid Extraction` / `Finish Centroid Extraction`: 抽出モードの開始・終了
    - 検出パラメータは `Advanced` モードで操作する。
    - `Recalculation Trigger` (`Auto` / `Manual`): 検出の再計算タイミングを制御する。
    - `Number of Groups (K)`: K-means のクラスタ数
    - `Boundary Offset`: 画像端のアーチファクトを除去
    - `Neck Separation`: 接触粒子の分離を強める
    - `Shape Complexity`: 形状が複雑な領域を抑制
    - `Particle Size Range (pix)`: 面積範囲でノイズを除去

  - 中央（候補点テーブル）
    - `Export XYZ`: target point の stage 座標を CSV で出力
    - `Clipboard`: 出力データをクリップボードへコピー
    - `Add Target`: 画像上に手動で target point を追加
    - `Update u, v`: 選択中点の位置を更新
    - `Clear`: target selection をクリア

---

## 座標系

PiXY には UI 上で次の座標モードがある。

- `Image` 座標: 読み込んだ画像のピクセル座標
- `Stage` 座標: 変換推定後の装置 stage 座標

画像のエクスポートでは、マーカー位置は元画像の Image 座標フレームに書き出される。これは、保存画像が画面に見えている位置と対応しているようにするためである。

---

## Fiducial Point Registration（オンラインモード）

- `Add Fiducial Point` を押して登録モードに入る。
- 画像上で fiducial をクリックする。
- 各 fiducial の stage 座標 (X, Y, Z) をテーブル入力する。
- 3 点以上を登録する。
- 残差を確認し、必要なら外れ値を除く。
- 変換が良好なら `Export XYZ` で target point の座標を出力する。

---

## ファイル形式

### プロジェクトファイル (.pixy)

`.pixy` は JSON ベースのプロジェクトアーカイブで、以下を含む。
- 画像データ
- 検出された target centroids とメタデータ
- fiducial 点と stage 座標
- パラメータ設定
- 抽出モード情報

### エクスポート形式 (CSV)

`Export XYZ` を押すと、CSV が出力される。一般的には次のような列を持つ。

```
Target_ID  Pixel_X  Pixel_Y  Stage_X  Stage_Y  Stage_Z
```

---

## 設定ファイル (pixy_settings.ini)

`pixy_settings.ini` では次を設定できる。
- Unified Control で使う Aggressiveness のプリセット
- UI の既定動作設定

---

## トラブルシューティング

### Centroids が検出されない / ノイズが多い

1. Particle Size Range を確認する。
2. `K` やセグメンテーション設定を調整する。
3. 画像のコントラストや品質を改善する。

### Fiducial 登録後に残差が大きい

1. stage 座標を再確認する。
2. fiducial を増やす。
3. 1 点だけ残差が大きければそこを除外する。

---

## ドキュメント

- `InstructionManual_EN_v1.5.5.md`
- `InstructionManual_JP_v1.5.5.md`
- `RELEASE_NOTES_v1.5.5.md`

---

## バージョン

**Current release: v1.5.5**
