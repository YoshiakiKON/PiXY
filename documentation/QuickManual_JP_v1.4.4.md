# クイックマニュアル — PiXY（日本語, v1.4.4）

短く、主要な操作をスクリーンショット付きで説明します。

## 概要
- 目的: 画像中の粒子重心を検出し、フィデューシャル点（fiducial point）でピクセル→実世界座標へ変換します。

※PiXY のフィデューシャル点は人工マーカーではなく、試料上の自然特徴点（傷、粒子先端など）を指します。

## 動作環境
- Python 3.10+
- 依存: requirements.txt を参照

## 起動
```powershell
python Main.py
```

## よく使う操作（スクショ参照）

1) 画像を開く
- `New Project` で画像を読み込みます。

![Open Image](images/quick_jp_1.png)

2) 重心抽出を開始する
- `START Centroid Extraction` を押します。
- 左パラメータを調整して重心を確認します。
- 現行仕様では `Shape Complexity` の既定値は `3` です。
- `Number of Groups` を変更すると、グループの `Show`/`Hide` 個別設定は解除され、全グループ表示に戻ります。
- Core/Rim の両方を表示するモードでは、同一粒子の Core-Rim が白線で接続表示されます。

![Detect Centroids](images/quick_jp_2.png)

3) グループを中カラムへ追加
- （任意）各 `Add GroupN` ボタン下の入力欄で手動グループ名を設定します。
- 左カードの `Add GroupN` で必要な点を中カラムへ追加します。
- `Number of Groups` を変更して再計算した場合、グループ名は色が近いグループへ引継ぎます。衝突時は元グループ点数が多い名前を優先します。

![Detect Centroids](images/quick_jp_2.png)

4) 抽出終了後にフィデューシャル点を追加
- `Finish Centroid Extraction` を押してから `Add Fiducial Point` を使います。
- 画像上でフィデューシャル点をクリックして観測点を追加します。

![Add Ref](images/quick_jp_3.png)

5) 変換とエクスポート
- フィデューシャルテーブルで画像座標 `u`/`v` と装置座標 `Stage X`/`Stage Y`/`Stage Z` を確認・編集してフィットを調整し、エクスポートでCSV出力できます。

![Export](images/quick_jp_4.png)

## トラブルシュート（簡易）
- 画像が読み込めない: 画像形式を確認。ppm, bmp, jpg 等を試してください。
- フィデューシャル点が合わない: フィデューシャル点を再度追加し、誤差（residual）を確認してください。
- `Replace Image` 後は、重心抽出モード中なら K-means が自動再計算され、左側の検出結果が新しい画像に同期されます。

---
※ 画像ファイルは `documentation/images/` に配置してください。スクリーンショットの撮影方法は `SCREENSHOT_GUIDE.md` を参照。

