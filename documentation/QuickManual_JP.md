# クイックマニュアル — PiXY（日本語）

短く、主要な操作をスクリーンショット付きで説明します。

## 概要
- 目的: 画像中の粒子重心を検出し、参照点でピクセル→実世界座標へ変換します。

## 動作環境
- Python 3.10+
- 依存: requirements.txt を参照

## 起動
```powershell
python Main.py
```

## よく使う操作（スクショ参照）

1) 画像を開く
- メニューまたは `Open Image` を選び、画像を読み込みます。

![Open Image](images/quick_jp_1.png)

2) 重心を検出する
- `Detect` または同等のボタンを押すと、画面上に重心が表示されます。

![Detect Centroids](images/quick_jp_2.png)

3) 参照点（RefPoint）の追加
- `Add Ref` をクリックして参照点モードに入り、画像上で参照点をクリックして観測点を追加します。

![Add Ref](images/quick_jp_3.png)

4) 変換とエクスポート
- 参照テーブルで画像座標 `u`/`v` と装置座標 `Stage X`/`Stage Y` を確認・編集してフィットを調整し、エクスポートでCSV出力できます。

![Export](images/quick_jp_4.png)

## トラブルシュート（簡易）
- 画像が読み込めない: 画像形式を確認。ppm, bmp, jpg 等を試してください。
- 参照点が合わない: 参照点を再度追加し、誤差（residual）を確認してください。

---
※ 画像ファイルは `documentation/images/` に配置してください。スクリーンショットの撮影方法は `SCREENSHOT_GUIDE.md` を参照。
