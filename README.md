# Px2XY — Centroid Finder

Px2XY は画像をポスタリゼーション（色クラスタリング）して領域ごとの重心を検出・表示する GUI ツールです。
主に画像解析、顕微鏡画像の特徴点抽出、参照点設定・エクスポートに使えます。

## 主な機能
- 画像の読み込みと表示（多数のフォーマットをサポート）
- ポスタリゼーションによる領域クラスタリングと重心検出
- PyQt5 ベースの GUI（参照点の追加・編集、重心の選択、CSV 形式でのエクスポート等）
- 自動/手動更新モード、軽負荷モード対応

## 必要条件
- Python 3.8 以上
- 以下の主要パッケージ（詳細は `requirements.txt` を参照）
  - numpy
  - opencv-python
  - PyQt5

注: 実行環境や用途に応じて追加パッケージが必要になる場合があります。

## インストール
仮想環境を作成し、依存パッケージをインストールする例（PowerShell）:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## 実行方法（簡易）
GUI を起動するにはリポジトリルートで:

```powershell
python Main.py
```

Main.py をコマンドライン引数で画像パスを与えて起動できる場合は、次のように実行します:

```powershell
python Main.py path\to\image.jpg
```

## 使い方（概要）
- PosterLevel, Min Area, Trim を調整して興味ある領域の抽出精度を調整します。
- 参照点（Ref）を追加して座標を固定・保存できます。
- 重心はテーブルから選択、エクスポート可能です。

詳しい操作手順や図は `paper.md`（JOSS 投稿用）およびドキュメントで説明します。

## ライセンス
このプロジェクトは `LICENSE` に記載のライセンス下で公開されています（例: MIT）。

## 引用
このソフトウェアを使った研究を報告する場合は `CITATION.cff` を確認してください。

## 貢献
PR や Issue を歓迎します。コードスタイルやテストを整備した上でプルリクエストを送ってください。

## 作者
あなたの名前（適宜更新してください）
