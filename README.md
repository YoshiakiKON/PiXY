# PiXY

PiXY は画像中の粒子の重心を検出し、参照点（RefPoint）を使ってピクセル座標を実世界座標に変換するための GUI ツールです。

## 特徴
- 画像から重心（centroids）を抽出
- 参照点を追加・編集してアフィン / 類似変換を推定
- 転置表示の参照テーブルと即時プレビュー

## 必要環境
- Python 3.10+（環境に合わせて調整してください）
- 依存パッケージは `requirements.txt` を参照

## Documentation

ユーザ向けマニュアルとクイックスタート（日本語 / English）は `documentation/` にあります:

- `documentation/QuickManual_JP.md` — クイックマニュアル（日本語、スクリーンショットは `documentation/images/` に配置してください）
- `documentation/QuickManual_EN.md` — Quick Manual (English)
- `documentation/Manual_JP.md` — 詳細マニュアル（日本語）
- `documentation/Manual_EN.md` — Manual (English)
- `documentation/SCREENSHOT_GUIDE.md` — スクリーンショット撮影手順

クイックマニュアル用のスクリーンショットは `documentation/images/` に保存してください。自動キャプチャ用スクリプトも `documentation/capture_screenshots.py` として用意していますが、手動で撮影して頂いても問題ありません。

## インストール（開発用）
```powershell
# PiXY — Centroid to Real-World Coordinate Converter

PiXY detects region centroids in images and converts pixel coordinates to real-world coordinates using user-defined reference points. It provides an interactive GUI for centroid inspection, reference-point editing, and export.

## Key Features

- Detect centroids from segmented image regions (posterization/clustering).
- Add and edit reference points to estimate affine/similarity transforms.
- Interactive, transposed reference table view for quick editing and verification.
- Export coordinates and reference data (CSV supported).

## Requirements

- Python 3.10 or newer (adjust as needed for your environment).
- See `requirements.txt` for Python package dependencies.

## Install (development)

PowerShell example:

```powershell
cd C:\Python\PiXY
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## Run

From the repository root:

```powershell
python Main.py
```

You can optionally pass an image path:

```powershell
python Main.py path\to\image.jpg
```

## Basic Usage

- Open an image with `Open Image`.
- Use `Add Ref` to enter reference-point mode, then click image points to add reference observations.
- Edit reference `u`/`v` (image coordinates) and `Stage X`/`Stage Y` (and `Stage Z` if applicable) in the reference table to refine the transform.
- The left transposed table shows residuals and transformed coordinates for quick inspection.

## Development & Contribution

- Report bugs and feature requests via GitHub Issues.
- Pull requests are welcome — please include tests or reproduction steps when possible.

## Citation

If you publish results generated with this software, please cite this repository. If you use a Zenodo artifact/link for an archived release, record that link in `CITATION.cff`.

## License

See the `LICENSE` file in the repository for license terms (e.g. MIT).

---

## AI-Assisted Development / AI支援について

This project was developed using AI-assisted pair programming (GitHub Copilot / GPT-5, Google Gemini, xAI Grok, Anthropic Claude (Sonnet)). All outputs were reviewed and validated by the human author(s); AI tools are not listed as authors. See NOTICE.md for details.

本プロジェクトはAI支援ペアプログラミング（GitHub Copilot / GPT-5、Google Gemini、xAI Grok、Anthropic Claude（Sonnet））を活用して開発しました。生成物は最終的に人間の著者がレビュー・検証しており、AIは著者として記載していません。詳細は NOTICE.md を参照してください。

---

If you'd like, I can also:

- Commit this change and push a branch/PR.
- Produce a shorter `README-short.md` for display on PyPI/GitHub release pages.
- Draft a `paper.md` for JOSS submission using the repository metadata and release link (once available).

