# コード移植チェックリスト（v1.1.5 → v1.1.9）

方針
- docs/資産は後回し。まず **実行コードとEXE安定性** に関係する変更だけを段階導入する。
- 1ステップ = 1コミット = 1回スモーク（[SMOKE_TEST_EXE.md](../SMOKE_TEST_EXE.md)）

## Baseline（固定）
- [x] ベースラインは v1.1.5 相当コード（タグ v1.1.5）
- [ ] EXEビルド設定は v1.1.5相当（console/windowed 等をまず一致）

## Baselineホットフィックス（v1.1.5のまま修正）
- [x] 参照点テーブル（転置ビュー）の Stage X/Y/Z が単クリックで入力開始できる

## 復旧ステップ（低リスク → 高リスク）

### Step 1: v1.1.6 UIフッター（低リスク）
- [x] `Ui.py`: タイトルからバージョン除去、フッターに表示（見た目のみ）
- [x] `pyproject.toml`: version更新（挙動に影響なし）

### Step 2: v1.1.7 安定性ガード（低〜中リスク）
- [x] `Ui.py`: `_update_image_actual()` の poster未定義ガード（UnboundLocalError回避）
- [x] `Ui.py`: overlay/boundary 描画のガード（poster無し時に落ちない）

### Step 3: v1.1.7 処理分離/Manual制御（中〜高リスク）
- [x] `Ui.py`: posterizationとcentroid再計算の分離（Manualでも見た目は即時、重い計算は抑制）
- [x] `Ui.py`: Auto/Manual（ReCalculate）導入

### Step 4: v1.1.8 EXEリソース探索（低〜中リスク/安定化寄り）
- [x] `Main.py`: PyInstaller凍結時に `sys._MEIPASS` を優先

### Step 5: v1.1.8 しきい値デフォルト（中リスク/結果が変わる）
- [x] `Config.py`: `DEFAULT_MAX_GRAIN_AREA` / `DEFAULT_MIN_GRAIN_AREA`
- [x] `Ui.py`: ヒストグラム初期選択に上限/下限キャップ

### Step 6: v1.1.8 テーブルレイアウト（低〜中リスク）
- [ ] `tables.py`: 左テーブルの幅固定（追加でレイアウトが揺れない）

### Step 7: v1.1.8 PyInstaller spec（環境依存/中リスク）
- [ ] `PiXY.spec`: icon/console/データ同梱の更新（EXE挙動差に直結）

---

## メモ（不安定化したら）
- 直前ステップを `git revert` して原因を確定する（まとめて戻さない）
- 必要ならフラグ化してON/OFF比較できるようにする

---

## Centroid Filter Order Spec

- Current centroid filtering/evaluation order is fixed as follows:
	1. `Trim (Boundary Offset)` erosion on per-color mask
	2. Connected-components extraction
	3. Early `min_area` rejection for original components
	4. `Neck Separation` split
	5. Per-result `min/max area` filtering
	6. `Shape Complexity` filtering
	7. Centroid/rim extraction and boundary contour accumulation

- For split results, `Shape Complexity` is evaluated on each split component (post-split shape), not on the pre-split original component.

## Group Visibility Reset Spec

- When `Number of Groups` is changed, per-group show/hide state is reset and all groups become visible once.
- This reset applies to slider edits, +/- nudges, and Enter-confirmed text edits for group-count controls.
