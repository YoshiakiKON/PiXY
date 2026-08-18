# PiXY v1.6 — 機能拡張仕様書（言語・テーマ・マルチレイヤー）

**対象版**: v1.6.0  
**作成日**: 2026-08-15  
**ステータス**: ロードマップ・仕様確定待ち

---

## 概要

v1.5.1 で Aggressiveness プリセットの ini ファイル管理を導入する。v1.6 では、これを基盤として言語切り替え・テーマ/色管理機能を追加する。

### 実装フェーズ

| フェーズ | v | 内容 |
|---|---|---|
| Phase 1 | 1.5.1 | ✅ **完了**: Aggressiveness プリセットを ini 管理 |
| Phase 2 | 1.6.0 | 📋 テーマ/色定義を JSON で管理 |
| Phase 3 | 1.6.0 | 📋 言語辞書を Strings.py に統合 |
| Phase 4 | 1.6.x | 📋 UI に「言語」「テーマ」選択メニュー実装 |
| Phase 5 | 1.6.0 | 📋 マルチレイヤー画像表示（軽量設計）|

### 1.5.3 からの持ち越し課題

- 1.5.3 では Add Target / Add Fiducial のオーバーレイ更新経路はまだ共通化されておらず、Add Target 側の状態更新が重い。1.6 では、手動ターゲット追加と参照点追加の後処理を整理して再現差を縮める。

---

## ファイル構成（v1.6 後）

```
Project Root
├── pixy_settings.ini          ← 拡張版（言語、デフォルト値など）
├── pixy_themes.json           ← **新規**：色パレット・テーマ定義
├── Config.py                  ← 拡張：JSON読み込みロジック追加
├── Strings.py                 ← 拡張：言語辞書統合
├── Ui.py                      ← 拡張：言語・テーマ選択UI
└── ...
```

---

## Phase 2: テーマ/色管理（pixy_themes.json）

### ファイル構造

**ファイル**: `pixy_themes.json`

```json
{
  "themes": {
    "Light": {
      "description": "Light theme (default)",
      "background": "#FFFFFF",
      "foreground": "#000000",
      "panel_background": "#F5F5F5",
      "primary": "#0066CC",
      "primary_hover": "#0052A3",
      "accent": "#FF6600",
      "accent_hover": "#CC5200",
      "success": "#00AA00",
      "warning": "#FFA500",
      "error": "#FF0000",
      "highlight_selected": "#FFEB3B",
      "highlight_selected_bg": "#FFF9C4",
      "border": "#CCCCCC",
      "grid_line": "#E0E0E0",
      "text_label": "#333333",
      "text_hint": "#666666",
      "disabled": "#CCCCCC"
    },
    "Dark": {
      "description": "Dark theme",
      "background": "#1E1E1E",
      "foreground": "#E0E0E0",
      "panel_background": "#2D2D2D",
      "primary": "#66B2FF",
      "primary_hover": "#4D94E6",
      "accent": "#FF9933",
      "accent_hover": "#E67E22",
      "success": "#66FF66",
      "warning": "#FFD700",
      "error": "#FF6666",
      "highlight_selected": "#FFD700",
      "highlight_selected_bg": "#4D4D00",
      "border": "#444444",
      "grid_line": "#333333",
      "text_label": "#E0E0E0",
      "text_hint": "#999999",
      "disabled": "#555555"
    },
    "HighContrast": {
      "description": "High contrast theme for accessibility",
      "background": "#000000",
      "foreground": "#FFFFFF",
      "panel_background": "#1A1A1A",
      "primary": "#FFFF00",
      "primary_hover": "#CCCC00",
      "accent": "#00FF00",
      "accent_hover": "#00CC00",
      "success": "#00FF00",
      "warning": "#FF00FF",
      "error": "#FF0000",
      "highlight_selected": "#00FFFF",
      "highlight_selected_bg": "#004040",
      "border": "#FFFFFF",
      "grid_line": "#444444",
      "text_label": "#FFFFFF",
      "text_hint": "#CCCCCC",
      "disabled": "#666666"
    }
  }
}
```

### 実装場所: Config.py 拡張

```python
def load_theme_colors(theme_name=None):
    """
    pixy_themes.json から指定テーマの色パレットを読み込む
    
    Args:
        theme_name: テーマ名 ('Light', 'Dark', 'HighContrast')
                   None の場合は ini から読み込んだデフォルトを使用
    
    Returns:
        dict: {
            'background': '#FFFFFF',
            'foreground': '#000000',
            ...
        }
    """
    import json
    
    DEFAULT_THEMES = {...}  # 上述のデフォルト色定義
    themes_file = "pixy_themes.json"
    
    if theme_name is None:
        theme_name = load_ui_defaults().get('theme', 'Light')
    
    try:
        if not os.path.exists(themes_file):
            if DEBUG:
                print(f"[Config] {themes_file} が見つかりません。デフォルト値を使用します。")
            themes = DEFAULT_THEMES
        else:
            with open(themes_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                themes = data.get('themes', DEFAULT_THEMES)
    except Exception as e:
        print(f"[Config] Theme loading failed: {e}")
        themes = DEFAULT_THEMES
    
    return themes.get(theme_name, themes.get('Light', DEFAULT_THEMES['Light']))


def get_current_theme_colors():
    """
    現在のテーマ（ini から読み込まれた）の色パレット全体を取得
    """
    return load_theme_colors()


def get_theme_color(color_key, theme_name=None):
    """
    指定テーマから特定の色を取得
    
    Args:
        color_key: 色キー ('background', 'primary', 'error' など)
        theme_name: テーマ名（None なら現在のテーマ）
    
    Returns:
        str: Hex カラーコード（例: '#FFFFFF'）
    """
    colors = load_theme_colors(theme_name)
    return colors.get(color_key, '#000000')


# グローバル変数として暴露
THEME_COLORS = get_current_theme_colors()
```

### Ui.py での使用例

```python
from Config import get_theme_color, THEME_COLORS

# ウィジェット作成時
self.setStyleSheet(f"""
    QWidget {{
        background-color: {THEME_COLORS['background']};
        color: {THEME_COLORS['foreground']};
    }}
    QPushButton {{
        background-color: {THEME_COLORS['primary']};
        color: {THEME_COLORS['foreground']};
    }}
    QPushButton:hover {{
        background-color: {THEME_COLORS['primary_hover']};
    }}
    QLineEdit {{
        background-color: {THEME_COLORS['panel_background']};
        border: 1px solid {THEME_COLORS['border']};
    }}
""")

# 画像レンダリング時
overlay_color = get_theme_color('primary')  # 現在のテーマで取得
highlight_color = get_theme_color('highlight_selected')
```

---

## Phase 3: 言語管理（Strings.py 拡張）

### 現在の構造（v1.5.1）

```python
# Strings.py
APP_TITLE = "PiXY"
STEP1_LABEL = "Step 1: Off-line Targeting"
```

### 目標構造（v1.6）

```python
# Strings.py - 拡張版

from Config import LANGUAGE

# 言語辞書（すべてのUI文字列を一元管理）
LANGUAGE_DICT = {
    'ja': {
        'APP_TITLE': 'PiXY',
        'STEP1_LABEL': 'ステップ1: オフライン対象設定',
        'STEP1_SUBLABEL': '計測前に画像を読み込んで対象点を指定します',
        'STEP2_LABEL': 'ステップ2: オンライン位置合わせ',
        'STEP2_SUBLABEL': '計測装置でフィデューシャル点を登録し、座標をエクスポート',
        'BUTTON_OPEN_IMAGE': 'イメージ開く',
        'BUTTON_EXPORT_CENTROIDS': '重心をエクスポート',
        'BUTTON_ADD_REF': 'フィデューシャル点を追加',
        'BUTTON_CLEAR': 'クリア',
        # ... 全UI文字列
    },
    'en': {
        'APP_TITLE': 'PiXY',
        'STEP1_LABEL': 'Step 1: Off-line Targeting',
        'STEP1_SUBLABEL': 'Load image & specify target points before using the instrument',
        'STEP2_LABEL': 'Step 2: On-line Alignment',
        'STEP2_SUBLABEL': 'Register fiducial points on the instrument, then export coordinates',
        'BUTTON_OPEN_IMAGE': 'Open Image',
        'BUTTON_EXPORT_CENTROIDS': 'Export Centroids',
        'BUTTON_ADD_REF': 'Add Fiducial Point',
        'BUTTON_CLEAR': 'Clear',
        # ... 全UI文字列
    }
}

# ヘルパー関数
def get_string(key, language=None):
    """
    UI文字列を言語別に取得
    
    Args:
        key: 文字列キー（例: 'APP_TITLE'）
        language: 言語コード ('ja', 'en')。None なら Config.LANGUAGE を使用
    
    Returns:
        str: ローカライズされた文字列
    """
    if language is None:
        language = LANGUAGE
    
    return LANGUAGE_DICT.get(language, {}).get(key, key)


# 互換性維持のためトップレベルに暴露
APP_TITLE = get_string('APP_TITLE')
STEP1_LABEL = get_string('STEP1_LABEL')
STEP1_SUBLABEL = get_string('STEP1_SUBLABEL')
...

# または、動的にプロキシする場合（言語変更時にも反応）
class LanguageProxy:
    """言語変更時に自動的に文字列を更新する"""
    def __getattr__(self, key):
        return get_string(key)

# 使用例
# STR.APP_TITLE  # 現在の言語で取得
```

### Ui.py での使用例

```python
from Strings import get_string, LANGUAGE_PROXY

# 言語切り替え時
def on_language_changed(new_language):
    # Config から言語を更新
    Config.LANGUAGE = new_language
    
    # UI のテキストをリセット
    self.setWindowTitle(get_string('APP_TITLE'))
    self.label_step1.setText(get_string('STEP1_LABEL'))
    self.label_step1_hint.setText(get_string('STEP1_SUBLABEL'))
    ...
```

---

## Phase 4: UI 実装（言語・テーマ選択メニュー）

### メニュー構成

**メニューバーに追加**:

```
Settings
├─ Language
│  ├─ 日本語 (ja) [●]
│  └─ English (en)
├─ Theme
│  ├─ Light [●]
│  ├─ Dark
│  └─ High Contrast
└─ Preferences... (将来予約)
```

### 実装パターン

```python
# Ui.py - __init__ メソッド内

# === メニューバー ===
menubar = self.menuBar()

# Settings メニュー
settings_menu = menubar.addMenu("Settings")

# Language サブメニュー
language_menu = settings_menu.addMenu("Language")
self.action_lang_ja = language_menu.addAction("日本語")
self.action_lang_en = language_menu.addAction("English")
self.action_lang_ja.triggered.connect(lambda: self._on_language_changed('ja'))
self.action_lang_en.triggered.connect(lambda: self._on_language_changed('en'))

# チェックマークをつける（現在の言語）
self._update_language_menu()

# Theme サブメニュー
theme_menu = settings_menu.addMenu("Theme")
self.action_theme_light = theme_menu.addAction("Light")
self.action_theme_dark = theme_menu.addAction("Dark")
self.action_theme_hc = theme_menu.addAction("High Contrast")
self.action_theme_light.triggered.connect(lambda: self._on_theme_changed('Light'))
self.action_theme_dark.triggered.connect(lambda: self._on_theme_changed('Dark'))
self.action_theme_hc.triggered.connect(lambda: self._on_theme_changed('HighContrast'))

# チェックマークをつける（現在のテーマ）
self._update_theme_menu()


def _on_language_changed(self, language_code):
    """言語を変更"""
    import Config
    
    Config.LANGUAGE = language_code
    Config.save_user_setting('language', language_code)
    
    # UI テキストを全て更新
    self._retranslate_ui()
    self._update_language_menu()


def _on_theme_changed(self, theme_name):
    """テーマを変更"""
    import Config
    
    Config.THEME_COLORS = Config.get_theme_colors(theme_name)
    Config.save_user_setting('theme', theme_name)
    
    # スタイルシートを更新
    self._apply_theme()
    self._update_theme_menu()


def _retranslate_ui(self):
    """全 UI テキストを現在の言語で更新"""
    from Strings import get_string
    
    self.setWindowTitle(get_string('APP_TITLE'))
    self.label_step1.setText(get_string('STEP1_LABEL'))
    # ... 全テキスト更新


def _apply_theme(self):
    """スタイルシートを現在のテーマで更新"""
    from Config import THEME_COLORS
    
    self.setStyleSheet(f"""
        QWidget {{ background-color: {THEME_COLORS['background']}; }}
        ...
    """)


def _update_language_menu(self):
    """言語メニューにチェックマークを付ける"""
    from Config import LANGUAGE
    
    self.action_lang_ja.setCheckable(True)
    self.action_lang_en.setCheckable(True)
    self.action_lang_ja.setChecked(LANGUAGE == 'ja')
    self.action_lang_en.setChecked(LANGUAGE == 'en')


def _update_theme_menu(self):
    """テーマメニューにチェックマークを付ける"""
    # 現在のテーマを ini から取得
    current_theme = Config.load_ui_defaults().get('theme', 'Light')
    
    self.action_theme_light.setCheckable(True)
    self.action_theme_dark.setCheckable(True)
    self.action_theme_hc.setCheckable(True)
    self.action_theme_light.setChecked(current_theme == 'Light')
    self.action_theme_dark.setChecked(current_theme == 'Dark')
    self.action_theme_hc.setChecked(current_theme == 'HighContrast')
```

### 設定の永続化

**Config.py に追加**:

```python
def save_user_setting(key, value):
    """
    ユーザー設定を pixy_settings.ini に保存
    
    Args:
        key: 設定キー ('language', 'theme' など)
        value: 設定値
    """
    import configparser
    
    settings_file = "pixy_settings.ini"
    config = configparser.ConfigParser()
    
    if os.path.exists(settings_file):
        config.read(settings_file, encoding='utf-8')
    
    # セクション別に設定
    if key == 'language':
        if not config.has_section('Language'):
            config.add_section('Language')
        config.set('Language', 'language', value)
    
    elif key == 'theme':
        if not config.has_section('UI'):
            config.add_section('UI')
        config.set('UI', 'theme', value)
    
    try:
        with open(settings_file, 'w', encoding='utf-8') as f:
            config.write(f)
        if DEBUG:
            print(f"[Config] 設定を保存しました: {key}={value}")
    except Exception as e:
        print(f"[Config] 設定保存失敗: {e}")
```

---

## 設定ファイル更新方針

### pixy_settings.ini (v1.6)

```ini
[Language]
# UI language: ja, en
language = ja

[Aggressiveness]
# Phase 1 で追加済み
preset_0  = 0, 0, 3
preset_1  = 1, 0, 3
...

[UI]
# Phase 1 で追加済み
default_aggressiveness = 5
unified_control_default = True

# Phase 2 で追加
theme = Light

# Phase 4 で追加予定
# window_width = 1200
# window_height = 900
# remember_last_project = True
```

---

## 実装チェックリスト（v1.6 のタスク）

### Phase 2: テーマ管理

- [ ] `pixy_themes.json` を作成
- [ ] `Config.py` に `load_theme_colors()` を追加
- [ ] `Config.py` に `get_theme_color()` を追加
- [ ] `THEME_COLORS` グローバル変数を暴露
- [ ] Ui.py でテーマ色を使用するよう更新
- [ ] rendering.py で画像オーバーレイ色をテーマから取得

### Phase 3: 言語管理

- [ ] `Strings.py` に `LANGUAGE_DICT` を追加（ja/en）
- [ ] `Strings.py` に `get_string()` ヘルパー関数を追加
- [ ] `Config.py` に `load_language()` ロジックを確認
- [ ] `Config.py` に `LANGUAGE` グローバル変数を暴露
- [ ] Ui.py 全体で文字列を `get_string()` で取得するよう修正
- [ ] Strings.py の既存グローバル定数を `get_string()` で初期化

### Phase 4: UI 実装

- [ ] メニューバーに Settings メニューを追加
- [ ] Language サブメニュー（ja, en）を実装
- [ ] Theme サブメニュー（Light, Dark, HighContrast）を実装
- [ ] `_on_language_changed()` メソッドを実装
- [ ] `_on_theme_changed()` メソッドを実装
- [ ] `_retranslate_ui()` メソッドを実装（全テキスト更新）
- [ ] `_apply_theme()` メソッドを実装（全色更新）
- [ ] `save_user_setting()` をテスト
- [ ] 設定がアプリ再起動後も保持されることを確認

---

## テスト計画

### 単体テスト

```python
# test_config.py
def test_load_theme_colors():
    colors = Config.load_theme_colors('Dark')
    assert colors['background'] == '#1E1E1E'
    assert 'primary' in colors

def test_get_theme_color():
    color = Config.get_theme_color('error', 'Light')
    assert color == '#FF0000'

def test_load_language():
    lang = Config.load_language()
    assert lang in ['ja', 'en']

def test_get_string():
    s = Strings.get_string('APP_TITLE', 'ja')
    assert s == 'PiXY'
    s = Strings.get_string('APP_TITLE', 'en')
    assert s == 'PiXY'
```

### 統合テスト

- [ ] 言語を ja → en に切り替え、UI テキストが全て英語に変わること
- [ ] テーマを Light → Dark に切り替え、UI 色が全て暗くなること
- [ ] 設定を変更 → アプリ再起動 → 設定が保持されていること
- [ ] ini ファイルが破損した場合、デフォルト値で起動すること
- [ ] themes.json が破損した場合、デフォルトテーマで起動すること

---

## 注意事項

1. **後方互換性**: 既存の `.pixy` ファイルが v1.6 でも読み込める必要あり
2. **デフォルト値**: ini/json ファイルがない場合も起動できるようにする
3. **エラーハンドリング**: ファイル読み込み失敗時は静かにデフォルト値を使用
4. **パフォーマンス**: 言語・テーマ切り替え時の遅延を最小化（キャッシング活用）

---

## 参考リンク

- 作成日時点のプロジェクト構成: v1.5.1
- Aggressiveness 実装: [Config.py の load_aggressiveness_presets()](Config.py#L60-L80)
- 既存の Strings.py: [Strings.py](Strings.py#L1-L50)

---

## Phase 5: マルチレイヤー画像表示（軽量設計）

### 背景と現在の構造

v1.5.1 で「3レイヤー構造」を導入済み：

```
Layer 1 (_display_pm_base)  : ベース画像 + グリッド + 全ポイント描画
                               → 再描画トリガー: リスト変更・ズーム・回転・画像変更
Layer 2 (Selection overlay) : 選択点を前面に大きく描画
                               → 再描画トリガー: 選択変更のみ
```

この構造を汎用的なマルチレイヤースタックに拡張する。

---

### 設計方針：「ダーティフラグ＋QPixmapキャッシュ」

**核心原則**:
- 各レイヤーは `QPixmap` として独立にキャッシュ
- 変更があったレイヤーのみ再描画（ダーティフラグ管理）
- 合成（コンポジット）は軽量な QPainter で行い、numpy/OpenCV を使わない
- スクロール・ズーム時も再計算は最小限に

```
重い処理（numpy/cv2）
  → 各レイヤーの QPixmap キャッシュに書き出す（変更時のみ）

軽い処理（QPainter）
  → キャッシュ済みの QPixmap を合成して表示
  → 毎フレーム行っても問題ないレベルの速度
```

---

### レイヤー定義

```python
# レイヤーID と再描画トリガーの対応表
LAYER_BASE       = 0  # ベース画像（回転・グリッド）
                      # トリガー: 画像変更、ズーム、回転、Flip
LAYER_DETECTIONS = 1  # 重心抽出結果（全候補点）
                      # トリガー: 検出パラメータ変更、K-means再計算
LAYER_TARGETS    = 2  # Target Points（全Target点、通常サイズ）
                      # トリガー: ポイントリスト変更（追加・削除）
LAYER_FIDUCIALS  = 3  # Fiducial Points（全Fiducial点、ラベル付き）
                      # トリガー: Fiducialsリスト変更
LAYER_SELECTION  = 4  # 選択ハイライト（前面、大きいドット）
                      # トリガー: 選択変更のみ
# 将来の拡張用
LAYER_ANNOTATIONS = 5 # 手動アノテーション（コメント、矢印など）
LAYER_OVERLAY_EXT = 6 # 外部画像オーバーレイ（別画像の重ね合わせ）
```

---

### クラス設計

#### `LayerCache` クラス（rendering.py または新規 `layers.py`）

```python
class LayerCache:
    """
    マルチレイヤー QPixmap キャッシュ管理
    
    各レイヤーは独立した QPixmap にキャッシュされ、
    ダーティフラグが立っているときのみ再描画される。
    合成は QPainter で行い、全体を毎フレーム再合成しても軽量。
    """
    
    def __init__(self, layer_count=7):
        # {layer_id: QPixmap or None}
        self._cache: dict[int, QPixmap | None] = {i: None for i in range(layer_count)}
        self._dirty: dict[int, bool] = {i: True for i in range(layer_count)}
        self._visible: dict[int, bool] = {i: True for i in range(layer_count)}
        self._opacity: dict[int, float] = {i: 1.0 for i in range(layer_count)}
        # ブレンドモード: 'normal' または 'multiply'
        self._blend_mode: dict[int, str] = {i: 'normal' for i in range(layer_count)}
        self._canvas_size: tuple[int, int] = (0, 0)
    
    def mark_dirty(self, layer_id: int):
        """指定レイヤーをダーティに（次のcompose時に再描画）"""
        self._dirty[layer_id] = True
    
    def set_pixmap(self, layer_id: int, pm: QPixmap):
        """レイヤーの QPixmap を更新しダーティを解除"""
        self._cache[layer_id] = pm
        self._dirty[layer_id] = False
    
    def set_visible(self, layer_id: int, visible: bool):
        """レイヤーの表示/非表示を切り替え（再描画不要）"""
        self._visible[layer_id] = visible
    
    def set_opacity(self, layer_id: int, opacity: float):
        """レイヤーの不透明度を設定（再描画不要、合成時に適用）"""
        self._opacity[layer_id] = max(0.0, min(1.0, opacity))
    
    def set_blend_mode(self, layer_id: int, mode: str):
        """ブレンドモードを設定: 'normal' または 'multiply'（再描画不要）"""
        assert mode in ('normal', 'multiply'), f"Unknown blend mode: {mode}"
        self._blend_mode[layer_id] = mode
    
    # QPainter.CompositionMode の対応表
    _BLEND_MODE_MAP = {
        'normal':   QPainter.CompositionMode_SourceOver,  # 通常合成（アルファブレンド）
        'multiply': QPainter.CompositionMode_Multiply,    # 乗算（Photoshop の乗算と同等）
    }
    
    def compose(self) -> QPixmap | None:
        """
        全可視レイヤーを QPainter で合成して返す。
        
        重い処理（numpy/cv2）は各レイヤーの set_pixmap() で事前に完了済み。
        ここは純粋な QPainter合成のみ → 軽量・高速。
        """
        visible_entries = [
            (i, self._cache[i], self._opacity[i], self._blend_mode.get(i, 'normal'))
            for i in sorted(self._cache.keys())
            if self._visible.get(i) and self._cache[i] is not None
        ]
        if not visible_entries:
            return None
        
        # キャンバスサイズは最大レイヤーサイズに合わせる
        max_w = max(pm.width() for _, pm, _, _ in visible_entries)
        max_h = max(pm.height() for _, pm, _, _ in visible_entries)
        
        result = QPixmap(max_w, max_h)
        result.fill(Qt.transparent)
        
        painter = QPainter(result)
        try:
            for _, pm, opacity, blend_mode in visible_entries:
                composition = self._BLEND_MODE_MAP.get(
                    blend_mode, QPainter.CompositionMode_SourceOver
                )
                painter.setCompositionMode(composition)
                painter.setOpacity(opacity)
                painter.drawPixmap(0, 0, pm)
        finally:
            painter.end()
        
        return result
    
    def is_dirty(self, layer_id: int) -> bool:
        return self._dirty.get(layer_id, True)
    
    def invalidate_all(self):
        """全レイヤーをダーティに（ズームやキャンバスサイズ変更時）"""
        for k in self._dirty:
            self._dirty[k] = True
```

---

### Ui.py での統合パターン

#### 初期化

```python
# __init__ 内で
from rendering import LayerCache
from rendering import LAYER_BASE, LAYER_DETECTIONS, LAYER_TARGETS, LAYER_FIDUCIALS, LAYER_SELECTION

self._layer_cache = LayerCache()
```

#### レイヤーごとの再描画関数

```python
def _rebuild_layer_base(self):
    """ベース画像レイヤーを再描画（重い処理）"""
    if not self._layer_cache.is_dirty(LAYER_BASE):
        return
    
    pm = build_base_layer(
        self.proc_img,
        self.proc_zoom,
        self.view_padding,
        rotation=self.image_rotate,
        grid=self.show_grid,
    )
    self._layer_cache.set_pixmap(LAYER_BASE, pm)


def _rebuild_layer_targets(self):
    """Target Points レイヤーを再描画"""
    if not self._layer_cache.is_dirty(LAYER_TARGETS):
        return
    
    pm = build_points_layer(
        size=self._layer_cache.canvas_size,
        centroids=self._get_all_centroids(),
        colors=self._get_group_colors(),
        display_offset=self._display_offset,
    )
    self._layer_cache.set_pixmap(LAYER_TARGETS, pm)


def _rebuild_layer_fiducials(self):
    """Fiducial Points レイヤーを再描画"""
    if not self._layer_cache.is_dirty(LAYER_FIDUCIALS):
        return
    
    pm = build_fiducials_layer(
        size=self._layer_cache.canvas_size,
        ref_points=self.ref_points,
        display_offset=self._display_offset,
    )
    self._layer_cache.set_pixmap(LAYER_FIDUCIALS, pm)


def _rebuild_layer_selection(self):
    """選択ハイライトのみ再描画（最も頻繁、最も軽量）"""
    if not self._layer_cache.is_dirty(LAYER_SELECTION):
        return
    
    pm = build_selection_layer(
        size=self._layer_cache.canvas_size,
        selected_index=self.selected_index,
        centroids=self._get_all_centroids(),
        display_offset=self._display_offset,
    )
    self._layer_cache.set_pixmap(LAYER_SELECTION, pm)
```

#### 統合エントリポイント

```python
def _apply_proc_zoom(self):
    """フルレンダリング（全レイヤーをダーティに）"""
    self._layer_cache.invalidate_all()
    self._compose_and_display()


def _refresh_list_and_selection(self):
    """ポイントリスト変更時（TargetsとSelectionのみ再描画）"""
    self._layer_cache.mark_dirty(LAYER_TARGETS)
    self._layer_cache.mark_dirty(LAYER_FIDUCIALS)
    self._layer_cache.mark_dirty(LAYER_SELECTION)
    self._compose_and_display()


def _refresh_selected_overlay_only(self):
    """選択変更のみ（Selectionレイヤーのみ再描画）← 最も軽量"""
    self._layer_cache.mark_dirty(LAYER_SELECTION)
    self._compose_and_display()


def _compose_and_display(self):
    """ダーティなレイヤーを再描画して合成・表示"""
    # ダーティなレイヤーのみ再描画（重い処理）
    self._rebuild_layer_base()
    self._rebuild_layer_detections()
    self._rebuild_layer_targets()
    self._rebuild_layer_fiducials()
    self._rebuild_layer_selection()
    
    # 全レイヤーを合成（QPainter、軽い）
    pm = self._layer_cache.compose()
    if pm is not None:
        self.img_label_proc.setPixmap(pm)
```

---

### ダーティフラグの管理規則

| 変更イベント | mark_dirty するレイヤー |
|---|---|
| 画像変更・ズーム・回転 | **全レイヤー** (invalidate_all) |
| ポイント追加・削除 | TARGETS, FIDUCIALS, SELECTION |
| 選択変更のみ | SELECTION のみ |
| 検出パラメータ変更 | DETECTIONS のみ |
| グリッド表示切り替え | BASE のみ |
| Fiducial Show/Hide | FIDUCIALS, SELECTION |
| レイヤー可視性変更 | **なし**（compose 時に skip） |

---

### UI: レイヤーパネル

左パネルに「Layers」セクションを追加：

```
┌─ Layers ────────────────────────────────┐
│ ☑ Base Image        [100%] [Normal   ▼] │
│ ☑ Detections        [100%] [Normal   ▼] │
│ ☑ Target Points     [100%] [Normal   ▼] │
│ ☑ Fiducial Points   [100%] [Normal   ▼] │
│ ☑ Selection         [100%] [Normal   ▼] │
│ ─────────────────────────────────────── │
│ ☐ External Overlay  [ 70%] [Multiply ▼] │  ← 将来拡張
└─────────────────────────────────────────┘
```

**実装**: `QCheckBox` + `QSlider`（不透明度）+ `QComboBox`（ブレンドモード）のリスト。  
可視性・不透明度・ブレンドモードの変更はいずれも再描画不要（合成時に反映）。

#### ブレンドモードの選択肢

| モード名 | QPainter 定数 | 用途 |
|---|---|---|
| **Normal** | `CompositionMode_SourceOver` | 通常の半透明合成（デフォルト） |
| **Multiply** | `CompositionMode_Multiply` | Photoshopの乗算。白は透過、暗い色が重なって表示される。BSEや透過画像のオーバーレイに有効 |

#### Multiply の実用例

- **2枚の顕微鏡画像を重ねる**: 同じ試料の BSE 画像と EDS マップを乗算で重ね、明るい元素分布が暗い粒子位置に重なって見える
- **外部アノテーション**: 白背景に描かれた手書きスケッチを Multiply で重ねると、白い部分が透過して画像だけが見える

#### UI イベント: ブレンドモード切り替え

```python
def _on_blend_mode_changed(self, layer_id: int, mode_text: str):
    mode = 'multiply' if mode_text == 'Multiply' else 'normal'
    self._layer_cache.set_blend_mode(layer_id, mode)
    # compose() のみ呼び出せばよい（再描画不要）
    pm = self._layer_cache.compose()
    if pm:
        self.img_label_proc.setPixmap(pm)
```

---

### パフォーマンス上の注意事項

1. **numpy/cv2 処理は `_rebuild_layer_*` 内だけ**  
   `compose()` は純粋 QPainter のみ → GUI スレッドで呼んでも安全

2. **大きい画像の LAYER_BASE**  
   現在の `build_zoomed_canvas()` と同じく MAX_PIXELS キャップを設ける  
   ズーム倍率が変わるたびに再描画は必要だが、他のレイヤーへの影響なし

3. **LAYER_SELECTION は超軽量**  
   選択変更時は SELECTION のみ再描画（1点の円を QPixmap に描くだけ）  
   ユーザー操作に対するレスポンスが向上

4. **将来の非同期化**  
   `_rebuild_layer_detections()` は K-means を含むため重い  
   将来的に `QThread` で非同期化する余地を残す  
   完了シグナルで `mark_dirty(LAYER_DETECTIONS)` + `_compose_and_display()` を呼ぶ

5. **外部オーバーレイ（LAYER_OVERLAY_EXT）**  
   別の顕微鏡画像や参照画像を半透明で重ね合わせる用途  
   `set_opacity()` で透明度を調整するだけで合成が変わる  
   データは `QPixmap` として外部から注入（重い処理はユーザーが実施）

---

### 実装チェックリスト（v1.6 Phase 5）

- [ ] `LayerCache` クラスを `rendering.py` に実装
- [ ] レイヤーID 定数を定義（`LAYER_BASE` 等）
- [ ] `build_base_layer()` を `build_zoomed_canvas()` から分離
- [ ] `build_points_layer()` を実装（Target Points 専用）
- [ ] `build_fiducials_layer()` を実装（Fiducials 専用）
- [ ] `build_selection_layer()` を実装（選択ハイライト専用）
- [ ] Ui.py を `_compose_and_display()` パターンに移行
- [ ] `_apply_proc_zoom()` → `invalidate_all()` + `_compose_and_display()`
- [ ] `_refresh_list_and_selection()` → 対象レイヤーのみ dirty
- [ ] `_refresh_selected_overlay_only()` → SELECTION のみ dirty
- [ ] レイヤーパネル UI を実装（可視性チェックボックス + 不透明度スライダー + ブレンドモード）
- [ ] プロジェクト保存時にレイヤー状態を保存

---

## Phase 5 補足: Offline マルチレイヤー画像の詳細設計

### Offline オーバーレイレイヤーの仕様

**ユースケース**: 概観画像（ベース）に複数の拡大サブ画像を貼り合わせて、ターゲット設定の参照として利用する。

**前提**:
- Offline モードでのみ有効（Online モードではベース画像のみ使用）
- 回転は **不要**（概観 + サブ画像の位置合わせは並進・スケールのみ）
- 参照点（各レイヤー上の対応点）からトランスフォームを計算

### レンダリングパイプライン（Offline オーバーレイ専用）

```
回転なし → warpAffine 不要 → crop + resize のみ

ビューポート（物理座標μm）
  ↓ 物理座標 → ネイティブpx座標に変換（スケール・並進のみ）
  ↓ native_img[y0:y1, x0:x1]  ← ビューポートに対応する矩形クロップ
  ↓
  ├── scale < 1.0（縮小）: cv2.resize(INTER_AREA) → スクリーン解像度まで
  └── scale >= 1.0（拡大）: cv2.resize(INTER_NEAREST) → ネイティブ1:1で停止

処理量は常に O(viewport_pixels)（元画像サイズに依存しない）
```

### 座標系ルール（最重要）

**全ての Target Point 座標はベース画像（Layer 0）座標系で記録する。**

```python
def _on_image_click(self, screen_x, screen_y):
    # ユーザーがどのレイヤーを見ていても、クリック座標は常に Layer 0 基準に変換
    base_px_x, base_px_y = self._screen_to_base_px(screen_x, screen_y)
    self._add_target_point(base_px_x, base_px_y)
```

この1点を守れば、Offline/Online 遷移でのずれは原理的に発生しない。

### ズーム上限

各レイヤーのズーム上限は「ネイティブ解像度まで」:

```
scale >= 1.0 の時点でズームを停止（1ネイティブpx ≥ 1スクリーンpx）
縮小時はスクリーン解像度以上のデータは不要（INTER_AREA で捨てる）
```

### Offline → Online 遷移設計

**Phase A（v1.6 初期実装）: ベース画像のみ引き継ぎ**

```python
def _transition_offline_to_online(self):
    # 座標系（常に必要・変更なし）
    self._online_base_img = self._layer_cache.get_layer_img(LAYER_BASE)
    self._online_targets = self._target_points  # ベース画像座標で記録済み

    # Phase A: 表示もベース画像のみ
    self._online_display_img = self._online_base_img
```

**Phase B（バグなし確認後）: 合成スナップショットを表示に使用**

```python
def _transition_offline_to_online(self):
    self._online_base_img = self._layer_cache.get_layer_img(LAYER_BASE)
    self._online_targets = self._target_points

    # Phase B: 全レイヤーの合成を表示に使う（座標計算には使わない）
    self._online_display_img = self._layer_cache.compose_snapshot(
        zoom=self.proc_zoom  # 現在のズームレベルで合成
    )
    # ↑ 表示専用。Fiducial クリック座標は _online_base_img 基準で計算し続ける。
```

Phase A/B は1行のコメントアウトで切り替え可能。

### ブレンドモード（Offline オーバーレイ専用）

サブ画像にはブレンドモードを個別設定可能:

| モード | 用途 |
|---|---|
| **Normal** (SourceOver) | 通常の半透明重ね合わせ |
| **Multiply** | BSE + EDS マップ等、暗い部分を強調して重ねる |

ブレンドモード変更は再描画不要（`compose()` 呼び出しのみ）。

### 参照点によるトランスフォーム計算

各サブ画像の配置は「2点以上の対応点」から自動計算:

```python
@dataclass
class OfflineLayerTransform:
    offset_x_um: float = 0.0   # 物理座標系での X オフセット
    offset_y_um: float = 0.0   # 物理座標系での Y オフセット
    scale: float = 1.0         # スケール（native_px_per_um から導出）
    flip_h: bool = False
    flip_v: bool = False
    # 注意: rotation は Offline モードでは使用しない

def calc_offline_transform(
    ref_pts_layer_px: list[tuple],  # サブ画像上の参照点（px）
    ref_pts_base_um: list[tuple],   # 対応するベース画像上の物理座標（μm）
    native_px_per_um: float,
) -> OfflineLayerTransform:
    """
    最小2点の対応点からトランスフォームを計算（回転なし・スケール＋並進のみ）。
    PiXY の既存 stage transform 計算ロジックを回転なし版として流用可能。
    """
    ...
```

### 実装チェックリスト（Offline オーバーレイ追加分）

- [ ] `OfflineLayerTransform` データクラスを定義
- [ ] `calc_offline_transform()` を実装（最小2点からスケール・並進を計算）
- [ ] `render_offline_layer()` を実装（crop + resize、回転なし）
- [ ] ズーム上限を `native_px_per_um` から動的に計算
- [ ] `_screen_to_base_px()` を拡張（サブ画像レイヤー経由のクリック座標をベース座標に変換）
- [ ] `_transition_offline_to_online()` を実装（Phase A: ベース画像のみ引き継ぎ）
- [ ] Phase A 動作確認後、Phase B（合成スナップショット）に切り替え
- [ ] プロジェクト保存: `OfflineLayerTransform` + サブ画像パスを `.pixy` に保存

---

**Last Updated**: 2026-08-15  
**Version**: 1.1 (Draft)  
**Author**: Design Phase Analysis


