# Audio2Text プロジェクト

音声データをテキストに変換するための Python プロジェクトです。

## プロジェクト構成

```text
Audio2Text/
├── src/                # ソースコード
│   ├── __init__.py
│   └── main.py         # メインのエントリーポイント
├── tests/              # テストコード
│   └── __init__.py
├── docs/               # ドキュメント
├── .gitignore          # Git除外設定
├── pyproject.toml      # プロジェクト設定
├── README.md           # このファイル
└── requirements.txt    # 必要なライブラリリスト
```

## セットアップ手順

1. 仮想環境の作成 (推奨)
   ```bash
   python -m venv .venv
   ```

2. 仮想環境の有効化
   - Windows: `.venv\Scripts\activate`
   - Mac/Linux: `source .venv/bin/activate`

3. 依存ライブラリのインストール
   ```bash
   pip install -r requirements.txt
   ```

## 詳しい使い方

`src/main.py` を実行して、動作を確認してください。
```bash
python src/main.py
```
