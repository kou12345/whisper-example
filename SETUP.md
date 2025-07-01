# Whisper リアルタイム文字起こしアプリ セットアップガイド

## システム要件
- Python 3.12+
- Node.js 18+
- uv (Python package manager)
- M4 MacBook Pro (16GB memory)

## セットアップ手順

### 1. 依存関係のインストール

#### Pythonバックエンド
```bash
# プロジェクトルートで実行
uv sync
```

#### Next.jsフロントエンド
```bash
cd frontend
npm install
```

### 2. アプリケーションの起動

#### バックエンドサーバー（ポート8000）
```bash
# プロジェクトルートで実行
uv run python backend/main.py
```

#### フロントエンドサーバー（ポート3000）
```bash
cd frontend
npm run dev
```

### 3. アクセス方法
ブラウザで `http://localhost:3000` にアクセス

## 機能説明

### 音声録音
- マイクボタンをクリックして録音開始/停止
- 2秒間隔で自動的に文字起こし実行
- 音声レベルメーターでマイク入力確認

### 文字起こし結果
- リアルタイムで結果表示
- 話者分離機能（話者名表示）
- タイムスタンプ表示

### 保存・エクスポート機能
- TXT、CSV、JSON形式でエクスポート
- ローカルストレージへの保存
- クリア機能

## 使用技術
- **フロントエンド**: Next.js 14, TypeScript, Tailwind CSS
- **バックエンド**: Python, FastAPI, OpenAI Whisper
- **音声処理**: Web Audio API, pyannote.audio（話者分離）
- **通信**: REST API

## トラブルシューティング

### マイクアクセスが拒否される場合
ブラウザの設定でマイクアクセスを許可してください。

### 話者分離が動作しない場合
初回実行時にpyannoteモデルのダウンロードが必要です。ネットワーク接続を確認してください。

### バックエンドエラーが発生する場合
Pythonの依存関係が正しくインストールされているか確認してください：
```bash
uv sync --all-packages
```