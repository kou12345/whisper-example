# MLX Whisper リアルタイム文字起こしアプリ セットアップガイド

## 概要
- **MLX Whisper**: Apple Silicon GPU最適化で高速処理
- **sounddevice**: PortAudio経由で16kHz PCM音声取得
- **webrtcvad**: Google製VADで無音カット
- **2秒チャンク + 0.5秒オーバーラップ**: 文脈保持

## システム要件
- Apple Silicon Mac (M1/M2/M3/M4)
- macOS 12.0+
- Python 3.12+
- Node.js 18+
- uv (Python package manager)

## セットアップ手順

### 1. 依存関係のインストール

#### システム依存関係
```bash
# PortAudio (sounddevice用)
brew install uv portaudio

# プロジェクトの依存関係
uv sync
```

#### フロントエンド
```bash
cd frontend
npm install
```

### 2. アプリケーションの起動

#### MLXバックエンドサーバー（ポート8000）
```bash
# プロジェクトルートで実行
uv run python -m src.main
```

#### フロントエンドサーバー（ポート3000）
```bash
cd frontend
npm run dev
```

### 3. アクセス方法
ブラウザで `http://localhost:3000` にアクセス

## 新機能

### MLX Whisper の利点
- **Metal GPU最適化**: M4なら large-v3でも実時間0.8倍
- **メモリ効率**: 7GB VRAMで large-v3が動作
- **低遅延**: 2秒チャンクを1.6秒以内で処理

### VAD (Voice Activity Detection)
- **WebRTC VAD**: Google製の高精度音声検出
- **無音スキップ**: 無駄な処理を削減
- **30%閾値**: フレームの30%以上で音声検出時のみ処理

### オーバーラップ処理
- **0.5秒オーバーラップ**: 単語の切れ目を防ぐ
- **文脈保持**: 前のチャンクとの連続性確保

## API エンドポイント

### 制御
- `POST /start` - リアルタイム文字起こし開始
- `POST /stop` - 停止
- `GET /results` - 文字起こし結果取得
- `DELETE /results` - 結果クリア

### 監視
- `GET /health` - ヘルスチェック
- `GET /status` - 詳細ステータス

## 設定

### 環境変数
```bash
# Whisperモデル (tiny/base/small/medium/large-v3)
export WHISPER_MODEL=large-v3

# ログレベル
export LOG_LEVEL=INFO
```

### 設定ファイル (src/config.py)
```python
# 音声設定
SAMPLE_RATE = 16000        # 16kHz
CHUNK_MS = 2000           # 2秒チャンク
OVERLAP_MS = 500          # 0.5秒オーバーラップ

# VAD設定
VAD_AGGRESSIVENESS = 2    # 0-3 (高いほど積極的)

# MLX Whisper設定
MODEL_NAME = "large-v3"
LANGUAGE = "ja"
BEAM_SIZE = 5
```

## テスト実行

```bash
# VADテスト
uv run python -m pytest tests/test_vad.py -v

# レイテンシテスト
uv run python -m pytest tests/test_latency.py -v

# 全テスト
uv run python -m pytest tests/ -v
```

## パフォーマンス指標

### M4 MacBook Pro (16GB) での性能
- **large-v3モデル**: 実時間0.8倍処理
- **メモリ使用量**: 約7GB (GPU)
- **目標遅延**: ≤1.6秒 (2秒チャンク)

## トラブルシューティング

### 音声デバイスが認識されない
```bash
# 利用可能デバイス確認
python -c "import sounddevice as sd; print(sd.query_devices())"
```

### Metal/MLXエラー
```bash
# システム情報確認
python -c "import mlx.core as mx; print(mx.metal.device_info())"
```

### VADが反応しない
- マイク音量を上げる
- `VAD_AGGRESSIVENESS`を下げる (0-1)
- 環境ノイズを確認

### メモリ不足
- モデルを小さくする (medium → small)
- 他のアプリを終了
- Swap使用量を確認

## 性能比較

| モデル | VRAM | 処理時間 | 精度 |
|--------|------|----------|------|
| tiny | 1GB | 0.2x | 低 |
| base | 1GB | 0.3x | 中 |
| small | 2GB | 0.4x | 中+ |
| medium | 5GB | 0.6x | 高 |
| large-v3 | 7GB | 0.8x | 最高 |

*処理時間は実時間比 (M4 MacBook Pro)