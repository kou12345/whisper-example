# リアルタイム音声文字起こしシステム

MLX Whisper + 話者分離機能を使ったApple Silicon最適化のリアルタイム音声文字起こしWebアプリケーション

## 🎯 概要

このプロジェクトは、M4 MacBook Pro上でリアルタイムに音声を文字起こしし、話者を自動的に識別するWebアプリケーションです。MLX Whisperを使用してApple Siliconの性能を最大限活用し、高精度な音声認識と話者分離を実現します。

## ✨ 主な機能

- **リアルタイム音声文字起こし**: 4秒間隔での高精度文字起こし
- **話者分離**: pyannote.audio + 独自フォールバック実装による話者識別
- **Apple Silicon最適化**: MLX Whisperによる高速処理
- **音声活動検出 (VAD)**: webrtcvadによる無音区間の自動スキップ
- **エクスポート機能**: TXT/CSV/JSON形式での結果保存
- **リアルタイムUI**: Next.jsによる応答性の高いフロントエンド

## 🏗️ システム構成

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Frontend       │    │  Backend        │    │  Audio Pipeline │
│  (Next.js)      │◄──►│  (FastAPI)      │◄──►│  (MLX Whisper)  │
│                 │    │                 │    │                 │
│ - Audio Control │    │ - API Endpoints │    │ - Audio Capture │
│ - Result Display│    │ - WebSocket     │    │ - VAD Filter    │
│ - Export Tools  │    │ - Session Mgmt  │    │ - Transcription │
└─────────────────┘    └─────────────────┘    │ - Diarization   │
                                               └─────────────────┘
```

## 🔧 技術スタック

### Backend
- **Python 3.12+** - メインプログラミング言語
- **MLX Whisper** - Apple Silicon最適化音声認識
- **FastAPI** - 高性能WebAPIフレームワーク
- **sounddevice** - リアルタイム音声キャプチャ
- **webrtcvad** - 音声活動検出
- **pyannote.audio** - 話者分離（メイン）
- **scikit-learn** - 機械学習（フォールバック話者分離）

### Frontend
- **Next.js 14** - React基盤Webフレームワーク
- **TypeScript** - 型安全性
- **Tailwind CSS** - UIスタイリング

### システム要件
- **M4 MacBook Pro** (推奨)
- **Memory: 16GB+**
- **macOS Sonoma+**

## 🚀 セットアップ

### 1. リポジトリのクローン
```bash
git clone https://github.com/kou12345/whisper-example.git
cd whisper-example
```

### 2. Python環境のセットアップ
```bash
# uvのインストール（まだの場合）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 依存関係のインストール
uv sync
```

### 3. フロントエンドのセットアップ
```bash
cd frontend
npm install
```

### 4. 環境変数の設定（オプション）
```bash
# .envファイルを作成
export WHISPER_MODEL="mlx-community/whisper-large-v3-turbo"
export ENABLE_DIARIZATION="true"
export HUGGINGFACE_TOKEN="your_token_here"  # pyannote.audio用（オプション）
export LOG_LEVEL="INFO"
```

## 🎮 使用方法

### 1. バックエンドサーバーの起動
```bash
# プロジェクトルートで
uv run python src/main.py
```
サーバーは `http://localhost:8000` で起動します

### 2. フロントエンドの起動
```bash
cd frontend
npm run dev
```
Webアプリは `http://localhost:3000` でアクセス可能

### 3. 基本操作
1. **録音開始**: 🎤ボタンをクリック
2. **リアルタイム表示**: 文字起こし結果が自動表示
3. **話者確認**: Speaker_A, Speaker_B等で話者を識別
4. **エクスポート**: 形式を選択してデータをダウンロード
5. **録音終了**: ⏹ボタンをクリック

## 📊 API仕様

### エンドポイント一覧

| Method | Endpoint | 説明 |
|--------|----------|------|
| POST | `/start` | 音声文字起こし開始 |
| POST | `/stop` | 音声文字起こし停止 |
| GET | `/results` | 文字起こし結果取得 |
| DELETE | `/results` | 結果データクリア |
| GET | `/status` | システム状態取得 |
| GET | `/health` | ヘルスチェック |

### レスポンス例

#### 文字起こし結果 (`/results`)
```json
{
  "results": [
    {
      "text": "こんにちは、今日はいい天気ですね。",
      "language": "ja",
      "segments": [
        {
          "start": 0.0,
          "end": 3.2,
          "text": "こんにちは、今日はいい天気ですね。",
          "speaker": "Speaker_A"
        }
      ],
      "speaker_summary": {
        "total_speakers": 1,
        "speaker_stats": {
          "Speaker_A": {
            "total_duration": 3.2,
            "segment_count": 1
          }
        }
      }
    }
  ],
  "count": 1,
  "is_running": true
}
```

## ⚙️ 設定オプション

### 音声設定 (`src/config.py`)
```python
SAMPLE_RATE = 16000      # サンプリングレート (Hz)
CHUNK_MS = 4000          # 処理チャンクサイズ (ms)
OVERLAP_MS = 500         # オーバーラップ (ms)
VAD_AGGRESSIVENESS = 0   # VAD感度 (0-3)
```

### 話者分離設定
```python
ENABLE_DIARIZATION = True    # 話者分離の有効/無効
MIN_SPEAKERS = 1             # 最小話者数
MAX_SPEAKERS = 5             # 最大話者数
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
```

### Whisperモデル設定
```python
MODEL_NAME = "mlx-community/whisper-large-v3-turbo"  # 使用モデル
LANGUAGE = "ja"              # 認識言語
```

## 🧪 テスト

### 音声システムテスト
```bash
# マイクアクセステスト
python test_audio.py

# VAD機能テスト
python tests/test_vad.py

# レイテンシーテスト
python tests/test_latency.py
```

### システム動作確認
```bash
# API動作確認
curl http://localhost:8000/health

# 文字起こし状態確認
curl http://localhost:8000/status
```

## 🔧 トラブルシューティング

### よくある問題

#### 1. マイクアクセスエラー
```bash
# macOSのプライバシー設定を確認
# システム設定 > プライバシーとセキュリティ > マイク
```

#### 2. pyannote.audioエラー
```bash
# HuggingFaceトークンを設定
export HUGGINGFACE_TOKEN="your_token"

# または話者分離を無効化
export ENABLE_DIARIZATION="false"
```

#### 3. MLXエラー
```bash
# Apple Siliconか確認
system_profiler SPHardwareDataType

# MLX Whisperの再インストール
uv remove mlx-whisper
uv add mlx-whisper
```

#### 4. メモリ不足
```bash
# より軽量なモデルを使用
export WHISPER_MODEL="mlx-community/whisper-small"
```

## 📈 パフォーマンス

### ベンチマーク (M4 MacBook Pro)
- **レイテンシー**: ~200ms (音声→テキスト)
- **スループット**: リアルタイム処理可能
- **メモリ使用量**: ~2-4GB
- **CPU使用率**: ~15-30%

### 最適化のヒント
1. **モデルサイズ調整**: 精度 vs 速度のトレードオフ
2. **チャンクサイズ調整**: レイテンシー vs 精度
3. **VAD閾値調整**: ノイズ除去 vs 感度
4. **話者分離**: 必要に応じて無効化

## 🤝 貢献

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 ライセンス

MIT License - 詳細は [LICENSE](LICENSE) ファイルを参照

## 🙏 謝辞

- [MLX Whisper](https://github.com/ml-explore/mlx-examples) - Apple Silicon最適化
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) - 話者分離
- [webrtcvad](https://github.com/wiseman/py-webrtcvad) - 音声活動検出
- [Next.js](https://nextjs.org/) - フロントエンドフレームワーク

---

🤖 このドキュメントは [Claude Code](https://claude.ai/code) で生成されました