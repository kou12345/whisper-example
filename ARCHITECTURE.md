# システムアーキテクチャドキュメント

## 📋 目次

1. [システム概要](#システム概要)
2. [アーキテクチャ設計](#アーキテクチャ設計)
3. [コンポーネント詳細](#コンポーネント詳細)
4. [データフロー](#データフロー)
5. [API設計](#api設計)
6. [パフォーマンス考慮事項](#パフォーマンス考慮事項)
7. [セキュリティ](#セキュリティ)

## システム概要

MLX Whisperを核とした、リアルタイム音声文字起こし・話者分離システム。Apple Silicon最適化により高速処理を実現し、Webベースの直感的なUIを提供する。

### 主要コンポーネント

```
┌─────────────────────────────────────────────────────────────────┐
│                        Web Application                         │
├─────────────────────────────────────────────────────────────────┤
│  Frontend (Next.js)                                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ AudioRecorder   │  │TranscriptionView│  │ ExportManager   │ │
│  │ Component       │  │ Component       │  │ Component       │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  API Layer (Next.js API Routes)                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ /api/start      │  │ /api/results    │  │ /api/stop       │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │ HTTP/REST API
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Backend Service                           │
├─────────────────────────────────────────────────────────────────┤
│  FastAPI Application (src/main.py)                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ TranscriptionSvc│  │ SessionManager  │  │ HealthMonitor   │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  Audio Processing Pipeline                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ AudioCapture    │  │ MLXTranscriber  │  │ SpeakerDiarizer │ │
│  │ (audio.py)      │  │ (transcribe.py) │  │ (diarization.py)│ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  Core Libraries                                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ sounddevice     │  │ MLX Whisper     │  │ pyannote.audio  │ │
│  │ webrtcvad       │  │ Apple Silicon   │  │ scikit-learn    │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## アーキテクチャ設計

### 1. レイヤー構成

#### Presentation Layer (Frontend)
- **フレームワーク**: Next.js 14 + TypeScript
- **状態管理**: React Hooks (useState, useEffect, useCallback)
- **スタイリング**: Tailwind CSS
- **責務**: UI/UX, ユーザーインタラクション, データ表示

#### API Gateway Layer (Next.js API Routes)
- **役割**: Frontend-Backend間の通信仲介
- **プロトコル**: HTTP/REST
- **認証**: なし（ローカル環境前提）
- **エラーハンドリング**: HTTP status codes

#### Business Logic Layer (FastAPI Backend)
- **フレームワーク**: FastAPI + asyncio
- **アーキテクチャパターン**: Service-oriented
- **並行処理**: async/await pattern
- **責務**: ビジネスロジック、音声処理オーケストレーション

#### Data Processing Layer (Audio Pipeline)
- **音声キャプチャ**: sounddevice (PortAudio wrapper)
- **音声認識**: MLX Whisper (Apple Silicon最適化)
- **話者分離**: pyannote.audio + fallback implementation
- **責務**: リアルタイム音声処理、ML推論

### 2. 設計原則

#### Single Responsibility Principle
各コンポーネントは単一の責務を持つ：
- `AudioCapture`: 音声キャプチャとVAD
- `MLXTranscriber`: 音声認識
- `SpeakerDiarization`: 話者分離
- `TranscriptionService`: オーケストレーション

#### Dependency Injection
設定ベースの依存性注入:
```python
# config.py での設定
ENABLE_DIARIZATION = os.getenv("ENABLE_DIARIZATION", "true").lower() == "true"

# transcribe.py での利用
self.diarization = SpeakerDiarization() if ENABLE_DIARIZATION else None
```

#### Error Handling Strategy
段階的フォールバック:
1. pyannote.audio (最高精度)
2. 独自実装 (フォールバック)
3. 単一話者仮定 (最後の手段)

## コンポーネント詳細

### Frontend Components

#### AudioRecorder Component
```typescript
interface AudioRecorderProps {
  onTranscriptionResult: (result: TranscriptionResult) => void;
  isRecording: boolean;
  setIsRecording: (recording: boolean) => void;
}
```

**責務**:
- 録音開始/停止制御
- バックエンドAPI呼び出し
- 結果ポーリング (1秒間隔)
- UI状態管理

**状態管理**:
- `status`: 'ready' | 'starting' | 'recording' | 'stopping' | 'error'
- `pollingInterval`: ポーリング用タイマー

#### TranscriptionDisplay Component
```typescript
interface TranscriptionDisplayProps {
  transcriptions: TranscriptionResult[];
}
```

**責務**:
- 文字起こし結果表示
- 話者別色分け表示
- エクスポート機能 (TXT/CSV/JSON)
- ローカルストレージ連携

### Backend Services

#### TranscriptionService Class
```python
class TranscriptionService:
    def __init__(self):
        self.audio_capture = AudioCapture()
        self.transcriber = MLXTranscriber()
        self.is_running = False
        self.transcription_task = None
```

**責務**:
- 音声処理パイプラインの統合
- 非同期タスク管理
- リソース管理 (初期化/クリーンアップ)

**主要メソッド**:
- `initialize()`: システム初期化
- `start_transcription()`: 音声処理開始
- `stop_transcription()`: 処理停止
- `cleanup()`: リソース解放

#### AudioCapture Class
```python
class AudioCapture:
    def __init__(self):
        self.sample_rate = SAMPLE_RATE
        self.chunk_size = get_chunk_size()
        self.overlap_size = get_overlap_size()
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
```

**音声処理フロー**:
1. `sounddevice.RawInputStream`でリアルタイム音声キャプチャ
2. `deque`バッファーで音声データ管理
3. WebRTC VADで音声活動検出
4. 4秒チャンク + 0.5秒オーバーラップ処理

**VAD (Voice Activity Detection)**:
- 10msフレーム単位での音声判定
- 10%以上のフレームで音声検出時に処理実行
- 音声レベル閾値 (0.001) による追加フィルタリング

#### MLXTranscriber Class
```python
class MLXTranscriber:
    def __init__(self):
        self.model_name = MODEL_NAME
        self.language = LANGUAGE
        self.diarization = SpeakerDiarization() if ENABLE_DIARIZATION else None
```

**処理フロー**:
1. 音声データ正規化 (`[-1, 1]`範囲)
2. MLX Whisperによる音声認識
3. 話者分離処理 (有効時)
4. 不要フレーズフィルタリング
5. 結果統合・構造化

**フィルタリング対象**:
- "ご視聴ありがとうございました"
- "チャンネル登録お願いします"
- その他一般的なハルシネーション

#### SpeakerDiarization Class
```python
class SpeakerDiarization:
    def __init__(self):
        self.pipeline = None
        self.is_loaded = False
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
```

**アルゴリズム**:
1. **メイン手法**: pyannote.audio
   - 事前学習済みモデル使用
   - HuggingFace認証対応
   - 高精度話者識別

2. **フォールバック手法**: 独自実装
   - 音声特徴量抽出 (ピッチ、MFCC、スペクトル特徴)
   - K-meansクラスタリング
   - エルボー法による最適クラスタ数決定

**特徴量**:
- 基本周波数 (80-400Hz範囲)
- スペクトル重心・ロールオフ・帯域幅
- MFCC様特徴 (13バンド)
- ゼロ交差率
- RMSエネルギー
- スペクトラルコントラスト

## データフロー

### 1. 音声キャプチャフロー
```
[マイク] → [sounddevice] → [リングバッファ] → [VAD] → [チャンク生成]
   ↓
[オーバーラップ処理] → [音声レベル確認] → [MLX Whisper]
```

### 2. 文字起こしフロー
```
[音声チャンク] → [MLX Whisper] → [テキスト生成] → [話者分離]
     ↓
[セグメント統合] → [フィルタリング] → [結果構造化] → [フロントエンド]
```

### 3. 話者分離フロー
```
[音声データ] → [pyannote.audio] → [話者セグメント]
     ↓ (失敗時)
[特徴量抽出] → [K-meansクラスタリング] → [話者ラベル]
     ↓
[転写結果とのマッピング] → [話者付きセグメント]
```

### 4. API通信フロー
```
[Frontend] --POST /start--> [API Route] --POST /start--> [Backend]
     ↓
[Polling] --GET /results--> [API Route] --GET /results--> [Backend]
     ↓
[Results] <--JSON-- [API Route] <--JSON-- [Backend]
```

## API設計

### RESTful Endpoints

#### POST /start
**リクエスト**: 
```json
{} // Body不要
```

**レスポンス**:
```json
{
  "status": "started" | "already_running"
}
```

#### GET /results
**レスポンス**:
```json
{
  "results": [
    {
      "text": "転写テキスト",
      "language": "ja",
      "segments": [
        {
          "start": 0.0,
          "end": 3.2,
          "text": "セグメントテキスト",
          "speaker": "Speaker_A"
        }
      ],
      "speaker_summary": {
        "total_speakers": 2,
        "speaker_stats": {
          "Speaker_A": {
            "total_duration": 5.5,
            "segment_count": 3,
            "words": ["こんにちは", "今日は"]
          }
        }
      },
      "diarization": {
        "speakers": ["Speaker_A", "Speaker_B"],
        "segments": [...],
        "num_speakers": 2
      }
    }
  ],
  "count": 1,
  "is_running": true
}
```

#### GET /status
**レスポンス**:
```json
{
  "transcription": {
    "is_running": true,
    "model_name": "mlx-community/whisper-large-v3-turbo",
    "language": "ja",
    "diarization_enabled": true,
    "diarization_loaded": true
  },
  "audio": {
    "sample_rate": 16000,
    "chunk_size": 64000,
    "overlap_size": 8000,
    "devices_available": 2
  },
  "results_count": 5
}
```

### Error Handling

#### HTTP Status Codes
- `200 OK`: 正常処理
- `400 Bad Request`: 不正なリクエスト
- `500 Internal Server Error`: サーバーエラー
- `503 Service Unavailable`: サービス利用不可

#### Error Response Format
```json
{
  "error": "error_code",
  "message": "Human readable error message",
  "details": {
    "timestamp": "2024-01-01T00:00:00Z",
    "request_id": "req_12345"
  }
}
```

## パフォーマンス考慮事項

### 1. メモリ管理

#### 音声バッファ管理
```python
# dequeを使用した効率的なリングバッファ
self.audio_buffer = deque(maxlen=self.chunk_size + self.overlap_size)

# 処理済みデータの定期削除
if len(transcription_results) > 50:
    transcription_results.pop(0)
```

#### モデルメモリ使用量
- MLX Whisper: ~1-2GB (モデルサイズ依存)
- pyannote.audio: ~500MB-1GB
- フォールバック実装: ~50MB

### 2. CPU最適化

#### 非同期処理
```python
# asyncio.run_in_executor での重い処理の分離
result = await loop.run_in_executor(
    None,
    self._transcribe_sync,
    audio_data
)
```

#### Apple Silicon最適化
- MLX: Metal Performance Shadersによる高速化
- MPS (Metal Performance Shaders) device利用
- メモリ共有による効率化

### 3. レイテンシ最小化

#### チューニングパラメータ
```python
CHUNK_MS = 4000          # レイテンシ vs 精度
OVERLAP_MS = 500         # コンテキスト保持
VAD_AGGRESSIVENESS = 0   # 感度 vs ノイズ除去
BLOCKSIZE = 64000        # 音声キャプチャバッファ
```

#### パフォーマンス目標
- 音声→テキスト: <300ms
- 話者分離: <200ms (フォールバック時)
- API応答: <50ms
- フロントエンド更新: <100ms

## セキュリティ

### 1. データプライバシー

#### ローカル処理
- 音声データは全てローカル処理
- 外部APIへのデータ送信なし
- ファイルシステムへの永続化なし

#### 一時データ管理
```python
# 一時ファイルの自動削除
with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
    try:
        # 処理
    finally:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
```

### 2. リソース保護

#### メモリリーク防止
```python
def unload_model(self):
    """Cleanup resources"""
    if self.diarization:
        self.diarization.unload_model()
    self.is_loaded = False
```

#### プロセス分離
- フロントエンド: Node.js プロセス
- バックエンド: Python プロセス
- 相互独立性による障害分離

### 3. 入力検証

#### 音声データ検証
```python
# データ型・レンジ検証
if audio_data.dtype != np.float32:
    audio_data = audio_data.astype(np.float32)

# 正規化
if np.max(np.abs(audio_data)) > 0:
    audio_data = audio_data / np.max(np.abs(audio_data))
```

#### API入力検証
- FastAPIのPydanticモデルによる自動検証
- 型安全性の保証
- 範囲外値の拒否

---

このアーキテクチャドキュメントは、システムの理解と保守性向上を目的として作成されました。実装の詳細や最新の変更については、ソースコードを参照してください。