# API仕様書

## 📋 目次

1. [概要](#概要)
2. [ベースURL](#ベースurl)
3. [認証](#認証)
4. [エンドポイント一覧](#エンドポイント一覧)
5. [データ型定義](#データ型定義)
6. [エラーハンドリング](#エラーハンドリング)
7. [使用例](#使用例)

## 概要

リアルタイム音声文字起こしシステムのREST API仕様。FastAPIベースで構築され、音声認識と話者分離機能を提供します。

### API特徴
- **RESTful設計**: 標準的なHTTPメソッドとステータスコード
- **非同期処理**: asyncio基盤による高速レスポンス
- **リアルタイム**: ポーリングベースのライブ結果取得
- **型安全**: Pydanticモデルによる入力検証

## ベースURL

```
http://localhost:8000
```

## 認証

現在のバージョンでは認証は不要です（ローカル環境想定）。

## エンドポイント一覧

### 1. システム制御

#### 🎤 音声文字起こし開始
```http
POST /start
```

音声キャプチャと文字起こし処理を開始します。

**リクエスト**
- **Headers**: `Content-Type: application/json`
- **Body**: なし

**レスポンス**
```json
{
  "status": "started"
}
```

**ステータスコード**
- `200 OK`: 正常に開始
- `500 Internal Server Error`: システムエラー

---

#### ⏹ 音声文字起こし停止
```http
POST /stop
```

実行中の音声処理を停止します。

**リクエスト**
- **Headers**: `Content-Type: application/json`
- **Body**: なし

**レスポンス**
```json
{
  "status": "stopped"
}
```

**ステータスコード**
- `200 OK`: 正常に停止
- `500 Internal Server Error`: システムエラー

---

### 2. データ取得

#### 📄 文字起こし結果取得
```http
GET /results
```

蓄積された文字起こし結果を取得します。ポーリング用エンドポイント。

**リクエスト**
- **Headers**: なし
- **Parameters**: なし

**レスポンス**
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
            "segment_count": 1,
            "words": ["こんにちは", "今日は", "いい天気ですね"]
          }
        }
      },
      "diarization": {
        "speakers": ["Speaker_A"],
        "segments": [
          {
            "start": 0.0,
            "end": 3.2,
            "duration": 3.2,
            "speaker": "Speaker_A"
          }
        ],
        "num_speakers": 1
      }
    }
  ],
  "count": 1,
  "is_running": true
}
```

**ステータスコード**
- `200 OK`: 正常取得
- `500 Internal Server Error`: システムエラー

---

#### 🗑 文字起こし結果クリア
```http
DELETE /results
```

蓄積された文字起こし結果をクリアします。

**リクエスト**
- **Headers**: なし
- **Body**: なし

**レスポンス**
```json
{
  "status": "cleared"
}
```

**ステータスコード**
- `200 OK`: 正常にクリア
- `500 Internal Server Error`: システムエラー

---

### 3. システム情報

#### ❤️ ヘルスチェック
```http
GET /health
```

システムの稼働状態を確認します。

**リクエスト**
- **Headers**: なし
- **Parameters**: なし

**レスポンス**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "is_transcribing": false,
  "audio_devices": 2
}
```

**ステータスコード**
- `200 OK`: システム正常
- `503 Service Unavailable`: システム異常

---

#### 📊 詳細ステータス
```http
GET /status
```

システムの詳細な状態情報を取得します。

**リクエスト**
- **Headers**: なし
- **Parameters**: なし

**レスポンス**
```json
{
  "transcription": {
    "is_running": false,
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
  "results_count": 0
}
```

**ステータスコード**
- `200 OK`: 正常取得
- `500 Internal Server Error`: システムエラー

---

#### 🏠 ルート
```http
GET /
```

API情報を返します。

**レスポンス**
```json
{
  "message": "MLX Whisper Real-time Transcription API"
}
```

## データ型定義

### TranscriptionResult
```typescript
interface TranscriptionResult {
  text: string;                    // 全体の転写テキスト
  language: string;                // 検出言語 (e.g., "ja")
  segments: TranscriptionSegment[]; // セグメント配列
  speaker_summary?: SpeakerSummary; // 話者統計情報
  diarization?: DiarizationResult;  // 話者分離結果
}
```

### TranscriptionSegment
```typescript
interface TranscriptionSegment {
  start: number;    // 開始時刻 (秒)
  end: number;      // 終了時刻 (秒)
  text: string;     // セグメントテキスト
  speaker: string;  // 話者ID (e.g., "Speaker_A")
}
```

### SpeakerSummary
```typescript
interface SpeakerSummary {
  total_speakers: number;
  speaker_stats: {
    [speakerId: string]: {
      total_duration: number;    // 総発話時間 (秒)
      segment_count: number;     // セグメント数
      words: string[];          // 発話単語リスト
    }
  }
}
```

### DiarizationResult
```typescript
interface DiarizationResult {
  speakers: string[];           // 検出話者ID配列
  segments: DiarizationSegment[]; // 話者分離セグメント
  num_speakers: number;         // 話者数
}
```

### DiarizationSegment
```typescript
interface DiarizationSegment {
  start: number;     // 開始時刻 (秒)
  end: number;       // 終了時刻 (秒)
  duration: number;  // 持続時間 (秒)
  speaker: string;   // 話者ID
}
```

### SystemStatus
```typescript
interface SystemStatus {
  transcription: {
    is_running: boolean;
    model_name: string;
    language: string;
    diarization_enabled: boolean;
    diarization_loaded: boolean;
  };
  audio: {
    sample_rate: number;
    chunk_size: number;
    overlap_size: number;
    devices_available: number;
  };
  results_count: number;
}
```

### HealthStatus
```typescript
interface HealthStatus {
  status: "healthy" | "unhealthy";
  model_loaded: boolean;
  is_transcribing: boolean;
  audio_devices: number;
}
```

## エラーハンドリング

### エラーレスポンス形式
```json
{
  "detail": "Error description",
  "error_code": "ERROR_CODE",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### 一般的なエラー

#### 400 Bad Request
```json
{
  "detail": "Invalid request format",
  "error_code": "INVALID_REQUEST"
}
```

#### 500 Internal Server Error
```json
{
  "detail": "Transcription service initialization failed",
  "error_code": "SERVICE_ERROR"
}
```

#### 503 Service Unavailable
```json
{
  "detail": "Audio device not available",
  "error_code": "AUDIO_DEVICE_ERROR"
}
```

### エラーコード一覧

| エラーコード | 説明 | 対処法 |
|-------------|------|-------|
| `SERVICE_ERROR` | サービス初期化失敗 | システム再起動 |
| `AUDIO_DEVICE_ERROR` | 音声デバイスエラー | マイク接続確認 |
| `MODEL_LOAD_ERROR` | モデル読み込み失敗 | モデルファイル確認 |
| `TRANSCRIPTION_ERROR` | 文字起こしエラー | 音声品質確認 |
| `DIARIZATION_ERROR` | 話者分離エラー | 設定確認 |
| `INVALID_REQUEST` | 不正なリクエスト | リクエスト形式確認 |

## 使用例

### 基本的な使用フロー

#### 1. システム状態確認
```bash
curl -X GET http://localhost:8000/health
```

#### 2. 文字起こし開始
```bash
curl -X POST http://localhost:8000/start \
  -H "Content-Type: application/json"
```

#### 3. 結果ポーリング (1秒間隔推奨)
```bash
curl -X GET http://localhost:8000/results
```

#### 4. 文字起こし停止
```bash
curl -X POST http://localhost:8000/stop \
  -H "Content-Type: application/json"
```

#### 5. 結果クリア
```bash
curl -X DELETE http://localhost:8000/results
```

### JavaScript/TypeScript での使用例

#### フロントエンド統合例
```typescript
class TranscriptionClient {
  private baseUrl = 'http://localhost:8000';
  private pollingInterval: NodeJS.Timeout | null = null;

  async start(): Promise<void> {
    const response = await fetch(`${this.baseUrl}/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    });
    
    if (!response.ok) {
      throw new Error(`Failed to start: ${response.status}`);
    }
    
    // ポーリング開始
    this.startPolling();
  }

  async stop(): Promise<void> {
    if (this.pollingInterval) {
      clearInterval(this.pollingInterval);
      this.pollingInterval = null;
    }
    
    const response = await fetch(`${this.baseUrl}/stop`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    });
    
    if (!response.ok) {
      throw new Error(`Failed to stop: ${response.status}`);
    }
  }

  private startPolling(): void {
    this.pollingInterval = setInterval(async () => {
      try {
        const response = await fetch(`${this.baseUrl}/results`);
        const data = await response.json();
        
        if (data.results.length > 0) {
          this.onResults(data.results);
          // 処理済み結果をクリア
          await fetch(`${this.baseUrl}/results`, { method: 'DELETE' });
        }
      } catch (error) {
        console.error('Polling error:', error);
      }
    }, 1000);
  }

  private onResults(results: TranscriptionResult[]): void {
    results.forEach(result => {
      console.log('Transcription:', result.text);
      console.log('Speakers:', result.speaker_summary?.total_speakers);
    });
  }
}
```

#### React Hook での使用例
```typescript
function useTranscription() {
  const [isRecording, setIsRecording] = useState(false);
  const [results, setResults] = useState<TranscriptionResult[]>([]);
  const [error, setError] = useState<string | null>(null);

  const startRecording = async () => {
    try {
      await fetch('/api/start-transcription', { method: 'POST' });
      setIsRecording(true);
      setError(null);
    } catch (err) {
      setError('Failed to start recording');
    }
  };

  const stopRecording = async () => {
    try {
      await fetch('/api/stop-transcription', { method: 'POST' });
      setIsRecording(false);
    } catch (err) {
      setError('Failed to stop recording');
    }
  };

  // 結果ポーリング
  useEffect(() => {
    if (!isRecording) return;

    const interval = setInterval(async () => {
      try {
        const response = await fetch('/api/transcription-results');
        const data = await response.json();
        
        if (data.results.length > 0) {
          setResults(prev => [...prev, ...data.results]);
          await fetch('/api/transcription-results', { method: 'DELETE' });
        }
      } catch (err) {
        setError('Failed to fetch results');
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [isRecording]);

  return {
    isRecording,
    results,
    error,
    startRecording,
    stopRecording
  };
}
```

### Python クライアント例
```python
import asyncio
import aiohttp
from typing import List, Dict, Any

class TranscriptionClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
        self.polling_task = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def start(self) -> Dict[str, Any]:
        """文字起こし開始"""
        async with self.session.post(f"{self.base_url}/start") as response:
            return await response.json()

    async def stop(self) -> Dict[str, Any]:
        """文字起こし停止"""
        async with self.session.post(f"{self.base_url}/stop") as response:
            return await response.json()

    async def get_results(self) -> Dict[str, Any]:
        """結果取得"""
        async with self.session.get(f"{self.base_url}/results") as response:
            return await response.json()

    async def clear_results(self) -> Dict[str, Any]:
        """結果クリア"""
        async with self.session.delete(f"{self.base_url}/results") as response:
            return await response.json()

    async def get_status(self) -> Dict[str, Any]:
        """ステータス取得"""
        async with self.session.get(f"{self.base_url}/status") as response:
            return await response.json()

# 使用例
async def main():
    async with TranscriptionClient() as client:
        # 開始
        start_result = await client.start()
        print(f"Started: {start_result}")
        
        # 10秒間ポーリング
        for _ in range(10):
            await asyncio.sleep(1)
            results = await client.get_results()
            if results['results']:
                print(f"New results: {len(results['results'])}")
                await client.clear_results()
        
        # 停止
        stop_result = await client.stop()
        print(f"Stopped: {stop_result}")

if __name__ == "__main__":
    asyncio.run(main())
```

## パフォーマンス考慮事項

### ポーリング頻度
- **推奨**: 1秒間隔
- **最小**: 500ms間隔
- **注意**: 100ms以下は非推奨（サーバー負荷）

### レスポンス時間目標
- `/start`, `/stop`: <100ms
- `/results`: <50ms
- `/status`, `/health`: <10ms
- 文字起こし処理: <300ms/chunk

### 同時接続
- 現在は単一クライアント想定
- 複数クライアント対応は将来的な拡張予定

---

この API 仕様書は v1.0 時点の情報です。最新の変更については GitHub リポジトリを確認してください。