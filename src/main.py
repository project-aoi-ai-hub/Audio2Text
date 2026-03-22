import os
import sys
import time
from dotenv import load_dotenv

load_dotenv() # HF_TOKENなどの読み込み

# Windows環境でNVIDIAの公式pipパッケージからDLLを読み込むための設定
if os.name == 'nt':
    for site_pkg in sys.path:
        cublas_bin = os.path.join(site_pkg, "nvidia", "cublas", "bin")
        cudnn_bin = os.path.join(site_pkg, "nvidia", "cudnn", "bin")
        if os.path.exists(cublas_bin) and os.path.exists(cudnn_bin):
            os.environ["PATH"] = f"{cublas_bin};{cudnn_bin};{os.environ.get('PATH', '')}"
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(cublas_bin)
                    os.add_dll_directory(cudnn_bin)
                except Exception:
                    pass
            break

# プロジェクト直下に配置された ffmpeg.exe のパスを通す
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in os.environ.get("PATH", ""):
    os.environ["PATH"] = project_root + os.pathsep + os.environ.get("PATH", "")

from faster_whisper import WhisperModel
import torch
import soundfile as sf
import numpy as np
import subprocess
import imageio_ffmpeg

# imageio_ffmpegパッケージ内の公式バイナリを取得（プロジェクト直下の破損 ffmpeg.exe を使わない）
ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
print(f"[INFO] ffmpeg: {ffmpeg_path}")

# ==========================================
# 1. グローバル設定（定数管理）の導入
# ==========================================
class Config:
    # --- モデル設定 ---
#    MODEL_NAME = "large-v3"
    MODEL_NAME = "large-v2"
#    MODEL_NAME = "turbo"
    
    # --- 文字起こし設定 ---
    BEAM_SIZE = 7  # 精度向上のために増やす
    LANGUAGE = "ja"
    
    # --- 抑制・フィルタ系 (ハルシネーション対策) ---
    CONDITION_ON_PREV = False
    VAD_FILTER = True
    COMPRESSION_THRESHOLD = 2.4
    NO_SPEECH_THRESHOLD = 0.6
    
    # --- 前処理・話者分離スイッチ ---
    ENABLE_NORMALIZATION = True
    ENABLE_DIARIZATION = True

# 初期プロンプトの読み込み関数
def get_initial_prompt():
    prompt_file = "initial_prompt.txt"
    if os.path.exists(prompt_file):
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""

Config.INITIAL_PROMPT = get_initial_prompt()

# ==========================================
# 2. 汎用的な音声前処理（pydub）
# ==========================================
def preprocess_audio(audio_path):
    """
    imageio_ffmpeg経由で取得した公式FFmpegて16kHz WAV変換し、
    soundfileでピーク正規化を行う堅牢な前処理関数。
    """
    print(f"[前処理] 音声ファイル '{audio_path}' の変換と正規化を開始します...")
    try:
        # パスは必ず絶対パスに変換して扱う
        abs_audio = os.path.abspath(audio_path)
        base_name = os.path.splitext(abs_audio)[0]
        temp_wav_path = f"{base_name}_temp_normalized.wav"
        
        # 1. imageio_ffmpegの公式FFmpegで安全に WAV（16kHzモノ）へ変換
        subprocess.run(
            [ffmpeg_path, "-y", "-i", abs_audio, "-ac", "1", "-ar", "16000", temp_wav_path],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        
        # 2. soundfile でピーク正規化
        data, sr = sf.read(temp_wav_path, dtype='float32')
        max_amp = np.abs(data).max()
        if max_amp > 0:
            data = data / max_amp
        sf.write(temp_wav_path, data, sr, format='WAV', subtype='PCM_16')
        
        print(f"[前処理] 変換・正規化完了: {temp_wav_path}")
        return temp_wav_path

    except Exception as e:
        print(f"[エラー] 音声前処理中にエラーが発生しました: {e}")
        print("[前処理] 正規化をスキップし、元の音声ファイルを使用します。")
        return audio_path

# ==========================================
# 4. 話者分離（pyannote-audio）の統合
# ==========================================
def perform_diarization(audio_path):
    """
    pyannote.audio を利用して話者分離を行う
    """
    print("[話者分離] pyannote.audio パイプラインを初期化中...")
    
    # HF_TOKEN が環境変数に設定されているか確認
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("[警告] HF_TOKEN が設定されていません。Hugging Faceのトークンを .env ファイルに記述してください。")
        print("[話者分離] 話者分離をスキップします。")
        return None
        
    try:
        from pyannote.audio import Pipeline
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token
        )
        
        # GPUが使える場合はGPUに送る
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
            
        print("[話者分離] 音声データをメモリにロード中（AudioDecoderエラー回避のため）...")
        
        # torchaudio(torchcodec) のバグを回避するため、soundfile で手動ロードを行う
        waveform_np, sample_rate = sf.read(audio_path, dtype='float32')
        # 次元を (時間, チャンネル) に整形
        if waveform_np.ndim == 1:
            waveform_np = waveform_np.reshape(-1, 1)
        # PyTorchの要求である (チャンネル, 時間) に転置してTensor化
        waveform_tensor = torch.from_numpy(waveform_np).T
        
        audio_in_memory = {"waveform": waveform_tensor, "sample_rate": sample_rate}
            
        print("[話者分離] 音声の解析を実行中...")
        diarization = pipeline(audio_in_memory)
        print("[話者分離] 解析が完了しました。")
        return diarization
    except Exception as e:
        print(f"[エラー] 話者分離の処理中にエラーが発生しました: {e}")
        return None

def assign_speaker(segment_start, segment_end, diarization_result):
    """
    Whisperの一つのセグメント時間帯に対して、最も長く重なっている話者を判定する
    """
    if diarization_result is None:
        return "Speaker_??"
        
    speaker_durations = {}
    
    try:
        # pyannote.core.Annotation 形式（最も一般的な形式）
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            overlap_start = max(segment_start, turn.start)
            overlap_end = min(segment_end, turn.end)
            overlap_duration = overlap_end - overlap_start
            if overlap_duration > 0:
                speaker_durations[speaker] = speaker_durations.get(speaker, 0) + overlap_duration
    except AttributeError:
        try:
            # DiarizeOutput 形式：(segment.start, segment.end, speaker)のタプルリストとして物理的にイテレート
            for item in diarization_result:
                if hasattr(item, 'start') and hasattr(item, 'end') and hasattr(item, 'speaker'):
                    overlap_start = max(segment_start, item.start)
                    overlap_end = min(segment_end, item.end)
                    overlap_duration = overlap_end - overlap_start
                    if overlap_duration > 0:
                        speaker_durations[item.speaker] = speaker_durations.get(item.speaker, 0) + overlap_duration
                elif isinstance(item, (list, tuple)) and len(item) >= 3:
                    start, end, speaker = item[0], item[1], item[-1]
                    overlap_start = max(segment_start, start)
                    overlap_end = min(segment_end, end)
                    overlap_duration = overlap_end - overlap_start
                    if overlap_duration > 0:
                        speaker_durations[speaker] = speaker_durations.get(speaker, 0) + overlap_duration
        except Exception as e:
            print(f"[話者割り当て] エラー発生 (Type: {type(diarization_result).__name__}): {e}")
            return "Speaker_??"
            
    if not speaker_durations:
        return "Speaker_??"
    
    # 最も重なりが長い話者を返す
    best_speaker = max(speaker_durations.items(), key=lambda x: x[1])[0]
    return best_speaker

# ==========================================
# メインの文字起こし処理
# ==========================================
def transcribe_audio(audio_path):
    """
    音声ファイルを文字起こしする関数
    """
    if not os.path.exists(audio_path):
        print(f"エラー: 指定されたファイルが見つかりません: {audio_path}")
        return

    print(f"[{audio_path}] の処理を開始します...")
    
    start_time = time.time()
    
    # 2. 前処理 (正規化) の実行
    target_audio_path = audio_path
    if Config.ENABLE_NORMALIZATION:
        target_audio_path = preprocess_audio(audio_path)
        
    # 話者分離の並行または事前実行
    diarization_result = None
    if Config.ENABLE_DIARIZATION:
        diarization_result = perform_diarization(target_audio_path)

    # Whisper モデルのロード
    print(f"[Whisper] モデル ({Config.MODEL_NAME}) をロード中...")
    device = "cuda" if torch.cuda.is_available() else "auto"
    model = WhisperModel(Config.MODEL_NAME, device=device, compute_type="default")
    print("[Whisper] モデルのロードが完了しました。")

    # 3. ロジックの条件分岐実装 (transcribeの引数に設定を渡す)
    print("[Whisper] 文字起こしを実行中...")
    segments, info = model.transcribe(
        target_audio_path,
        language=Config.LANGUAGE,
        beam_size=Config.BEAM_SIZE,
        initial_prompt=Config.INITIAL_PROMPT,
        condition_on_previous_text=Config.CONDITION_ON_PREV,
        compression_ratio_threshold=Config.COMPRESSION_THRESHOLD,
        no_speech_threshold=Config.NO_SPEECH_THRESHOLD,
        vad_filter=Config.VAD_FILTER
    )

    print(f"[Whisper] 検出言語: {info.language} (確率: {info.language_probability:.2f})")
    print("-" * 50)

    base_name = os.path.splitext(audio_path)[0]
    output_path = f"{base_name}_transcription.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        for segment in segments:
            # 話者分離結果との照合
            speaker_label = assign_speaker(segment.start, segment.end, diarization_result) if Config.ENABLE_DIARIZATION else "Speaker"
            
            # [開始時間][話者ラベル] テキスト 形式で出力
            line = f"[{segment.start:.2f}s -> {segment.end:.2f}s][{speaker_label}] {segment.text}"
            print(line)
            f.write(line + "\n")

    # 一時ファイルのクリーンアップ
    if Config.ENABLE_NORMALIZATION and target_audio_path != audio_path:
        if os.path.exists(target_audio_path):
            os.remove(target_audio_path)
            print(f"[前処理] 一時ファイル {target_audio_path} を削除しました。")

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("-" * 50)
    print(f"文字起こし完了: {output_path}")
    print(f"総処理時間: {elapsed_time:.2f} 秒")

if __name__ == "__main__":
    # テスト用の音声ファイルのパスを指定
    target_file = "..\\2026-03-14 13_24_39.ogg"
    transcribe_audio(target_file)