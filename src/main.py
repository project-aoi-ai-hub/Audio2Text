import os
import sys
import time

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

from faster_whisper import WhisperModel

def transcribe_audio(audio_path, model_size="large-v3"):
    """
    音声ファイルを文字起こしする関数
    """
    if not os.path.exists(audio_path):
        print(f"エラー: 指定されたファイルが見つかりません: {audio_path}")
        return

    print(f"[{audio_path}] の文字起こしを開始します...")
    print(f"使用モデル: {model_size}")
    
    start_time = time.time()

    # device="auto" と compute_type="default" を指定することで、
    # 実行環境にNVIDIA GPU(CUDA)があればそれを使用し(float16などで動作)、
    # なければ自動的にCPU(int8などで動作)にフォールバックします。
#    model = WhisperModel(model_size, device="auto", compute_type="default")
#    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    model = WhisperModel(model_size, device="cuda", compute_type="default")

    # 文字起こしの実行
    segments, info = model.transcribe(audio_path, beam_size=5)

    print(f"検出された言語: {info.language} (確率: {info.language_probability:.2f})")
    print("-" * 30)

    # 結果をテキストファイルに書き出す準備
    base_name = os.path.splitext(audio_path)[0]
    output_path = f"{base_name}_transcription.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        for segment in segments:
            # タイムスタンプ付きで出力（[00:00.000 -> 00:05.000] テキスト...）
            line = f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}"
            print(line)
            f.write(line + "\n")

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("-" * 30)
    print(f"文字起こし完了。結果を保存しました: {output_path}")
    print(f"処理時間: {elapsed_time:.2f} 秒")


if __name__ == "__main__":
    # テスト用の音声ファイルのパスを指定してください
    # 例: target_file = "test_audio.wav"
#    target_file = "sample.mp3" 
    target_file = "..\\2026-03-14 13_24_39.ogg"
    
    # CPU環境でテストする場合は model_size="small" などに変更すると早く終わります
    transcribe_audio(target_file, model_size="large-v3")