"""
Speech-to-text transcriptiong service based on OpenAI Whisper.
"""

import tempfile
import time

from modal import Image, method, enter
from .common import stub

MODEL_NAME = "base.en"


def download_model():
    import whisper

    whisper.load_model(MODEL_NAME)

transcriber_image = (
    Image.debian_slim(python_version="3.10.8")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "https://github.com/openai/whisper/archive/v20230314.tar.gz",
        "ffmpeg-python",
        "funasr", 
        "modelscope",
        "torch",
        "torchaudio"
    )
    .run_function(download_model)
)


def load_audio(data: bytes, sr: int = 16000):
    import ffmpeg
    import numpy as np

    try:
        fp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        fp.write(data)
        
        fp.close()
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = (
            ffmpeg.input(
                fp.name,
                threads=0,
                format="f32le",
                acodec="pcm_f32le",
                ac=1,
                ar="48k",
            )
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(
                cmd=["ffmpeg", "-nostdin"],
                capture_stdout=True,
                capture_stderr=True,
            )
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.float32).flatten()

@stub.cls(
    gpu="A10G",
    container_idle_timeout=180,
    image=transcriber_image,
)
class Whisper:
    @enter()
    def load_model(self):
        import torch
        import whisper
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        self.use_gpu = torch.cuda.is_available()
        device = "cuda" if self.use_gpu else "cpu"
        self.model = whisper.load_model(MODEL_NAME, device=device)
        self.emotion2vec_pipeline = pipeline(task=Tasks.emotion_recognition, model="iic/emotion2vec_plus_base")
    
    @method()
    def transcribe_segment(
        self,
        audio_data: bytes,
    ):
        t0 = time.time()
        np_array = load_audio(audio_data)
        rec_result = self.emotion2vec_pipeline(np_array, output_dir="./outputs", granularity="utterance", extract_embedding=True)
        top_emotion = None
        emotion_dict = rec_result[0]
        if emotion_dict['scores']:
            # Find the index of the maximum score
            max_score_index = emotion_dict['scores'].index(max(emotion_dict['scores']))
            
            # Get the corresponding emotion label using the index
            top_emotion = emotion_dict['labels'][max_score_index]
            
            print("Top scored emotion:", top_emotion)
        result = self.model.transcribe(np_array, language="en", fp16=self.use_gpu)  # type: ignore
        print(f"Transcribed in {time.time() - t0:.2f}s")

        return {"text": result["text"], "top_emotion": top_emotion}
    

