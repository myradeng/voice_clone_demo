"""
Speech-to-text transcriptiong service based on OpenAI Whisper.
"""

import tempfile
import time

from modal import Image, method, enter
from .common import stub
from .transcriber import load_audio

def download_model():
    from funasr import AutoModel

    AutoModel(model="iic/emotion2vec_base_finetuned")

e2v_image = (
    Image.debian_slim(python_version="3.10.8")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "ffmpeg-python",
        "funasr", 
        "modelscope",
        "torch==2.0.1",
        "torchaudio"
    )
    .run_function(download_model)
)

@stub.cls(
    gpu="A10G",
    container_idle_timeout=180,
    image=e2v_image,
)
class Emotion2Vec:

    @enter()
    def load_model(self):
        from funasr import AutoModel

        self.emotion_model = AutoModel(model="iic/emotion2vec_base_finetuned")
    
    @method()
    def get_emotion_rag(
        self,
        audio_data: bytes,
    ):
        t0 = time.time()
        np_array = load_audio(audio_data)
        rec_result = self.emotion_model.generate(np_array, output_dir="./outputs", granularity="utterance", extract_embedding=True)
        top_emotion = None
        emotion_dict = rec_result[0]
        if emotion_dict['scores']:
            # Find the index of the maximum score
            max_score_index = emotion_dict['scores'].index(max(emotion_dict['scores']))
            
            # Get the corresponding emotion label using the index
            top_emotion = emotion_dict['labels'][max_score_index]
            top_emotion = top_emotion.split('/')[-1]
            
            print("Top scored emotion:", top_emotion)
        print(f"E2V in {time.time() - t0:.2f}s")

        return top_emotion