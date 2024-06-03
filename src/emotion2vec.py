"""
Speech-to-text transcriptiong service based on OpenAI Whisper.
"""

import tempfile
import time

from modal import Image, method, enter, Mount, Volume
from pathlib import Path

from .common import stub
from .transcriber import load_audio

from .constants import IDX_TO_EMOTION

e2v_data_path = Path(__file__).with_name("e2v_data").resolve()

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
    mounts=[Mount.from_local_dir(e2v_data_path, remote_path="/e2v_data")]
)
class Emotion2Vec:

    def __init__(self, person) -> None:
        self.person = person

    @enter()
    def load_model(self):
        from funasr import AutoModel
        import numpy as np

        self.emotion_vectors = np.load(f'/e2v_data/{self.person}_e2v.npy')
        self.emotion_model = AutoModel(model="iic/emotion2vec_base_finetuned")
    
    @method()
    def get_emotion_rag(
        self,
        audio_data: bytes,
    ):
        import numpy as np
        
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

            rag_distances = np.sum((self.emotion_vectors[max_score_index] \
                - emotion_dict['feats'][None, :])**2, axis=-1)
            print(rag_distances)
            best_sample_idx = np.argmin(rag_distances) + 1
            best_file = f'/audio/{self.person}_{IDX_TO_EMOTION[max_score_index]}_{best_sample_idx}.wav'

        print(f"E2V in {time.time() - t0:.2f}s")

        return best_file
    
audio_path = Path(__file__).with_name("audio").resolve()
vol = Volume.from_name('my-test-volume')

# modal run src.emotion2vec::generate_emotion_vectors
@stub.function(
    image=e2v_image,
    mounts=[Mount.from_local_dir(audio_path, remote_path="/audio")],
    volumes={'/tmp_data': vol},
    gpu='A10G',
)
def generate_emotion_vectors():
    from scipy.io import wavfile
    import numpy as np
    from funasr import AutoModel

    e2v = AutoModel(model="iic/emotion2vec_base_finetuned")
    emotions = ['angry', 'disgusted', 'fearful', 'happy', 'neutral',
                'other', 'sad', 'surprised', 'unknown']
    person = 'myra'
    all_vecs = np.zeros((9, 5, 768), dtype=np.float32)
    for i, emot in enumerate(emotions):
        print(i, emot)
        for j in range(1, 6):
            sr, audio_np = wavfile.read(f'/audio/{person}_{emot}_{j}.wav')
            rec_result = e2v.generate(audio_np.astype(np.float32), output_dir="./outputs", granularity="utterance", extract_embedding=True)
            all_vecs[i, j-1] = rec_result[0]['feats']
    np.save(f'/tmp_data/myra_e2v.npy', all_vecs)
    vol.commit()

