"""
Calling ElevenLabs TTS API
"""

import io
import tempfile
from modal import Image, method, build, enter, Mount
from modal.functions import FunctionCall
from fastapi.responses import Response
from pathlib import Path
from .common import stub
from .constants import AUDIO_RAG_TEXT, EMOTION_TO_IDX

audio_path = Path(__file__).with_name("audio").resolve()
voicecraft_name = "330M_TTSEnhanced.pth"

def load_model():
    from models import voicecraft

    voicecraft.VoiceCraft.from_pretrained(f"pyp1/VoiceCraft_{voicecraft_name.replace('.pth', '')}")

tortoise_image = (
    Image.debian_slim(python_version="3.10")
    .apt_install(
        "ffmpeg",
        "git",
        "espeak-ng",
        "python3-pyaudio",
    )
    .pip_install(
        "git+https://github.com/dillionverma/audiocraft",
        "xformers",
        "torchaudio",
        "phonemizer==3.2.1",
        "datasets==2.16.0",
        "torchmetrics==0.11.1",
        "huggingface_hub==0.22.2",
        "sounddevice"
    )
    .run_function(load_model)
)
with tortoise_image.imports():
    import sys, os
    from argparse import Namespace

    import torch
    import torchaudio
    import pydub

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["USER"] = "root"

    sys.path.append('./src/VoiceCraft')
    from models import voicecraft
    from inference_tts_scale import inference_one_sample
    from data.tokenizer import (
        AudioTokenizer,
        TextTokenizer,
    )

@stub.cls(
    image=tortoise_image,
    gpu="A10G",
    container_idle_timeout=600,
    timeout=600,
    mounts=[Mount.from_local_dir(audio_path, remote_path="/audio")]
)
class TTS:
    def __init__(self, person):
        self.decode_config = {
            'top_k': 0,
            'top_p': 0.9,
            'temperature': 1,
            'stop_repetition': 1,
            'kvcache': 1,
            "codec_audio_sr": 16000,
            "codec_sr": 50,
            "silence_tokens": [1388, 1898, 131],
            "sample_batch_size": 16,
        }
        self.person = person
    
    @enter()
    def load_model(self):
        device = torch.device('cuda')
        self.device = device

        self.model = voicecraft.VoiceCraft.from_pretrained(f"pyp1/VoiceCraft_{voicecraft_name.replace('.pth', '')}")
        self.model.to(device)
        self.config = vars(self.model.args)
        self.phn2num = self.model.args.phn2num

        encodec_fn = "src/VoiceCraft/pretrained_models/encodec_4cb2048_giga.th"
        self.audio_tokenizer = AudioTokenizer(signature=encodec_fn, device=device)
        self.text_tokenizer = TextTokenizer(backend="espeak")

    def process_synthesis_result(self, result):
        """
        Converts a audio torch tensor to a binary blob.
        """

        with tempfile.NamedTemporaryFile() as converted_wav_tmp:
            torchaudio.save(
                converted_wav_tmp.name + ".wav",
                result,
                16000,
            )
            wav = io.BytesIO()
            _ = pydub.AudioSegment.from_file(
                converted_wav_tmp.name + ".wav", format="wav"
            ).export(wav, format="wav")

        # return wav.read().decode('utf-8', errors='ignore')
        return wav

    @method()
    def speak(self, text, voice_id=None, top_emotion=None, *args, **kwargs):
        if not text: # empty string in noop case
            return
        if top_emotion == -1: # default, should never happen but in case
            top_emotion = f'/audio/{self.person}_neutral_1.wav'
        else:
            fc = FunctionCall.from_id(top_emotion)
            try:
                top_emotion = fc.get(timeout=120)
            except TimeoutError:
                return Response(status_code=202)
            print("in speak top emotion: ", top_emotion)

        audio_prompt = top_emotion
        emotion_type = top_emotion.split('_')[-2]
        type_idx = EMOTION_TO_IDX[emotion_type]
        sample_idx = int(top_emotion.split('_')[-1].split('.')[0]) - 1
        print(top_emotion, type_idx, sample_idx)
        pre_prompt_text = AUDIO_RAG_TEXT[type_idx][sample_idx]
        text_with_prompt = pre_prompt_text + ' ' + text
        info = torchaudio.info(audio_prompt)
        num_frames = info.num_frames

        concated_audio, gen_audio = inference_one_sample(
            model=self.model,
            model_args=Namespace(**self.config),
            phn2num=self.phn2num,
            text_tokenizer=self.text_tokenizer,
            audio_tokenizer=self.audio_tokenizer,
            audio_fn=audio_prompt,
            target_text=text_with_prompt,
            device=self.device,
            decode_config=self.decode_config,
            prompt_end_frame=num_frames,
        )
        audio_blob = self.process_synthesis_result(gen_audio[0].cpu())

        return audio_blob


# For local testing, run `modal run src.tts_voicecraft::test_voicecraft --text "Where is the best sushi in New York?"`
@stub.function(image=tortoise_image, mounts=[Mount.from_local_dir(audio_path, remote_path="/audio")])
def test_voicecraft(text: str):
    # from pydub.playback import play
    # import sounddevice as sd

    tts = TTS()
    audio_bytes = tts.speak.remote(text)
    # sd.play(audio_bytes, 44100)


