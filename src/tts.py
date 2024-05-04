"""
Calling ElevenLabs TTS API
"""

import io
import os
from modal import Image, method, build, enter, Mount
from pathlib import Path
from .common import stub

audio_path = Path(__file__).with_name("audio").resolve()

tortoise_image = (
    Image.debian_slim(python_version="3.12")
    .pip_install(
        "elevenlabs",
    )
)
with tortoise_image.imports():
    from elevenlabs.client import ElevenLabs
    from elevenlabs import play, save

@stub.cls(
    image=tortoise_image,
    gpu="A10G",
    container_idle_timeout=300,
    timeout=180,
    mounts=[Mount.from_local_dir(audio_path, remote_path="/audio")]
)
class TTS:
    import io 
    @build()
    def download_model(self):
        # from huggingface_hub import snapshot_download
        # snapshot_download(MODEL_NAME)
        print("Downloading model in TTS")
        return


    @enter()
    def load_model(self):
        #t0 = time.time()
        print("Loading model in TTS")
        return

        #self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        #self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    @method()
    def speak(self, text, voices=["geralt"]):
        print("In TTS speak")
        client = ElevenLabs(
            api_key="53173cda2e720caa4b7d7e00b3cdb7fb", # Defaults to ELEVEN_API_KEY
        )
        print("Current working directory:", os.getcwd())
        voice = client.clone(
            name="Myra",
            description="Myra's voice", # Optional
            files=["/audio/myra_1.wav", "/audio/myra_2.wav"],
        )

        audio = client.generate(text=text, voice=voice)

        # Concatenate the audio chunks
        audio_data = b"".join(chunk for chunk in audio)

        # Convert the audio data to a binary blob
        audio_blob = io.BytesIO(audio_data)
        return audio_blob