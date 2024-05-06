"""
Calling ElevenLabs TTS API
"""

import io
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
    from elevenlabs import Voice, VoiceSettings

@stub.cls(
    image=tortoise_image,
    gpu="A10G",
    container_idle_timeout=300,
    timeout=180,
    mounts=[Mount.from_local_dir(audio_path, remote_path="/audio")]
)
class TTS:
    @build()
    def download_model(self):
        print("Downloading model in TTS")
        return


    @enter()
    def load_model(self):
        #t0 = time.time()
        print("Loading model in TTS")
        return
    
    @method()
    def speak(self, text, elevenlabs_api_key):
        client = ElevenLabs(
            api_key=elevenlabs_api_key,
        )
        audio = client.generate(
            text=text,
            voice=Voice(
            voice_id='wKk7Km7FXh9RF5T7EnrG',
            settings=VoiceSettings(stability=0.6, similarity_boost=0.5, style=0.0, use_speaker_boost=True)
            )
        )

        # Concatenate the audio chunks
        audio_data = b"".join(chunk for chunk in audio)

        # Convert the audio data to a binary blob
        audio_blob = io.BytesIO(audio_data)
        return audio_blob