
import sounddevice as sd 
from scipy.io.wavfile import write

def record_audio(duration, sample_rate):
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    return recording

def save_audio(recording, sample_rate, filename):
    write(filename, sample_rate, recording)

def speech_to_text(client, audio_data):
    with open(audio_data, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            response_format="text",
            file=audio_file, 
            language = "en"
        )
    return transcript

def autoplay_audio(b64_audio):
    md = f"""
    <audio autoplay>
    <source src="data:audio/mp3;base64,{b64_audio}" type="audio/mp3">
    </audio>
    """
    return md
