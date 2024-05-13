"""
Main web application service. Serves the static frontend as well as
API routes for transcription, language model generation and text-to-speech.
"""

import json
from pathlib import Path
import os
from modal import Mount, asgi_app, Secret

from .common import stub
from .llm_gpt import GPT
from .transcriber import Whisper
from .tts import TTS

static_path = Path(__file__).with_name("frontend").resolve()

PUNCTUATION = [".", "?", "!", ":", ";", "*"]


@stub.function(
    mounts=[Mount.from_local_dir(static_path, remote_path="/assets")],
    container_idle_timeout=300,
    timeout=600,
    secrets=[Secret.from_name("my-openai-secret"), Secret.from_name("my-elevenlabs-secret")]
)
@asgi_app()
def web():
    from fastapi import FastAPI, Request
    from fastapi.responses import Response, StreamingResponse
    from fastapi.staticfiles import StaticFiles

    web_app = FastAPI()
    transcriber = Whisper()
    llm = GPT(os.environ["OPENAI_API_KEY"])
    tts = TTS(os.environ["ELEVENLABS_API_KEY"])

    @web_app.post("/transcribe")
    async def transcribe(request: Request):
        bytes = await request.body()
        result = transcriber.transcribe_segment.remote(bytes)
        return result["text"]

    @web_app.post("/generate")
    async def generate(request: Request):
        body = await request.json()
        tts_enabled = True
        print("In app.py generate() tts_enabled", body["tts"])

        if "noop" in body:
            llm.generate.spawn("")
            # Warm up 3 containers for now.
            if tts_enabled:
                for _ in range(3):
                    tts.speak.spawn("")
            return

        def speak(sentence, elevenlabs_voice_id=None):
            if tts_enabled:
                fc = tts.speak.spawn(sentence, elevenlabs_voice_id)
                return {
                    "type": "audio",
                    "value": fc.object_id,
                }
            else:
                print("Not tts")
                return {
                    "type": "sentence",
                    "value": sentence,
                }

        def gen():
            sentence = ""
            
            elevenlabs_voice_id = os.environ["ELEVENLABS_VOICE_ID"]
            for segment in llm.generate.remote_gen(body["input"]):
                yield {"type": "text", "value": segment}
                sentence += segment
                for p in PUNCTUATION:
                    if p in sentence:
                        prev_sentence, new_sentence = sentence.rsplit(p, 1)
                        yield speak(prev_sentence, elevenlabs_voice_id)
                        sentence = new_sentence

            if sentence:
                yield speak(sentence, elevenlabs_voice_id)

        def gen_serialized():
            for i in gen():
                yield json.dumps(i) + "\x1e"

        return StreamingResponse(
            gen_serialized(),
            media_type="text/event-stream",
        )

    @web_app.get("/audio/{call_id}")
    async def get_audio(call_id: str):
        from modal.functions import FunctionCall
        
        function_call = FunctionCall.from_id(call_id)
        try:
            result = function_call.get(timeout=30)
        except TimeoutError:
            return Response(status_code=202)

        if result is None:
            return Response(status_code=204)
        print("streaming response in app_py")
        return StreamingResponse(result, media_type="audio/wav")

    @web_app.delete("/audio/{call_id}")
    async def cancel_audio(call_id: str):
        from modal.functions import FunctionCall

        print("Cancelling", call_id)
        function_call = FunctionCall.from_id(call_id)
        function_call.cancel()

    web_app.mount("/", StaticFiles(directory="/assets", html=True))
    return web_app
