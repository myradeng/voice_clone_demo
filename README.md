# Expressive Voice Agents

Recent advancements in speech and language models have enabled the generation of high quality, natural-sounding speech. However,
current voice chat systems often lack the ability to generate properly emotive and personalized responses. In this project, we present
an **integrated voice agent system** that leverages these advancements to create **expressive and personalized voice agents**. 

Our system consists of three main components: speech transcription, language model generation with question answering, and text-to-speech synthesis. We
personalize the language model and speech synthesis components to specific target individuals, focusing on the use case of casual coffee
chats or "get-to-know-you" conversations. Our experiments demonstrate the effectiveness of retrieval augmented generation for providing
realistic and personalized answers, and finetuned TTS models for generating high quality personalized speech. Furthermore, we introduce AudioRAG, a novel method for retrieving emotion-matching audio samples to generate expressive speech. The resulting voice agent system achieves high quality in terms of realism, expressiveness, and personalization with reasonable latency, offering a promising direction for AI-driven personal voice clones.

## File structure

1. React frontend ([`src/frontend/`](./src/frontend/))
2. FastAPI server ([`src/app.py`](./src/app.py))
3. Whisper transcription module ([`src/transcriber.py`](./src/transcriber.py))
4. Tortoise text-to-speech module ([`src/tts.py`](./src/tts.py))
5. Zephyr language model module ([`src/llm_zephyr.py`](./src/llm_zephyr.py))

Read the accompanying [docs](https://modal.com/docs/examples/llm-voice-chat) for a detailed look at each of these components.

## Developing locally

### Requirements

- `modal` installed in your current Python virtual environment (`pip install modal`)
- A [Modal](http://modal.com/) account
- A Modal token set up in your environment (`modal token new`)

- To get the VoiceCraft submodule, do
```
git submodule update
```
The way VoiceCraft is coded right now requies manually downloading the encodec function, so do `wget https://huggingface.co/pyp1/VoiceCraft/resolve/main/encodec_4cb2048_giga.th` and make sure the file is in the directory: `VoiceCraft/pretrained_models/`

### Develop on Modal

To [serve](https://modal.com/docs/guide/webhooks#developing-with-modal-serve) the app on Modal, run this command from the root directory of this repo:

```shell
modal serve src.app
```

In the terminal output, you'll find a URL that you can visit to use your app. While the `modal serve` process is running, changes to any of the project files will be automatically applied. `Ctrl+C` will stop the app.

### Deploy to Modal

Once you're happy with your changes, [deploy](https://modal.com/docs/guide/managing-deployments#creating-deployments) your app:

```shell
modal deploy src.app
```

[Note that leaving the app deployed on Modal doesn't cost you anything! Modal apps are serverless and scale to 0 when not in use.]
