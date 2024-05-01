"""
GPT language model

Path to weights provided for illustration purposes only,
please check the license before using for commercial purposes!
"""
import os
import asyncio
from pathlib import Path

from modal import Image, Secret, build, enter, method 
import time
from .common import stub


GPT_image = (
    Image.debian_slim(python_version="3.11")
    .pip_install(
        "openai",
        "bs4",
        "scipy",
        "numpy",
        "requests",
        "langchain",
        "langchain-community",
        "langchain-openai",
        "faiss-cpu",
        "langchain-core",
        "langchain-text-splitters",
        "langchainhub",
        "openai",
        "tiktoken",
    )
)

with GPT_image.imports():
    from threading import Thread
    from .llm_utils import load_data
    from langchain_openai import ChatOpenAI
#    from transformers import AutoTokenizer, TextIteratorStreamer
#    from awq import AutoAWQForCausalLM


@stub.cls(image= GPT_image, gpu="T4", container_idle_timeout=300)
class GPT:
    @build()
    def download_model(self):
        # from huggingface_hub import snapshot_download
        # snapshot_download(MODEL_NAME)
        return


    @enter()
    def load_model(self):
        t0 = time.time()
        print("Loading AWQ quantized model...")
        #self.model = AutoAWQForCausalLM.from_quantized(MODEL_NAME, fuse_layers=False, version="GEMV")

        print(f"Model loaded in {time.time() - t0:.2f}s")

        #self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        #self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

    @method()
    async def generate(self, input, api_key=None, history=[]):
        if input == "":
            return

        t0 = time.time()
        self.model = ChatOpenAI(model="gpt-3.5-turbo-0613", openai_api_key=api_key)
        rag_chain = load_data(self.model, api_key)
        # Run generation on separate thread to enable response streaming.
        self.output = ""
        thread = Thread(target=self.generate_output, args=(rag_chain, input))
        thread.start()

        while thread.is_alive():
            if self.output:
                yield self.output
                self.output = ""
            await asyncio.sleep(0.1)

        thread.join()
        print(rag_chain.invoke(input))
        print(f"Output generated in {time.time() - t0:.2f}s")

    def generate_output(self, rag_chain, input):
        for token in rag_chain.stream(input):
            self.output += token


# For local testing, run `modal run -q src.llm_gpt::main --input "Where is the best sushi in New York?"`
@stub.function(secrets=[Secret.from_name("my-openai-secret")])
def test(input: str):
    model = GPT()
    for val in model.generate.remote_gen(input, api_key=os.environ["OPENAI_API_KEY"]):
        print(val, end="", flush=True)
