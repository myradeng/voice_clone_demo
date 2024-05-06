"""
GPT language model

Path to weights provided for illustration purposes only,
please check the license before using for commercial purposes!
"""
import os
from pathlib import Path

from modal import Image, Secret, build, enter, method, Mount
import time
from .common import stub

docs_path = Path(__file__).with_name("docs").resolve()

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
        "unstructured", 
        "docx2txt"
    )
)

with GPT_image.imports():
    from threading import Thread
    from .llm_utils import load_data
    from langchain_openai import ChatOpenAI


@stub.cls(image= GPT_image, 
          gpu="T4", 
          container_idle_timeout=300, 
          mounts=[Mount.from_local_dir(docs_path, remote_path="/docs")])
class GPT:
    @build()
    def download_model(self):
        self.web_loader = None
        self.word_loaders = None
        return

    @enter()
    def load_model(self):
        return
    
    @method()
    async def generate(self, input, api_key=None, history=[]):
        if input == "":
            return

        t0 = time.time()
        self.model = ChatOpenAI(model="gpt-3.5-turbo-0613", openai_api_key=api_key, max_tokens=100)
        rag_chain = load_data(self.model, api_key)
        # Run generation on separate thread to enable response streaming.
        thread = Thread(target=self.generate_output, args=(rag_chain, input))
        thread.start()

        while thread.is_alive():
            yield rag_chain.invoke(input)

        thread.join()
        #print(rag_chain.invoke(input))
        print(f"Output generated in {time.time() - t0:.2f}s")

    def generate_output(self, rag_chain, input):
        for token in rag_chain.stream(input):
            self.output += token


# For local testing, run `modal run -q src.llm_gpt::test_gpt --input "Where is the best sushi in New York?"`
@stub.function(secrets=[Secret.from_name("my-openai-secret")])
def test_gpt(input: str):
    model = GPT()
    for val in model.generate.remote_gen(input, api_key=os.environ["OPENAI_API_KEY"]):
        print(val, end="", flush=True)
