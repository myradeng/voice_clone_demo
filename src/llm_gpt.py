"""
GPT language model

Path to weights provided for illustration purposes only,
please check the license before using for commercial purposes!
"""
import os
from pathlib import Path
from queue import Queue

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
        "docx2txt",
        "loguru",
    )
)

with GPT_image.imports():
    from loguru import logger
    from threading import Thread
    from .llm_utils import setup_rag_chain
    from langchain_openai import ChatOpenAI
    from langchain_community.chat_message_histories import ChatMessageHistory
    from langchain_core.runnables.history import RunnableWithMessageHistory

@stub.cls(image=GPT_image, 
          gpu="T4", 
          container_idle_timeout=300, 
          mounts=[Mount.from_local_dir(docs_path, remote_path="/docs")])
class GPT:
    def __init__(self, openai_api_key, person='Myra'):
        self.model = ChatOpenAI(model="gpt-3.5-turbo-0613", openai_api_key=openai_api_key, max_tokens=100)
        self.rag_chain = setup_rag_chain(self.model, openai_api_key, person)
        self.store = {}
        self.conversational_rag_chain = RunnableWithMessageHistory(
            self.rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
    
    def get_session_history(self, session_id: str):
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    @method()
    async def generate(self, input):
        if input == "":
            return

        t0 = time.time()
        
        # Run generation on separate thread to enable response streaming.
        streamer = Queue()
        thread = Thread(target=self.generate_output, args=(self.conversational_rag_chain, input, streamer))
        thread.start()

        timeout_start = time.time()
        while time.time() < timeout_start + 10:
            output = streamer.get(timeout=10)
            if output is None:
                break
            yield output

        thread.join()

        logger.info(f"Output generated in {time.time() - t0:.2f}s")

    def generate_output(self, rag_chain, input, streamer):
        for token in rag_chain.stream({"input": input}, config={"configurable": {"session_id": "123"}}):
            if 'answer' in token:
                streamer.put(token['answer'])
        streamer.put(None)


# For local testing, run `modal run -q src.llm_gpt::test_gpt --input "Where is the best sushi in New York?"`
@stub.function(image=GPT_image, secrets=[Secret.from_name("my-openai-secret")])
def test_gpt(input: str):
    input1 = 'Where did you go to college?'
    model = GPT(os.environ["OPENAI_API_KEY"])
    for val in model.generate.remote_gen(input1):
        print(val, end="", flush=True)
    input2 = 'Could you repeat the answer?'
    for val in model.generate.remote_gen(input2):
        print(val, end="", flush=True)
