
import bs4
import sounddevice as sd 

from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import NewsURLLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
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

def load_data(llm, api_key):
    # Load, chunk and index the contents of the blog.
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )

    # loader = NewsURLLoader(
    #     urls=("https://reinvented-magazine.medium.com/women-who-reign-myra-yujia-deng-b5b7688e88dc/",)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(openai_api_key=api_key))

    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def autoplay_audio(b64_audio):
    md = f"""
    <audio autoplay>
    <source src="data:audio/mp3;base64,{b64_audio}" type="audio/mp3">
    </audio>
    """
    return md
