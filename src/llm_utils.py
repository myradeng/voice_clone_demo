
from langchain_community.document_loaders import WebBaseLoader
from langchain.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from scipy.io.wavfile import write
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import Docx2txtLoader

template = """Use the following pieces of context to answer the question at the end.
Respond naturally like you're having a casual conversation. 
Use two sentences maximum and keep the answer as concise as possible.

{context}

Question: {question}

Helpful Answer:"""

def load_data(llm, api_key, loaders=[]):
    # Load web blog posts
    web_loader = WebBaseLoader(
        web_paths=(["https://www.unusual.vc/post/andy-rachleff-on-coining-the-term-product-market-fit",
                   "https://myradeng.substack.com/p/why-chatgpt-responds-the-way-it-does?utm_source=profile&utm_medium=reader2/"])
    )
    # Load Word documents
    word_docs = ["/docs/diversity_statement.docx",
                   "/docs/girls_who_reign.docx",
                #    "/docs/gsb_essays.docx",
                   "/docs/personal_statement.docx",
                   "/docs/resume.docx"
                #    "/docs/sop.docx",
                #    "/docs/stanford_intro.docx"]

    # Load each Word document using a list comprehension
    word_loaders = [Docx2txtLoader(doc) for doc in word_docs]

    # Combine the loaded documents from web and Word sources
    docs = web_loader.load()
    for loader in word_loaders:
         docs.extend(loader.load())

    print(str(len(docs)) + " total docs loaded in llm_utils.py")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings(openai_api_key=api_key))

    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever()
    #prompt = hub.pull("rlm/rag-prompt")
    prompt = PromptTemplate.from_template(template)
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain