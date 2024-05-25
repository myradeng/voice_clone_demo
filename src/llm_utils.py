'''
Based on: https://python.langchain.com/v0.1/docs/use_cases/question_answering/quickstart/
'''

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from scipy.io.wavfile import write
from langchain_community.document_loaders import Docx2txtLoader

# template = """Use the following pieces of context to answer the question at the end.
# Respond naturally and colloquially like you're having a casual conversation. 
# Use two sentences maximum and keep the answer as concise as possible.

# {context}

# Question: {question}

# Helpful Answer:"""

def setup_rag_chain(llm, api_key, person='Myra'):

    if person == 'Myra':
        # Load web blog posts
        web_loader = WebBaseLoader(
            web_paths=(["https://www.unusual.vc/post/andy-rachleff-on-coining-the-term-product-market-fit",
                    "https://myradeng.substack.com/p/why-chatgpt-responds-the-way-it-does?utm_source=profile&utm_medium=reader2/"])
        )
        # Load Word documents
        word_docs = [
            "/docs/myra/diversity_statement.docx",
            "/docs/myra/girls_who_reign.docx",
        #    "/docs/myra/gsb_essays.docx",
            "/docs/myra/personal_statement.docx",
            "/docs/myra/resume.docx"
        #    "/docs/myra/sop.docx",
        #    "/docs/myra/stanford_intro.docx"
        ]
    elif person == 'George':
        web_loader = WebBaseLoader(
            web_paths=([
                "https://lostgeorge.substack.com/p/humanity-and-in-domain-behavior",
            ])
        )
        word_docs = [
            "/docs/george/essay_1.pdf",
            "/docs/george/essay_2.pdf",
            "/docs/george/essay_3.pdf",
            "/docs/george/essay_4.pdf",
            "/docs/george/dump.txt",
            "/docs/george/resume.docx",
        ]
        

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

    ### Contextualize question ###
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    ### Answer question ###
    qa_system_prompt = """Use the following pieces of context to answer the question at the end as if you are Myra.
    Respond naturally and colloquially like you're having a casual conversation. 
    Use two sentences maximum and keep the answer as concise as possible\
    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain
