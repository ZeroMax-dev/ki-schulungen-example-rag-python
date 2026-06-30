# In LangChain v1 Chroma lives in its own package (pip install langchain-chroma).
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Document loader
# https://docs.langchain.com/oss/python/integrations/document_loaders
from langchain_community.document_loaders import TextLoader
loader = TextLoader("alice_in_wonderland.md", encoding="utf-8")
documents = loader.load()

# Split documents with text splitter
# https://docs.langchain.com/oss/python/langchain/retrieval
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=0
)
chunks = []
for document in documents:
    chunks += (text_splitter.create_documents([document.page_content], [document.metadata]))

# Store our documents in a vector store
# https://docs.langchain.com/oss/python/integrations/vectorstores
# db is persistent and can be reused
db = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory="./chroma_db")

# Chat model with stdout streaming output
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import StreamingStdOutCallbackHandler
llm = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0)

# Create a retriever with our vector store
# Use MultiQueryRetriever
# https://docs.langchain.com/oss/python/langchain/retrieval
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
retriever = MultiQueryRetriever.from_llm(
    retriever=db.as_retriever(), llm=llm
)

# A standard RAG prompt (equivalent to the well-known "rlm/rag-prompt" that used
# to be pulled from LangChain Hub, inlined here so there is no extra dependency).
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("human",
     "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"),
])

# create a chain using LCEL
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
rag_chain = (
    {
        "context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# questions to ask
questions = [
    "Who is the main character in the story?",
    "What happens when Alice meets the Cheshire Cat?",
    "What does the White Rabbit say?",
    "What happens at the tea party?",
    "How does Alice get to Wonderland?",
]

# invoke the LCEL chain for each question
for question in questions:
    print(f"Question: {question}\n")
    result = rag_chain.invoke(question)
    print(result)
    print("\n#########################################\n")
