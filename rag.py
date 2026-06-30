# In LangChain v1 Chroma lives in its own package (pip install langchain-chroma),
# not langchain_community anymore.
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Document loader
# https://docs.langchain.com/oss/python/integrations/document_loaders
print("Document loader is loading documents...")
from langchain_community.document_loaders import TextLoader
loader = TextLoader("alice_in_wonderland.md", encoding="utf-8")
documents = loader.load()

# Split documents with text splitter
# https://docs.langchain.com/oss/python/langchain/retrieval
print("Text splitter is splitting documents...")
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=0
)
chunks = []
for document in documents:
    chunks += (text_splitter.create_documents([document.page_content], [document.metadata]))

# Store our documents in a vector store
# https://docs.langchain.com/oss/python/integrations/vectorstores
# (optional) add persist_directory so we can reuse the db without re-creating it
print("Storing documents and embeddings in vector store...")
db = Chroma.from_documents(chunks, OpenAIEmbeddings())

print("Ready to ask!\n###########################################\n")

# Create a retriever with our vector store
# https://docs.langchain.com/oss/python/langchain/retrieval
retriever = db.as_retriever()

# Chat model with stdout streaming output
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import StreamingStdOutCallbackHandler
llm = ChatOpenAI(model="gpt-5.4-mini", streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0)

# Create a prompt template
# https://docs.langchain.com/oss/python/langchain/messages
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("human",
     "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"),
])

# list of questions to ask
questions = [
    "Who is the main character in the story?",
    "What happens when Alice meets the Cheshire Cat?",
    "What does the White Rabbit say?",
    "What happens at the tea party?",
    "How does Alice get to Wonderland?",
]

# put everything together
for question in questions:
    print(f"Question: {question}\n")
    # search for similar documents
    docs = retriever.invoke(question)
    # create context merging docs together
    context = "\n\n".join(doc.page_content for doc in docs)
    # get valorized prompt from template
    prompt_val = prompt.invoke({"context": context, "question": question})
    # get response from llm
    result = llm.invoke(prompt_val.to_messages())
    print("\n#########################################\n")
