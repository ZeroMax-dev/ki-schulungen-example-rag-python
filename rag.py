# 2-step RAG: retrieval ALWAYS runs before the model answers.
# Docs: https://docs.langchain.com/oss/python/deepagents/retrieval (section "2-step RAG")
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables from .env file
load_dotenv()

# 1. Load the document
# A plain text/markdown file needs no special loader: read it and wrap it in
# LangChain Documents (the splitter below does that for us).
print("Loading document...")
text = Path("alice_in_wonderland.md").read_text(encoding="utf-8")

# 2. Split it into chunks
# https://docs.langchain.com/oss/python/integrations/splitters
print("Text splitter is splitting the document...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,  # remember where each chunk came from
)
chunks = text_splitter.create_documents(
    [text], metadatas=[{"source": "alice_in_wonderland.md"}]
)
print(f"Split the document into {len(chunks)} chunks")

# 3. Embed the chunks and store them in a vector store
# https://docs.langchain.com/oss/python/integrations/vectorstores
# InMemoryVectorStore is perfect for demos; see rag_agent.py for a persistent
# vector database (Chroma).
print("Storing chunks and embeddings in the vector store...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(chunks)

print("Ready to ask!\n###########################################\n")


# 4. Retrieve + augment: before every model call, search the vector store for
# the user's question and put the matching chunks into the system prompt.
# https://docs.langchain.com/oss/python/langchain/middleware/overview
@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    question = request.state["messages"][-1].text
    docs = vector_store.similarity_search(question, k=4)
    context = "\n\n".join(doc.page_content for doc in docs)
    return (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "Use three sentences maximum and keep the answer concise."
        f"\n\nContext:\n{context}"
    )


# 5. Generate: an agent without tools is simply "prompt -> model -> answer".
agent = create_agent(
    model="openai:gpt-5.4-mini",
    tools=[],
    middleware=[prompt_with_context],
)

# list of questions to ask
questions = [
    "Who is the main character in the story?",
    "What happens when Alice meets the Cheshire Cat?",
    "What does the White Rabbit say?",
    "What happens at the tea party?",
    "How does Alice get to Wonderland?",
]

for question in questions:
    print(f"Question: {question}\n")
    # stream_mode="messages" streams the answer token by token
    # https://docs.langchain.com/oss/python/langchain/streaming
    for token, metadata in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="messages",
    ):
        print(token.text, end="", flush=True)
    print("\n\n#########################################\n")
