# Agentic RAG: the agent decides WHEN and HOW to search.
# Retrieval is just a tool; the model writes its own search queries and may
# search several times (e.g. once per sub-question) before it answers.
# Docs: https://docs.langchain.com/oss/python/deepagents/retrieval (section "Agentic RAG")
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables from .env file
load_dotenv()

# Store our chunks in a persistent vector database (Chroma, pip install langchain-chroma).
# The database lives in ./chroma_db and is reused on the next run.
# https://docs.langchain.com/oss/python/integrations/vectorstores/chroma
vector_store = Chroma(
    collection_name="alice_in_wonderland",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    persist_directory="./chroma_db",
)

# Only load, split and embed the book if the database is still empty
if not vector_store.get(limit=1)["ids"]:
    print("Indexing alice_in_wonderland.md (first run only)...")
    text = Path("alice_in_wonderland.md").read_text(encoding="utf-8")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        add_start_index=True,
    )
    chunks = text_splitter.create_documents(
        [text], metadatas=[{"source": "alice_in_wonderland.md"}]
    )
    vector_store.add_documents(chunks)


# The retrieval tool. The docstring tells the model what the tool is for.
# response_format="content_and_artifact" sends the text to the model and keeps
# the raw Documents as an artifact on the ToolMessage (handy for citations).
# https://docs.langchain.com/oss/python/langchain/tools
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Search the book "Alice's Adventures in Wonderland" for passages relevant to the query."""
    docs = vector_store.similarity_search(query, k=4)
    serialized = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}" for doc in docs
    )
    return serialized, docs


# gpt-6-luna is a reasoning model. Reasoning + function tools requires OpenAI's
# Responses API, so we turn it on explicitly.
# https://docs.langchain.com/oss/python/integrations/chat/openai
model = ChatOpenAI(
    model="gpt-6-luna",
    reasoning={"effort": "medium"},  # "none" | "low" | "medium" | "high" | ...
    use_responses_api=True,
)

agent = create_agent(
    model=model,
    tools=[retrieve_context],
    system_prompt=(
        "You answer questions about the book \"Alice's Adventures in Wonderland\". "
        "Always use the retrieve_context tool to look up relevant passages before "
        "answering. If the passages don't contain the answer, say that you don't know. "
        "Keep the answer concise (three sentences maximum)."
    ),
)

def print_update(update):
    """Print what each agent step produced: tool calls, tool results or the answer."""
    for step, data in update.items():
        message = data["messages"][-1]
        if getattr(message, "tool_calls", None):
            for call in message.tool_calls:
                print(f"[{step}] calls {call['name']}({call['args']})")
        elif step == "tools":
            print(f"[{step}] {message.name} returned {len(message.text)} characters")
        else:
            # .text joins the text blocks (the Responses API also returns reasoning blocks)
            print(f"[{step}] answer:\n{message.text}")


# questions to ask
questions = [
    "Who is the main character in the story?",
    "What happens when Alice meets the Cheshire Cat?",
    "What does the White Rabbit say?",
    "What happens at the tea party?",
    "How does Alice get to Wonderland?",
]

for question in questions:
    print(f"Question: {question}\n")
    # stream_mode="updates" yields what each step (model / tools) produced, so
    # we can watch the agent call the tool (and with which query) before it answers.
    # https://docs.langchain.com/oss/python/langchain/streaming
    for update in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="updates",
    ):
        print_update(update)
    print("\n#########################################\n")
