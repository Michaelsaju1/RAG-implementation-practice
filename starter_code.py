"""
Trideum RAG Pipeline — Coding Interview
=========================================
Fill in the functions below to build a working RAG pipeline.
Each function has a docstring explaining what it should do.

Start with Part 1 and work your way down.
"""

import os
from dotenv import load_dotenv
import glob

load_dotenv()

# ──────────────────────────────────────────────
# PART 1: Core RAG Pipeline
# ──────────────────────────────────────────────

def load_documents(directory: str = "documents"):
    """
    Task 1.1 — Load all .txt files from the given directory.

    Returns:
        A list of LangChain Document objects.

    Hints:
        - Look at langchain_community.document_loaders
        - DirectoryLoader or TextLoader are good options
    
    file_paths = glob.glob(os.path.join(directory, "*.txt"))
    print(file_paths)

    documents = []
    for each_file in file_paths:
        with open(each_file, "r", encoding="utf-8") as file:
            content = file.read()
            documents.append(content)
    print(f"loaded {len(documents)} files.")
    """
    

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Task 1.2 — Split documents into chunks.

    Args:
        documents: List of Document objects from load_documents()
        chunk_size: Maximum characters per chunk
        chunk_overlap: Overlap between consecutive chunks

    Returns:
        A list of chunked Document objects.

    Hints:
        - Look at langchain.text_splitter
        - RecursiveCharacterTextSplitter is a solid default
    """
    # YOUR CODE HERE
    pass


def create_vector_store(chunks):
    """
    Task 1.3 — Create embeddings and store them in a vector store.

    Args:
        chunks: List of chunked Document objects from split_documents()

    Returns:
        A vector store instance that supports similarity search.

    Hints:
        - You need an embedding model (OpenAIEmbeddings, or a free alternative)
        - Chroma is a good local vector store choice
    """
    # YOUR CODE HERE
    pass


def retrieve(vector_store, query: str, k: int = 4):
    """
    Task 1.4 — Retrieve the top-k most relevant chunks for a query.

    Args:
        vector_store: The vector store from create_vector_store()
        query: The user's question
        k: Number of chunks to retrieve

    Returns:
        A list of relevant Document objects.

    Hints:
        - Most vector stores have a .similarity_search() method
    """
    # YOUR CODE HERE
    pass


def generate_answer(question: str, context_docs):
    """
    Task 1.5 — Generate an answer using an LLM and the retrieved context.

    Args:
        question: The user's question
        context_docs: List of relevant Document objects from retrieve()

    Returns:
        A string containing the generated answer.

    Hints:
        - Combine the context documents into a single string
        - Use ChatOpenAI or another LangChain LLM wrapper
        - A simple prompt: "Answer the question based on the context"
    """
    # YOUR CODE HERE
    pass


# ──────────────────────────────────────────────
# PART 2: Improve the Pipeline
# ──────────────────────────────────────────────

def create_rag_prompt():
    """
    Task 2.1 — Create a custom prompt template for the RAG chain.

    The prompt should instruct the LLM to:
      - Only answer based on the provided context
      - Cite which document(s) the answer came from
      - Say "I don't have enough information" when context is insufficient

    Returns:
        A LangChain PromptTemplate or ChatPromptTemplate.

    Hints:
        - Look at langchain_core.prompts
        - Include placeholders for {context} and {question}
    """
    # YOUR CODE HERE
    pass


def format_answer_with_sources(answer: str, source_docs):
    """
    Task 2.2 — Format the answer to include source citations.

    Args:
        answer: The generated answer string
        source_docs: The Document objects used as context

    Returns:
        A formatted string with the answer and source information.

    Hints:
        - Each Document has a .metadata dict with source info
        - You added source filename and chunk index during chunking, right?
    """
    # YOUR CODE HERE
    pass


def preprocess_query(query: str) -> str:
    """
    Task 2.3 — Clean and validate the user query.

    Args:
        query: Raw user input

    Returns:
        Cleaned query string, or raises ValueError for invalid queries.

    Hints:
        - Strip whitespace
        - Handle empty or whitespace-only strings
        - Optionally: lowercase, remove special chars, etc.
    """
    # YOUR CODE HERE
    pass


# ──────────────────────────────────────────────
# PART 3: Conversational RAG
# ──────────────────────────────────────────────

def create_conversational_chain(vector_store):
    """
    Task 3.1 — Create a RAG chain with conversation memory.

    Args:
        vector_store: The vector store for retrieval

    Returns:
        A chain or function that maintains conversation history
        and can answer follow-up questions.

    Hints:
        - Look at ConversationBufferMemory or ChatMessageHistory
        - The chain needs to consider chat history when reformulating queries
        - A "condense question" step can rephrase follow-ups into standalone questions
    """
    # YOUR CODE HERE
    pass


def run_conversation(conversational_chain):
    """
    Task 3.2 — Run an interactive conversation loop.

    Args:
        conversational_chain: The chain from create_conversational_chain()

    The loop should:
        - Prompt the user for input
        - Process the query through the conversational chain
        - Display the answer with sources
        - Continue until the user types 'quit' or 'exit'
    """
    # YOUR CODE HERE
    pass


# ──────────────────────────────────────────────
# MAIN — Wire it all together
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # Part 1: Build the core pipeline
    print("Loading documents...")
    docs = load_documents()

    print("Splitting documents...")
    chunks = split_documents(docs)

    print("Creating vector store...")
    store = create_vector_store(chunks)

    # Quick test
    test_question = "What are the data classification levels?"
    print(f"\nTest question: {test_question}")
    relevant_docs = retrieve(store, test_question)
    answer = generate_answer(test_question, relevant_docs)
    print(f"Answer: {answer}")

    # Part 2: Uncomment after completing Part 2
    # formatted = format_answer_with_sources(answer, relevant_docs)
    # print(formatted)

    # Part 3: Uncomment after completing Part 3
    # chain = create_conversational_chain(store)
    # run_conversation(chain)
