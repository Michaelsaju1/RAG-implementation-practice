"""
Trideum RAG Pipeline — REFERENCE SOLUTION
==========================================
DO NOT look at this until you've attempted the interview yourself!

This is one valid approach. There are many ways to solve this.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# PART 1: Core RAG Pipeline
# ──────────────────────────────────────────────

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def load_documents(directory: str = "documents"):
    """Task 1.1"""
    loader = DirectoryLoader(
        directory,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    print(f"  Loaded {len(documents)} documents")
    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Task 1.2"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # Add chunk index metadata (useful for Part 2)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        # Clean up the source path to just the filename
        chunk.metadata["source"] = os.path.basename(chunk.metadata.get("source", "unknown"))

    print(f"  Split into {len(chunks)} chunks")
    return chunks


def create_vector_store(chunks):
    """Task 1.3"""
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        # persist_directory="./chroma_db",  # Uncomment to persist to disk
    )
    print(f"  Vector store created with {vector_store._collection.count()} entries")
    return vector_store


def retrieve(vector_store, query: str, k: int = 4):
    """Task 1.4"""
    results = vector_store.similarity_search(query, k=k)
    return results


def generate_answer(question: str, context_docs):
    """Task 1.5"""
    # Combine context documents into a single string
    context = "\n\n---\n\n".join([doc.page_content for doc in context_docs])

    # Simple prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the question based on the provided context."),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({"context": context, "question": question})
    return answer


# ──────────────────────────────────────────────
# PART 2: Improve the Pipeline
# ──────────────────────────────────────────────

def create_rag_prompt():
    """Task 2.1"""
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """You are a security policy assistant for the Defense Information Systems Division.
Your role is to answer questions accurately based ONLY on the provided context documents.

Rules:
- Only use information from the provided context to answer questions.
- Always cite which document(s) your answer comes from.
- If the context does not contain enough information to answer the question,
  say "I don't have enough information to answer that based on the available documents."
- Be precise and reference specific section numbers when available."""),
        ("human",
         """Context documents:
{context}

Question: {question}

Provide a thorough answer with citations to the source documents."""),
    ])
    return prompt


def format_answer_with_sources(answer: str, source_docs):
    """Task 2.2"""
    # Collect unique sources
    sources = set()
    for doc in source_docs:
        source_name = doc.metadata.get("source", "Unknown")
        chunk_idx = doc.metadata.get("chunk_index", "?")
        sources.add(f"{source_name} (chunk {chunk_idx})")

    formatted = f"\n{'='*60}\n"
    formatted += f"ANSWER:\n{answer}\n"
    formatted += f"\n{'─'*60}\n"
    formatted += "SOURCES:\n"
    for src in sorted(sources):
        formatted += f"  - {src}\n"
    formatted += f"{'='*60}\n"
    return formatted


def preprocess_query(query: str) -> str:
    """Task 2.3"""
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    cleaned = query.strip()
    return cleaned


# ──────────────────────────────────────────────
# PART 3: Conversational RAG
# ──────────────────────────────────────────────

from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


def create_conversational_chain(vector_store):
    """Task 3.1"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # Step 1: Condense follow-up questions into standalone questions
    condense_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the chat history and a follow-up question, rephrase the follow-up "
         "into a standalone question that captures the full context. If it's already "
         "a standalone question, return it as-is."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    # Step 2: Answer using context
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system",
         """You are a security policy assistant for the Defense Information Systems Division.
Answer based ONLY on the provided context. Cite your sources. If the context
doesn't contain the answer, say "I don't have enough information to answer that."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])

    chat_history = []

    def ask(question: str):
        # Condense the question if there's history
        if chat_history:
            condense_chain = condense_prompt | llm | StrOutputParser()
            standalone_q = condense_chain.invoke({
                "chat_history": chat_history,
                "question": question,
            })
        else:
            standalone_q = question

        # Retrieve relevant docs
        docs = retriever.invoke(standalone_q)

        # Build context string
        context = "\n\n---\n\n".join([
            f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
            for doc in docs
        ])

        # Generate answer
        answer_chain = answer_prompt | llm | StrOutputParser()
        answer = answer_chain.invoke({
            "chat_history": chat_history,
            "context": context,
            "question": standalone_q,
        })

        # Update history
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer))

        return answer, docs

    return ask


def run_conversation(conversational_chain):
    """Task 3.2"""
    print("\n" + "=" * 60)
    print("RAG Conversational Assistant")
    print("Type 'quit' or 'exit' to end the conversation.")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        try:
            query = preprocess_query(user_input)
            answer, docs = conversational_chain(query)
            print(format_answer_with_sources(answer, docs))
        except ValueError as e:
            print(f"Invalid query: {e}")
        except Exception as e:
            print(f"Error: {e}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("DISD RAG Pipeline")
    print("=" * 60)

    # Part 1
    print("\n[Part 1] Building core pipeline...")
    docs = load_documents()
    chunks = split_documents(docs)
    store = create_vector_store(chunks)

    # Part 1 test
    test_q = "What are the data classification levels?"
    print(f"\nTest: {test_q}")
    relevant = retrieve(store, test_q)
    answer = generate_answer(test_q, relevant)
    print(f"Answer: {answer}")

    # Part 2 test
    print("\n[Part 2] Testing with sources...")
    rag_prompt = create_rag_prompt()
    context_str = "\n\n---\n\n".join([
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in relevant
    ])
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = rag_prompt | llm | StrOutputParser()
    answer2 = chain.invoke({"context": context_str, "question": test_q})
    print(format_answer_with_sources(answer2, relevant))

    # Part 3
    print("\n[Part 3] Starting conversation mode...")
    convo = create_conversational_chain(store)
    run_conversation(convo)
