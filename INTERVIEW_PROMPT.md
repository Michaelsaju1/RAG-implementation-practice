# Trideum Corporation — Technical Coding Interview
## RAG Pipeline Implementation
**Time Allotted: 1.5 Hours (90 Minutes)**

---

### Scenario

You are building a Retrieval-Augmented Generation (RAG) system for a defense
organization. Analysts need to ask natural language questions about a collection
of internal policy and operational documents and receive accurate, grounded
answers with source citations.

You have been provided a set of sample documents in the `documents/` folder.
Your task is to build a working RAG pipeline that can answer questions about
these documents.

---

### Rules (Read These First)

1. **Work in `starter_code.py`** — it has skeleton functions you need to fill in.
2. You may use **LangChain**, **LangGraph**, and any of their sub-packages.
3. You may use any open-source embedding model or OpenAI (an `.env` pattern is
   provided — use whatever API keys you have available).
4. You may reference documentation freely (LangChain docs, Stack Overflow, etc.).
5. Focus on **working code** over perfect code. Get something running, then improve.
6. Talk through your thought process out loud — interviewers care about *how*
   you think, not just the final product.

---

### Tasks

Complete the tasks in order. Each builds on the previous one.

#### Part 1 — Core RAG Pipeline (Target: ~35 minutes)

Build a basic end-to-end RAG pipeline:

- [ ] **Task 1.1:** Load all documents from the `documents/` folder.
- [ ] **Task 1.2:** Split documents into chunks with appropriate size and overlap.
- [ ] **Task 1.3:** Generate embeddings and store them in a vector store.
- [ ] **Task 1.4:** Implement a retrieval function that returns the top-k most
      relevant chunks for a given query.
- [ ] **Task 1.5:** Build a generation chain that takes retrieved context and a
      user question, then produces a grounded answer.

**Checkpoint:** You should be able to run a query like
*"What are the data classification levels?"* and get a sensible answer.

---

#### Part 2 — Improve the Pipeline (Target: ~30 minutes)

Make the pipeline more robust and useful:

- [ ] **Task 2.1:** Add a custom prompt template that instructs the LLM to:
  - Only answer based on the provided context
  - Cite which document(s) the answer came from
  - Say "I don't have enough information to answer that" when the context is insufficient
- [ ] **Task 2.2:** Add metadata to your chunks (source filename, chunk index)
      and return source info alongside the generated answer.
- [ ] **Task 2.3:** Implement basic **query preprocessing** — at minimum,
      strip whitespace and handle empty queries gracefully.

**Checkpoint:** Ask *"What is the policy on removable media?"* — you should get
an answer that references which document it came from.

---

#### Part 3 — Conversational RAG (Target: ~20 minutes)

Add multi-turn conversation support:

- [ ] **Task 3.1:** Add conversation memory so the system can handle follow-up
      questions (e.g., "Tell me more about that" or "What about for SECRET
      documents specifically?").
- [ ] **Task 3.2:** Implement a simple conversation loop (can be CLI-based)
      that lets a user ask multiple questions in sequence.

**Checkpoint:** You should be able to ask a question, get an answer, then ask
a follow-up that relies on the prior context.

---

#### Bonus (If Time Remains)

Pick any of these if you finish early:

- [ ] **Bonus A:** Add a re-ranking step after initial retrieval to improve
      relevance.
- [ ] **Bonus B:** Implement hybrid search (combine keyword + semantic search).
- [ ] **Bonus C:** Add basic evaluation — given a question and expected answer,
      measure how well the system performs.
- [ ] **Bonus D:** Use LangGraph to build the pipeline as a stateful graph
      with explicit nodes for retrieval, generation, and a hallucination check.

---

### Sample Questions to Test With

Use these to verify your pipeline works:

1. "What are the data classification levels?"
2. "What is the policy on using removable media?"
3. "How should classified information be transmitted?"
4. "What training is required for personnel handling sensitive data?"
5. "What are the incident response procedures for a data breach?"
6. "What is the difference between CONFIDENTIAL and SECRET handling?"

---

### Evaluation Criteria

Your work will be assessed on:

| Criteria | Weight |
|---|---|
| **Functional correctness** — does it run and produce answers? | 30% |
| **RAG fundamentals** — chunking, retrieval, prompt engineering | 25% |
| **Code quality** — readable, organized, reasonable structure | 20% |
| **Thought process** — how you approach problems and debug | 15% |
| **Enhancements** — Parts 2–3, bonus work | 10% |

---

### Getting Started

```bash
# 1. Install dependencies
pip install langchain langchain-community langchain-openai chromadb python-dotenv

# 2. Set up your API key (create a .env file)
echo "OPENAI_API_KEY=your-key-here" > .env

# 3. Start coding
# Open starter_code.py and begin with Part 1
```

Good luck!
