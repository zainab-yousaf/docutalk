# 📄 PDF Q&A Chatbot

Upload any PDF, ask questions about it in plain English, and get answers
**grounded in the document** with page-number citations so you can verify
every claim. Built as a learning project to understand Retrieval-Augmented
Generation (RAG) end to end, from raw PDF bytes to a chat UI.

No OpenAI account needed. No paid embeddings. Just a free OpenRouter key.

---

## Table of contents

- [What this actually does](#what-this-actually-does)
- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Quick start](#quick-start)
- [Getting an OpenRouter API key (free)](#getting-an-openrouter-api-key-free)
- [Running the app](#running-the-app)
- [How it works, for learners](#how-it-works-for-learners)
- [Running tests](#running-tests)
- [Troubleshooting](#troubleshooting)
- [Swapping the LLM provider](#swapping-the-llm-provider)
- [Deploying it publicly (read this first)](#deploying-it-publicly-read-this-first)
- [Why these choices](#why-these-choices)

---

## What this actually does

1. You upload a PDF.
2. The app reads it, splits it into small overlapping chunks, and converts
   each chunk into a vector (a list of numbers that captures its meaning).
3. When you ask a question, the app converts your question into a vector
   too, and finds the chunks whose vectors are most similar i.e. the
   parts of the PDF most relevant to what you asked.
4. Those chunks (not the whole PDF) get handed to an LLM along with your
   question, with strict instructions: *answer only from this context, and
   say which page each fact came from.*
5. You get a streamed answer with citations, and can expand "Sources" to
   see the exact text it used.

This is called **RAG (Retrieval-Augmented Generation)** instead of
relying on what the model memorized during training, you retrieve the
relevant facts yourself and hand them to the model at question time. It's
the standard pattern behind most "chat with your documents" products.

---

## Architecture

```
                 PDF upload
                     │
                     ▼
        ┌─────────────────────────┐
        │  1. INGESTION            │  pdfplumber extracts text per page
        │  src/ingestion.py        │  → split into ~800-token chunks,
        │                          │    120-token overlap, tagged with
        │                          │    the page number they came from
        └────────────┬─────────────┘
                     ▼
        ┌─────────────────────────┐
        │  2. EMBEDDING             │  sentence-transformers
        │  src/embeddings.py        │  (all-MiniLM-L6-v2) runs locally,
        │                          │  free, no API key
        │                          │  → vectors stored in ChromaDB
        └────────────┬─────────────┘
                     ▼
   question ──▶ ┌─────────────────────────┐
                │  3. RETRIEVAL              │  embed the question (same
                │  src/retrieval.py          │  local model) → cosine
                │                            │  similarity search → top 4
                │                            │  most relevant chunks
                └────────────┬───────────────┘
                             ▼
                ┌─────────────────────────┐
                │  4. GENERATION             │  chunks + question →
                │  src/generation.py          │  prompt template →
                │  src/prompts.py             │  OpenRouter
                │  src/llm_client.py          │  (openai/gpt-5.6-luna) →
                │                             │  streamed, cited answer
                └────────────┬───────────────┘
                             ▼
                ┌─────────────────────────┐
                │  5. UI                      │  Streamlit: upload, chat,
                │  app.py                     │  streaming text, expandable
                │                             │  "Sources" with page numbers
                └─────────────────────────────┘
```

**Why local embeddings + OpenRouter for chat, instead of one provider for
everything?** OpenRouter gives access to a huge range of chat models
through one API key and one bill, which is great for experimenting but
it doesn't offer an embeddings endpoint. Running embeddings locally via
`sentence-transformers` means the "understand the document" step is
completely free forever, no matter how many PDFs you index, and only the
"generate an answer" step draws on paid API credit.

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| PDF parsing | [`pdfplumber`](https://github.com/jsvine/pdfplumber) | Reliable per-page text extraction |
| Chunking | [`langchain-text-splitters`](https://python.langchain.com/) | Token-aware splitting, not naive character cuts |
| Embeddings | [`sentence-transformers`](https://www.sbert.net/) (`all-MiniLM-L6-v2`) | Free, runs locally, no API key, good enough quality for this scale |
| Vector store | [`ChromaDB`](https://www.trychroma.com/) | Zero-config, persists to a local folder, no server to run |
| LLM | [OpenRouter](https://openrouter.ai/) (`openai/gpt-5.6-luna`) | One key for many models, pay-as-you-go, easy to swap models later |
| UI | [`Streamlit`](https://streamlit.io/) | Built-in chat components, fastest way to ship a usable UI |

---

## Project structure

```
pdf-qa-chatbot/
├── app.py                  # Streamlit entrypoint upload, chat, citations
├── src/
│   ├── config.py            # every env-driven setting, in one place
│   ├── ingestion.py          # PDF → page-aware, token-sized chunks
│   ├── embeddings.py         # local embedding model + Chroma storage
│   ├── retrieval.py          # top-k similarity search
│   ├── prompts.py            # the system prompt + context assembly
│   ├── llm_client.py         # swappable LLM provider wrapper
│   └── generation.py         # wires retrieval + prompt + LLM together
├── tests/
│   └── test_prompts.py       # unit tests that need no API key at all
├── requirements.txt
├── pyproject.toml            # ruff + pytest config
├── .env.example               # copy this to .env and fill in your key
└── .gitignore
```

**Why split it this way?** Each file does exactly one job in the pipeline
above. If you want to swap the vector store, you touch `embeddings.py` and
`retrieval.py` nothing else changes. If you want to swap the LLM, you
touch `llm_client.py` the prompt, retrieval, and UI code never need to
know or care which provider is behind it.

---

## Quick start

You need **Python 3.11 or newer** installed. Everything else gets set up
below.

```bash
# 1. Clone this repo (or just cd into it if you already have it)
git clone <your-repo-url>
cd pdf-qa-chatbot

# 2. Create a virtual environment — keeps this project's packages
#    separate from everything else on your machine
python -m venv venv

# 3. Activate it
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 4. Install dependencies
pip install -r requirements.txt

# 5. Set up your API key
copy .env.example .env       # Windows
# cp .env.example .env       # macOS / Linux
# now open .env and paste in your OpenRouter key (see next section)

# 6. Run it
streamlit run app.py
```

The app opens automatically at **http://localhost:8501**.

---

## Getting an OpenRouter API key (free)

1. Go to **https://openrouter.ai** and sign up (free, no credit card required
   just to create an account)
2. Go to **https://openrouter.ai/keys** and click **Create Key**
3. Copy the key (starts with `sk-or-`)
4. Paste it into your `.env` file:
   ```
   OPENROUTER_API_KEY=sk-or-your-actual-key-here
   ```
5. Add credit to your OpenRouter account under **Settings → Credits** you
   don't need much. Chat models on OpenRouter are typically fractions of a
   cent per question; a couple of dollars covers a lot of testing.

> ⚠️ **Never commit your `.env` file or paste your real key into chat,
> code comments, or a GitHub issue.** `.env` is already listed in
> `.gitignore` so `git` won't track it — but always double-check with
> `git status` before your first commit.

---

## Running the app

Once `streamlit run app.py` is running:

1. Click **"Upload a PDF"** and choose any PDF file
2. Wait for **"Indexed ... chunks — Ask away!"** this is steps 1 and 2 of
   the pipeline running (parsing + embedding). Takes a few seconds for a
   typical document.
3. Type a question in the chat box at the bottom
4. Watch the answer stream in, then click **"Sources: pages ..."** to see
   exactly which parts of the PDF it used

Re-uploading the *same* PDF later skips re-embedding — each PDF's vectors
are cached locally under `.chroma/`, keyed by a hash of the file content.

---

## How it works, for learners

If you're reading this to actually learn RAG (not just run the app), here's
the order to read the code in:

1. **`src/ingestion.py`** the simplest file. Takes a PDF path, returns a
   list of `Chunk` objects. Read `chunk_pdf()` first.
2. **`src/embeddings.py`** `embed_texts()` turns a list of strings into a
   list of vectors. `build_or_load_collection()` stores those vectors in
   Chroma alongside the original text and page number, so a search later
   can return "here's the matching text, and here's its page."
3. **`src/retrieval.py`** `retrieve()` embeds *the question* using the
   exact same model, then asks Chroma for the closest-matching stored
   vectors. This is the "search" half of RAG.
4. **`src/prompts.py`** `build_messages()` is where retrieved chunks
   actually become a prompt. Look at `SYSTEM_PROMPT` — this is what stops
   the model from making things up: it's explicitly told to answer only
   from the given context and to admit when it doesn't know.
5. **`src/generation.py`** `answer_question()` is the one function that
   ties retrieval + prompting + the LLM call together. This is "the RAG
   pipeline" as a single call.
6. **`app.py`** everything above, wired into a UI. `st.session_state`
   is Streamlit's way of remembering things (chat history, which PDF is
   loaded) between user interactions.

**Key idea to take away:** the LLM never sees the whole PDF. It only ever
sees the ~4 chunks that retrieval decided were most relevant to your
specific question. This is *why* RAG scales to huge documents that would
never fit in a model's context window, and why answers can be traced back
to specific pages.

---

## Running tests

```bash
pytest
```

`tests/test_prompts.py` checks the prompt-assembly logic without touching
the network no API key required, runs in under a second. Good file to
copy the pattern from if you add more tests.

---

## Troubleshooting

These are real errors this project hit during setup leaving them here so
you don't have to debug them from scratch.

| Error | Cause | Fix |
|---|---|---|
| `TypeError: Client.__init__() got an unexpected keyword argument 'proxies'` | `openai` package version too old for the installed `httpx` version | `pip install --upgrade "openai>=1.55.0,<2.0.0"` |
| `RateLimitError: ... insufficient_quota` / `credit_balance_exhausted` | Your API account (OpenAI or OpenRouter) has $0 balance | Add credit at the relevant provider's billing page |
| `RuntimeError: OPENROUTER_API_KEY is not set` | `.env` file missing, or missing that line | Copy `.env.example` to `.env` and fill in your key |
| App hangs or errors on first PDF upload | The local embedding model is downloading (~90MB) the very first time it runs | Wait — it only downloads once, then it's cached under `~/.cache/huggingface` |
| `pip install` fails trying to build `numpy` from source | Your Python version is newer than the pinned dependency versions support (no pre-built wheel) | Make sure `chromadb>=1.0.0` in `requirements.txt` (older pins force an old `numpy` with no wheel for newer Python) |
| Model not found / 400 error on chat | `LLM_MODEL` in `.env` is set to something that doesn't exist on OpenRouter | Check the exact model ID at [openrouter.ai/models](https://openrouter.ai/models) — copy it exactly, OpenRouter model IDs are case-sensitive |

---

## Swapping the LLM provider

Everything about "which model answers questions" lives in
**`src/llm_client.py`**. To point at a different provider:

1. Write a new class with a `stream_chat(messages: list[dict]) -> Iterator[str]`
   method — same shape as `OpenRouterChatClient`.
2. Point `get_llm_client()` at your new class.

Nothing in `app.py`, `generation.py`, `prompts.py`, or `retrieval.py` needs
to change this is the whole point of keeping the LLM behind one thin
wrapper.

---

## Deploying it publicly (read this first)

Hosting on **Streamlit Community Cloud** or **Hugging Face Spaces** is
free. The thing that *isn't* free is API usage and if you deploy with
your real API key baked in as a "secret," **anyone who finds the public
link can ask it questions using your credit.**

If you do deploy:

1. Push this repo to GitHub (your `.env` is gitignored, so your key won't
   go with it good)
2. Add `OPENROUTER_API_KEY` as a **secret** in your hosting platform's
   settings (not committed to the repo)
3. **Set a hard spending limit on your OpenRouter key first** under your
   OpenRouter dashboard, you can cap a key at a fixed dollar amount, so
   even worst-case public traffic costs you a known, bounded amount
   instead of an open-ended risk

For a portfolio, a screen recording or GIF of the app running locally is
just as effective as a live public link, with zero ongoing cost risk.

---

## Why these choices

A few decisions worth explaining, since "why not X instead" is a fair
question for anyone learning from this repo:

- **ChromaDB over Pinecone/a hosted vector DB** zero setup, persists to
  a local folder, nothing to sign up for. Swap it out once you actually
  need multi-machine access or scale past what a laptop can hold.
- **Token-based chunking over character-based** chunk sizes are set in
  *tokens* (what the model actually counts against its context window),
  not characters, so a chunk of "800 tokens" is a meaningful, consistent
  unit regardless of how dense the text is.
- **15% chunk overlap** prevents a sentence that happens to straddle a
  chunk boundary from losing its meaning in both halves.
- **Streamlit over a custom frontend** this project is about the RAG
  pipeline, not building a chat UI from scratch. `st.chat_message`,
  `st.chat_input`, and `st.write_stream` cover everything needed here in
  a few lines each.
