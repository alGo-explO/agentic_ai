# Project 12 — Milestone 2 (Agentic AI Assessment Design Assistant)

This repository implements **Milestone 2 only** from `Project 12_AI_ML.pdf`:
an **agentic assessment design assistant** that:
- accepts assessment design queries,
- analyzes difficulty/performance patterns described in the query (heuristics only; no Milestone 1 ML),
- retrieves pedagogy / assessment guidelines using **Chroma RAG**,
- generates **structured improvement recommendations**,
- produces a lightweight **grounding / citation evaluation report**,
- and runs as a **Streamlit** app (deployable).

## Tech stack (as required)
- **Database (RAG)**: Chroma
- **UI / deployment target**: Streamlit
- **LLM**: Groq (free-tier API key)
- **Agent framework**: LangGraph

## Run locally

1) Create a virtualenv and install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Set the Groq API key:

```bash
export GROQ_API_KEY="YOUR_KEY"
```

3) Start Streamlit:

```bash
streamlit run app.py
```

## Using the app
- Enter an assessment design query (question text, improvement goals, known misconceptions, etc.).
- (Optional) Add your institution’s assessment policy text to the RAG corpus in the **RAG** tab.
- Click **Generate report**.

## Notes
- This code intentionally avoids Milestone 1 (classical ML difficulty prediction).
- The Chroma collection persists locally in `.chroma/` inside the workspace.

## Deploying
- Do **not** commit your Groq key. Set it as a secret on your hosting platform.
  - **Streamlit Community Cloud**: set `GROQ_API_KEY` in the app secrets.
  - Other hosts: set an environment variable named `GROQ_API_KEY`.
- If Streamlit deploy logs show an unexpected Python version (e.g. `python3.14`) and Chroma fails to import with a protobuf error, keep the repo’s `runtime.txt` (set to Python 3.12) so dependency pins remain compatible.

