# Agentic-AI — LangGraph Workflows

A small collection of example workflows built on top of a `langgraph`-style orchestration layer and several LangChain provider integrations. The repo demonstrates sequential, parallel, conditional, iterative, and human-in-the-loop workflows using modern LLM providers and retrieval tools.

## Highlights
- Example workflows: sequential, parallel, conditional (RAG), iterative, human-in-the-loop.
- Integrations with LangChain provider packages (Mistral, Groq, Tavily, etc.).
- A Streamlit front-end for the conditional (RAG) assistant.

## Repository structure

- `01-LangGraph/01-Workflows/`
  - `01-sequental-workflow/` — simple editorial → script → translate flow
  - `02-parallel-workflow/` — parallel content-safety checks
  - `03-conditional-workflow/` — RAG-based College Assistant (includes Streamlit UI)
  - `04-iterative -workflow/` — iterate + review flow with tool-augmented LLMs
  - `05-human-in-the-loop/` — human approval interrupt example

## Requirements
- Python 3.10+ recommended
- See `requirements.txt` for the full list of Python packages required.

## Quick setup

1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Make the local `langgraph` package importable

If `langgraph` is a local package in this workspace, either install it editable or add the project root to `PYTHONPATH`:

```bash
# from repo root
pip install -e 01-LangGraph
# OR
export PYTHONPATH="$PYTHONPATH:$(pwd)/01-LangGraph"
```

4. Create a `.env` file with provider credentials

Add any API keys required by the provider integrations used by the examples. Typical variables you may need to set (provider-specific):

- `MISTRAL_API_KEY`
- provider-specific keys for Groq/Tavily, etc.

## Running the examples

- Sequential workflow (CLI):

```bash
python 01-LangGraph/01-Workflows/01-sequental-workflow/sequental-workflow.py
```

- Parallel workflow (CLI):

```bash
python 01-LangGraph/01-Workflows/02-parallel-workflow/parallel-workflow.py
```

- Conditional (RAG) assistant (CLI):

```bash
python 01-LangGraph/01-Workflows/03-conditional-workflow/conditional-workflow.py
```

- Conditional assistant (Streamlit UI):

```bash
streamlit run 01-LangGraph/01-Workflows/03-conditional-workflow/main.py
```

- Iterative workflow (CLI):

```bash
python "01-LangGraph/01-Workflows/04-iterative -workflow/01-iterative-workflow.py"
```

- Human-in-the-loop workflow (CLI):

```bash
python 01-LangGraph/01-Workflows/05-human-in-the-loop/01-human-in-the-loop.py
```

## Notes & troubleshooting

- The examples import third-party LangChain provider integrations such as `langchain-mistralai`, `langchain-groq`, and `langchain-tavily`. If you encounter import errors, ensure the provider packages are installed (see `requirements.txt`) and any provider-specific native libraries are available.
- FAISS requires native dependencies; if `faiss-cpu` installation fails, follow the platform-specific guidance in the FAISS project docs or switch to a CPU-compatible wheel.
- If the `langgraph` package is not installed as a standard pip package, ensure it is importable via `pip install -e 01-LangGraph` or by setting `PYTHONPATH` as shown above.

## Contributing

Contributions welcome — open an issue or submit a PR with improvements, additional examples, or packaging changes to make `langgraph` installable via pip.

## License

This repository does not include a license file. Add a license if you intend to publish or share the code.
