# Model Lab for IT344

Compare language models for your project by cost, speed, response time, and quality. Find alternatives similar to a model you already use.

**[Open Model Lab](https://it344-model-lab.streamlit.app/)** · **[Open Model Lab · Agent](https://it344-model-agent.streamlit.app/)**

Both apps include model data and similarity search. Students need no Artificial Analysis API key.

## Run on your computer

Download this repository with **Code → Download ZIP**, unzip it, and open a terminal in that folder. Use Python 3.12:

```bash
python -m venv .venv
```

Activate it: `source .venv/bin/activate` on macOS/Linux, or `.venv\Scripts\Activate.ps1` in Windows PowerShell. Use `python3` instead of `python` above if needed.

```bash
python -m pip install -r requirements.txt
python -m streamlit run python_app.py
```

For the agent app, run `python -m streamlit run agent_app.py`. Choose **Scripted rehearsal (no LLM)** to explore its workflow without a download. To run actual local AI, follow **Controller settings → Local model setup** in the app; it downloads roughly 3 GB and needs no LLM API key.

## Explore the code

- `python_app.py` and `agent_app.py` start the two apps.
- `model_compare/core.py` contains the ten notebook functions, ranking, cost, and similarity calculations.
- `model_compare/ui.py` contains the interface; `model_compare/agent.py` checks the agent's work.
- Other files in `model_compare/` load data or run the local model. `cloud_agent_app.py` is the launcher used by the hosted agent.

**Using the apps:** choose **Rank models** for single or combined metrics, or **Find similar models** to select a reference model and features. Data is a dated snapshot; use your own project examples to evaluate the models you shortlist.

Tests, maintenance scripts, hosting/key setup, and detailed documentation live in the **[developer repository](https://github.com/elaheJ/it344-model-lab-dev)**.
