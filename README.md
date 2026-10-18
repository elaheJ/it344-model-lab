# AI Model Comparison using Artificial Analysis

Compare language models for your project by cost, speed, response time, and quality. Find alternatives similar to a model you already use.

**[Open Model Lab](https://it344-model-lab.streamlit.app/)** · **[Open Model Lab · Agent](https://it344-model-agent.streamlit.app/)**

Both hosted apps include model data and similarity search. Students need no Artificial Analysis API key for those links.

## Run in Google Colab

**[Colab · Compare (no agent)](https://colab.research.google.com/github/elaheJ/it344-model-lab/blob/main/colab/Model_Lab_Compare.ipynb)** · **[Colab · Agent](https://colab.research.google.com/github/elaheJ/it344-model-lab/blob/main/colab/Model_Lab_Agent.ipynb)**

Save a copy in Drive. In Colab’s **Secrets** panel (key icon), add **AA_API_KEY** and enable **Notebook access**. Then choose **Runtime → Run all**; the notebooks read the key from Secrets automatically. Both notebooks fetch fresh data and open a Gradio app with ranking, composite metrics, similarity search, and downloads. The agent runs Qwen in your runtime without a separate LLM key; choose a T4 GPU before running it. Keep Colab connected while using the app. The temporary app link serves the snapshot and analysis; anyone with the link can open it.

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
- `colab/` contains the two notebooks; `model_compare/colab_ui.py` is their shared Gradio interface.
- Other files in `model_compare/` load data or run the local model. `cloud_agent_app.py` is the launcher used by the hosted agent.

**Using the apps:** choose **Rank models** for single or combined metrics, or **Find similar models** to select a reference model and features. Data is a dated snapshot; use your own project examples to evaluate the models you shortlist.

Tests, maintenance scripts, hosting/key setup, and detailed documentation live in the **[developer repository](https://github.com/elaheJ/it344-model-lab-dev)**.
