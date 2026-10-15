# Context rot in a small local agent: does Deep Agents' context engineering help?

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CJosh88/ML/blob/main/experiments/2026-10-deep-agents-context-rot/notebook.ipynb)

## Question

A small local model works through a multi-step data task whose tool outputs overflow its context window. How much do [Deep Agents](https://github.com/langchain-ai/deepagents)' **offloading** and **summarisation** keep it accurate, compared with a plain tool-calling loop? This tests two of the techniques in LangChain's [Context Engineering: How Deep Agents Remember](https://blog.langchain.com/context-engineering-deep-agents-remember) (14 October 2026), on deepagents 0.7.25.

## Dataset

[Our World in Data CO₂ and greenhouse gas emissions](https://github.com/owid/co2-data), licensed **CC BY 4.0**, pinned to commit [`382ee6c`](https://github.com/owid/co2-data/tree/382ee6c662b0ece26e111f263b44c029afad7787).

- **Tool:** `get_country_report(country)` returns one country's 2004–2023 values for 12 metrics (CO₂ by fuel, methane, total GHG, energy use and others), one value per line. Each report is a few thousand tokens. The pool is countries with complete data and at least 10 million people.
- **Questions:** 24 generated questions (seed 42), with gold answers computed in pandas from the same rounded values the agent sees. There are 8 at each level: 1, 3 or 5 country reports needed. Every answer has the form `ANSWER: <country or year> | <number>`, and a number counts as correct within 1%.

## Arms

Everything is held constant except the harness: `qwen3.5:4b` through Ollama on a T4, a 16k-token window, greedy decoding, the same tool, system prompt and questions.

| Arm | What it is |
|---|---|
| 1 Plain loop | `langchain.agents.create_agent` with the tool; no context management |
| 2 Deep Agents, off | Deep Agents harness with offloading and summarisation disabled |
| 3 + Offloading | Tool results over 4,000 characters go to a file; the model sees a preview and reads the file with `read_file`/`grep` |
| 4 + Offloading + summarisation | Plus summarising older messages at 85% of the window (counted with the Qwen tokenizer) |
| 5 + Offloading + early summarisation | The same, but summarising at 50% of the window. Added after the first run, where arm 4's 85% trigger never fired. |
| 6 + Subagent (optional) | Plus a researcher subagent per country that returns only the numbers |

Deep Agents' defaults are tuned for frontier models: offloading starts at about 20k tokens, and summarisation at 170k tokens for models without a context-size profile (all Ollama models). Neither would ever fire at this scale, so the notebook sets both thresholds explicitly.

## Metrics

- Accuracy overall and by level (1/3/5 countries), with Wilson 95% intervals and exact McNemar tests against the plain loop. The headline is **lenient** accuracy: the reply's last line names the gold country or year and the gold number. **Strict** accuracy also needs the `ANSWER:` line and is reported alongside it, because the Deep Agents arms often stated the right answer in prose and dropped that line.
- Whether the summariser actually fired, per arm
- Peak and total tokens sent per question, and the share of runs with a call over the window (Ollama then silently drops part of the conversation)
- Model calls, summariser calls, `read_file`/`grep` use, and wall time per question
- Side-by-side answers and a condensed trace of a failure

## How to run

1. Click **Open in Colab** above, then choose **Runtime → Change runtime type → T4 GPU**.
2. Choose **Runtime → Run all**. The notebook installs Ollama 0.40.4 and the pinned packages, then pulls `qwen3.5:4b` (about 4 GB).
3. `USE_DRIVE = True` is set, so Colab asks to mount Google Drive, and results and traces are cached under `MyDrive/ML/2026-10-deep-agents-context-rot/`. A rerun after a disconnect skips finished questions. Set it to `False` to cache under `outputs/` in the Colab session instead.
4. Optional: set `SMOKE_N = 1` in the Config cell first for a quick end-to-end check (one question per level, separate cache). Set `RUN_SUBAGENT_ARM = True` for arm 6.
5. Expect roughly 1.5 hours on a T4 for the five default arms. If arms 1–4 are already cached on Drive, a rerun only runs arm 5 (about 15–20 minutes). The last cell zips and downloads all results.
6. Commit `figures/`, `questions.csv`, `results.csv`, `summary.csv` and `accuracy_by_level.csv`.

No API keys are needed; everything runs locally.

## Results

TODO: fill after running
