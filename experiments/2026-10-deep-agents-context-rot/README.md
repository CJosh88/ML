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

## Part 2: summarisation

In Part 1, offloading kept the context so small that summarisation barely fired. Part 2 isolates summarisation with a task where the context grows from **many small results** that are never offloaded:

- **Tool:** `get_series(country, metric)` returns one country's 2004–2023 values for one metric (about 370 tokens).
- **Questions:** 18 questions (6 per level) on 6, 30 or 50 countries, so about 2k, 11k or 18k tokens of lookups: below both triggers, between them, and over the 16k window.

| Arm | What it is |
|---|---|
| S1 No summary | Deep Agents harness, summarisation off |
| S2 Summary at 85% | Automatic summarisation at 13.9k tokens |
| S3 Summary at 50% | Automatic summarisation at 8.2k tokens |
| S4 85% + keep-numbers prompt | Deep Agents' default summary prompt plus an instruction to list every looked-up value verbatim |
| S5 `compact_conversation` tool | No automatic trigger; the model decides when to compact (allowed from 20 messages), with a short system-prompt nudge |

As well as accuracy, Part 2 checks every summary's numbers against the data: for each summarised country, whether the summary kept the needed value, stated a different number, or dropped it.

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
5. Expect roughly 1.5 hours on a T4 for Part 1's five default arms, and roughly 2 hours for Part 2, which has many more model calls per question. Cached runs are skipped, so if Part 1 is already on Drive, a rerun only runs Part 2. `RUN_PART1` and `RUN_PART2` in the Config cell switch either part off. The last cell zips and downloads all results.
6. Commit `figures/` and the CSVs: `questions.csv`, `results.csv`, `summary.csv`, `accuracy_by_level.csv`, plus Part 2's `questions_part2.csv`, `results_part2.csv`, `summary_part2.csv`, `accuracy_by_level_part2.csv` and `summary_fidelity.csv`.

No API keys are needed; everything runs locally.

## Results

TODO: fill after running
