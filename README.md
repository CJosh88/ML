# ML

Small, reproducible machine-learning experiments: new libraries and models tested against well-known baselines on open datasets. Each experiment is a self-contained Jupyter notebook that runs on Google Colab's free GPU.

## Experiments

| Folder | Question | Open |
|---|---|---|
| [`experiments/2026-10-kumo-tabular`](experiments/2026-10-kumo-tabular) | Does zero-tuning NVIDIA Kumo Tabular beat default XGBoost/LightGBM on OpenML tables, and how does it compare with my hand-tuned 2020 Bank Marketing models? **Yes: 25/25 splits, biggest gains on small data, best PR-AUC on Bank Marketing.** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CJosh88/ML/blob/main/experiments/2026-10-kumo-tabular/notebook.ipynb) |
| [`experiments/2026-10-warehouse-hazards-d1-vlm`](experiments/2026-10-warehouse-hazards-d1-vlm) | Spotting warehouse safety hazards in images: LiquidAI d1 decision models vs generative VLMs (including d1-3B's own base), a fine-tuned 0.8B VLM and Claude Opus 5.5. Accuracy, missed hazards, false alarms and cost per 1,000 images. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CJosh88/ML/blob/main/experiments/2026-10-warehouse-hazards-d1-vlm/notebook.ipynb) |
| [`experiments/2026-10-deep-agents-context-rot`](experiments/2026-10-deep-agents-context-rot) | When a small local agent's tool outputs overflow its 16k window, how much do Deep Agents' offloading and summarisation keep it accurate, compared with a plain tool-calling loop? qwen3.5:4b on Our World in Data CO₂ reports. **Overflow silently broke the plain loop; offloading recovered about half of those questions; summarisation helped only once three defaults were changed, and dropped 16–34% of the numbers it summarised.** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CJosh88/ML/blob/main/experiments/2026-10-deep-agents-context-rot/notebook.ipynb) |
| [`experiments/2026-10-granite-speech-clinical-kwb`](experiments/2026-10-granite-speech-clinical-kwb) | On clinical speech none of the models was trained on (PriMock57 consultations, AfriSpeech-200 accented clinical English), does Granite Speech 5.0's keyword biasing get medical terms right more often than plain transcription and Whisper prompting, and how often does it write listed terms nobody said? | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CJosh88/ML/blob/main/experiments/2026-10-granite-speech-clinical-kwb/notebook.ipynb) |

## Layout

```
experiments/
  YYYY-MM-<slug>/
    notebook.ipynb     # runnable top to bottom
    requirements.txt   # pinned versions
    README.md          # question, dataset, how to run, results
    figures/           # charts saved by the notebook
```
