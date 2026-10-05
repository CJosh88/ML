# Can a pretrained model beat XGBoost without training? NVIDIA Kumo Tabular vs gradient-boosted trees, plus a rematch with my 2020 Bank Marketing models

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CJosh88/ML/blob/main/experiments/2026-10-kumo-tabular/notebook.ipynb)

## TL;DR

- **With zero tuning, NVIDIA Kumo Tabular beat default XGBoost and LightGBM on 25 of 25 train/test splits** across five OpenML datasets, by +0.034 ROC-AUC on average. The two other in-context models, Kumo small and TabICLv2, also beat the GBDTs on nearly every split.
- **The advantage is largest with little data and shrinks as data grows.** On adult it was +0.05 ROC-AUC at 100 training rows and +0.01 at 10,000.
- **On Bank Marketing, the zero-tuning model edged out my hand-tuned 2020 models.** Kumo large had the best PR-AUC both with the leaky `duration` feature (0.566 vs 0.542 for tuned XGBoost) and without it (0.373 vs 0.349).
- **Rebuilding the 2020 models exposed a bug in the 2020 notebook.** Its random forest scores look inflated by train/test leakage from an unseeded shuffle combined with pickled models.
- **The cost is speed.** On a 10k-row table Kumo large took about 80 s on a T4 GPU, while XGBoost took 0.3 s on CPU. Kumo small was nearly as accurate at about 11 s.

## Question

With zero tuning, does a pretrained tabular foundation model ([NVIDIA Kumo Tabular](https://huggingface.co/blog/nvidia/kumo-tabular)) beat default XGBoost and LightGBM on small and medium classification tables? And how does it compare with the hand-tuned models from my [2020 Bank Marketing notebook](https://github.com/CJosh88/scriptz/blob/master/Bank%20Marketing%20ML.ipynb)?

## How in-context learning works

A gradient-boosted tree model is **trained**. You give it your training rows, it builds hundreds of trees to fit them, and you then use those trees to predict new rows. Every dataset gets its own freshly fitted model, usually with hyperparameter tuning on top.

Kumo Tabular is **never trained on your data**. Its weights were fixed once, by NVIDIA, after pretraining on tens of millions of *synthetic* tables. The tables were generated from random causal graphs, so the model has seen a huge variety of "features cause a target" patterns, but no real dataset. To make a prediction, you pass it two things in a single forward pass:

1. your labelled training rows (the **context**), and
2. the unlabelled rows you want predictions for (the **queries**).

Inside the transformer, each query row attends to the context rows and their labels and infers the relationship on the fly, much as an LLM picks up a task from worked examples in its prompt. Nothing is fitted, nothing is tuned and no weights change. The "learning" happens inside one forward pass, which is why it's called **in-context learning** (ICL).

| | XGBoost / LightGBM | Kumo Tabular (ICL) |
|---|---|---|
| Uses your training rows | to fit a new model | as context in a forward pass |
| Weights change? | yes, built from scratch per dataset | no, frozen after pretraining |
| Hyperparameter tuning | usually needed | none |
| What it learned beforehand | nothing | priors from ~35–137M synthetic tables |
| Compute | cheap CPU | GPU; cost grows with context size |
| Practical limit | none really | context size (here 10k rows per ensemble member) |

So this isn't zero-shot. The model needs labelled examples, and in every comparison here it gets exactly the same training rows as XGBoost and LightGBM. It's zero-*training*. The pretrained prior is what lets it do well from very few rows.

## Setup

| Part | What | Models |
|---|---|---|
| A | 5 OpenML-CC18 binary datasets × 5 stratified 75/25 splits (train capped at 10k rows, test at 5k) | XGBoost (default), LightGBM (default), TabICLv2, Kumo Tabular small (28M params) and large (215M) |
| Learning curve | adult and Bank Marketing (no `duration`), 100 to 10k training rows, 3 seeds | XGBoost, LightGBM, Kumo Tabular large |
| B | UCI Bank Marketing on the 2020 split (80/20, `random_state=20`) and 2020 encoding, with and without the leaky `duration` feature | The four 2020 models, rebuilt with the best hyperparameters the 2020 search found, plus all Part A models |
| Controls | Leak check for the 2020 numbers; shuffled-label check for the in-context models | as above |

- **No tuning anywhere** except the 2020 models, which keep their 2020 tuned settings. The GBDTs use library defaults with native categorical handling.
- **Kumo settings:** 8 ensemble members, fp16, at most 10k context rows per member. Larger training sets get a different random 10k subsample per member.
- **Metrics:** ROC-AUC, PR-AUC (average precision) and log loss. Threshold-0.5 accuracy, precision, recall and F1 are included to line up with the 2020 table. Timings cover fit plus predict.
- **Hardware:** Colab T4 GPU for the in-context models, Colab's 2 vCPUs for the trees.
- **Reproducibility:** a full rerun reproduced every Part A ROC-AUC to 4 decimal places.

Kumo Tabular and TabICLv2 come from NVIDIA's [`structured-data-models`](https://github.com/NVIDIA/structured-data-models) library (alpha), pinned to commit `ce95710`. It can't be installed from PyPI, because the PyPI name is an empty `0.0.0a0` placeholder, even though the Hugging Face model card says `pip install structured-data-models`. The code snippet in NVIDIA's blog post also has typos, so the notebook follows the repo's own examples instead.

## Results

See full writeup on my Quarto page: https://cjosh88.github.io/ai-portfolio/experiments/2026-10-kumo-tabular/

## Data

| Dataset | Source | Rows | Licence |
|---|---|---|---|
| credit-g | [OpenML 31](https://www.openml.org/d/31) | 1,000 | Public |
| diabetes | [OpenML 37](https://www.openml.org/d/37) | 768 | Public |
| kc1 | [OpenML 1067](https://www.openml.org/d/1067) | 2,109 | Public |
| phoneme | [OpenML 1489](https://www.openml.org/d/1489) | 5,404 | Public |
| adult | [OpenML 1590](https://www.openml.org/d/1590) | 48,842 | Public |
| Bank Marketing (`bank-full.csv`) | [UCI 222](https://archive.ics.uci.edu/dataset/222/bank+marketing) | 45,211 | CC BY 4.0 |

Bank Marketing citation: S. Moro, P. Cortez and P. Rita (2014), *A Data-Driven Approach to Predict the Success of Bank Telemarketing*, Decision Support Systems 62:22–31.

## How to run

1. Click the Colab badge and pick **Runtime → Change runtime type → T4 GPU**.
2. Run all cells. The first cell installs the pinned packages. Kumo weights (about 1 GB for small and large) download from Hugging Face on first use; no token is needed.
3. Optional: set `QUICK = True` in the Config cell for a few-minute pipeline check, and `USE_DRIVE = True` to keep the result cache across Colab sessions.

Locally: Python 3.11+, a CUDA GPU, `pip install -r requirements.txt`, then run `notebook.ipynb`.

Results are cached as they're computed, so a disconnect only loses the run in progress.

## Files

- [`notebook.ipynb`](notebook.ipynb): the full experiment.
- [`results_openml.csv`](results_openml.csv), [`summary_openml.csv`](summary_openml.csv): Part A, per split and averaged.
- [`results_learning_curve.csv`](results_learning_curve.csv): the learning curve.
- [`results_bank.csv`](results_bank.csv): Part B, all metrics.
- [`summary_shuffled_labels.csv`](summary_shuffled_labels.csv): shuffled-label control, mean and max over 3 permutations. It was rebuilt from the notebook's printed output, because the runtime disconnected before the raw CSV was saved.
- [`figures/`](figures): all charts.
