# Keyword biasing on clinical speech: Granite Speech 5.0 vs Whisper on PriMock57 and AfriSpeech

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CJosh88/ML/blob/main/experiments/2026-10-granite-speech-clinical-kwb/notebook.ipynb)

## Question

On real clinical speech that none of these models was trained on, does [Granite Speech 5.0](https://huggingface.co/blog/ibm-granite/granite-speech-5)'s keyword biasing get medical terms right more often than plain transcription and Whisper's prompting, and how often does it write listed terms that nobody said?

IBM reports keyword-biasing F1 and Open ASR Leaderboard WER for granite-speech-5.0-2b (released 20 October 2026), but AMI and Earnings-22, the obvious test sets for conversations and jargon, are both in its training data. This experiment uses two clinical datasets that are not.

## Datasets

- **[PriMock57](https://github.com/babylonhealth/primock57)** (Babylon Health), licensed **CC BY 4.0**, pinned to commit [`cd2ac70`](https://github.com/babylonhealth/primock57/tree/cd2ac707ad03cb4d2531f4ec6b90c659bf4357c5). 57 mock primary-care consultations (8.6 hours) with doctor and patient on separate 16 kHz channels and hand-timed utterance transcripts. Utterances containing `<UNIN/>` (unintelligible) are dropped, `<UNSURE>` tags are removed, and only utterances of 1–30 s are kept.
- **[AfriSpeech-200](https://huggingface.co/datasets/intronhealth/afrispeech-200)** (Intron Health), licensed **CC BY-NC-SA 4.0**, gated. The clinical-domain test set: read clinical sentences from many African accents. The notebook takes 30 clips from each accent with at least 30 clinical test clips, up to 40 accents (1,200 clips; 42 accents qualify). Most of those accents have only 2–3 test speakers, so per-accent results mostly reflect individual voices.

Neither dataset appears in Granite Speech 5.0's published training data, and both postdate Whisper's.

## Conditions

Within each model, only the keyword list changes.

| Condition | Model | Keywords |
|---|---|---|
| A | [Whisper-small](https://huggingface.co/openai/whisper-small) | none |
| B | Whisper-small | the list as a Whisper prompt (`prompt_ids`, "Keywords: …") |
| C | [Granite Speech 5.0 Turbo CTC](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc) (470M) | none (a CTC model takes no prompt) |
| D | [Granite Speech 5.0 2B](https://huggingface.co/ibm-granite/granite-speech-5.0-2b), prompt `transcribe the speech to text.` | none |
| E | Granite Speech 5.0 2B, prompt `transcribe the speech to text. Keywords: …` | the list |

**Keyword lists.** Each PriMock57 consultation, and each block of 5 AfriSpeech clips, gets one list:
- **Targets:** the group's own rare terms, meaning words of at least 4 letters with a [wordfreq](https://pypi.org/project/wordfreq/) Zipf frequency below 3.0 (about once per million words), such as *fexofenadine*, *co-codamol* or *gastroenteritis*.
- **Distractors:** an equal number of rare terms from other groups that nobody in this group said.

Every condition is scored against the same lists, including those that never see them.

## Metrics

- **WER** after Whisper's English text normaliser (lower case, no punctuation, fillers removed, numbers and spellings standardised)
- **B-WER and U-WER** ([Le et al., 2021](https://arxiv.org/abs/2104.02194)): WER on listed words and on all other words, to show whether biasing helps the terms without hurting the rest
- **Target recall:** the share of target-term occurrences that appear in the transcript
- **Distractor insertions per 100 utterances:** listed terms the model wrote though nobody in the group said them
- **RTFx** (seconds of audio per second of processing) and peak GPU memory on a T4
- Doctor vs patient channel (PriMock57) and WER by accent (AfriSpeech)
- 95% intervals from a bootstrap that resamples whole consultations or speakers, plus paired differences for B vs A, E vs D, D vs A, C vs A and E vs B
- Worked examples: keywords fixing a term, keywords producing a hallucinated distractor, and terms every condition missed

## How to run

1. Request access to [AfriSpeech-200](https://huggingface.co/datasets/intronhealth/afrispeech-200) (approval is manual), then add a Hugging Face token as a Colab secret named `HF_TOKEN` and allow the notebook to read it. Without it the notebook runs on PriMock57 only.
2. Click **Open in Colab** above, then choose **Runtime → Change runtime type → T4 GPU**.
3. Optional: set `SMOKE = True` in the Config cell for a 15-minute end-to-end check (3 consultations, 2 accents, separate cache).
4. Choose **Runtime → Run all**. The notebook downloads PriMock57 (about 1.7 GB) and only the AfriSpeech archives it needs.
5. `USE_DRIVE = True` is set, so Colab asks to mount Google Drive and caches transcripts under `MyDrive/ML/2026-10-granite-speech-clinical-kwb/`. A rerun after a disconnect skips finished utterances. Set it to `False` to cache under `outputs/` in the session instead.
6. A sanity-check cell transcribes 8 utterances with each model before the full run. Granite is published in bfloat16, which a T4 doesn't support natively, so it runs in float16. If its output there is empty or garbled, set `GRANITE_DTYPE = "float32"` and rerun.
7. Expect roughly 1.5–2.5 hours on a T4 for all five conditions. The last cell zips and downloads the results.
8. Commit `figures/`, `summary.csv`, `differences.csv` and `by_subset.csv`. `results.csv` holds per-utterance transcripts. Its AfriSpeech rows are derived from CC BY-NC-SA data, so commit it only with that licence noted, or drop those rows. Never commit the audio.

No API keys are needed apart from the Hugging Face token for AfriSpeech.

## Results

TODO: fill after running
