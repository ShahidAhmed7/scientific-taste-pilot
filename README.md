# Scientific Taste pilot: a bag-of-words baseline on SciJudgeBench

## Experiment summary

**Question.** In *AI Can Learn Scientific Taste* (arXiv 2603.14473), a judge model trained with RL picks which of two papers will be cited more, using only titles, abstracts and dates. How much of that accuracy can a model with no reasoning reach, using word counts alone?

**Method.**
- **Input text:** title + abstract.
- **Classifier:** TF-IDF logistic regression on the difference between the two papers' features.
- **Training pairs:** taken from the public SciJudgeBench data; each pair was added in both orders.
- **Overlap removal:** any pair sharing a paper with the test sets was removed first.

**Result (pilot 2, largest training size):**

| | Main test (n = 1,000) | Future-year (n = 904) |
|---|---|---|
| Qwen3-4B-Instruct, untrained (paper v3, Tables 3 and 4) | 58.1 | 64.7 |
| SciJudge-Qwen3-4B (paper v3) | 77.3 | 80.9 |
| Bag-of-words, 222,713 training pairs (this pilot) | 74.6 | 79.0 |
| Gap to SciJudge-4B | 2.7 | 1.9 |

The words that most predict "more cited" include *large*, *theory*, *neural*, *language models*, *llms*, *https*, *github* and *survey* (pilot 1, words-only model). This is a provisional comparison. It has not been reproduced independently, and it does not show that the judge relies on the same cues.

## Contents

| Path | What it is |
|---|---|
| `pilot.ipynb` | The two original scripts, each with the output it printed when it ran. Not re-executed. |
| `requirements.txt` | Package versions of the Python environment the scripts ran in. |
| `artifacts/original_scripts/` | `pilot.py` and `pilot2.py`, byte-for-byte as run. |
| `artifacts/run_logs/` | Each download and run command, with its full output, copied verbatim from the session log. Paths are shortened. |
| `artifacts/checksums.sha256` | SHA-256 of the scripts and the data files they read. |
| `artifacts/MANIFEST.md` | Each requested record, marked available or unavailable. |

The data files are not included because they are public and large. To get them, see "Dataset revision" below.

## 1. Code, paper and dataset revisions

- **Code revision:**
  - The scripts were not under version control when they ran.
  - Their SHA-256 hashes are `b264d44e…c811` (`pilot.py`) and `e37d3aa6…275e` (`pilot2.py`); full hashes are in `artifacts/checksums.sha256`.
  - This repository's first commit packages them unchanged.
- **Paper version:**
  - arXiv 2603.14473v3.
  - The comparison numbers come from Table 3 (main test) and Table 4 (future-year test, 904 pairs). They were checked against the v3 HTML.
- **Dataset revision:**
  - `OpenMOSS-Team/SciJudgeBench`, downloaded from `resolve/main`.
  - The commit hash was **not recorded** at download time.
  - The SHA-256 of each local test file matched the LFS hash of the same file at `main` = `1a4cb0f3410526adaf54ac1b94150bdfe94413d5`. That is `main`'s latest commit, made before the download.
  - The training slice is the first 1,200,000,001 bytes of `train.jsonl` (LFS SHA-256 `b01832f8…a10f`). That slice was **not** hash-checked against the remote file.

## 2. Splits, sample size, seed and overlap removal

- **Test sets:**
  - All of `test.jsonl` (1,000 pairs) and `test_ood_year.jsonl` (904 pairs) were used, with nothing filtered out.
  - Their paper IDs are the `paper_a_arxiv_id` and `paper_b_arxiv_id` fields of those public files.
  - The label balance of `test.jsonl` is 0.500 "A". The balance of the future-year set was not recorded.
- **Pilot 1 training:**
  - Source: `test_extended.jsonl` (8,830 pairs). This is a public **evaluation** split, not the paper's training split.
  - 7,830 pairs were kept after overlap removal.
- **Pilot 2 training:**
  - Source: `train_part.jsonl`, parsed into 222,713 pairs; the incomplete final line was skipped.
  - The overlap filter removed **0** pairs. This is inferred because the printed count was 222,713 both before and after filtering.
  - The pairs were shuffled with Python's `random.seed(0)` and `random.shuffle`. The models used the first 25,000 pairs, the first 100,000, and all 222,713.
  - 222,713 is 30.9% of the 720,341 training pairs the paper reports.
- **Overlap removal:**
  - Exact arXiv-ID match against `test.jsonl` and `test_ood_year.jsonl` only, applied to either paper in a pair.
  - There was no near-duplicate or title matching. `test_ood_iclr.jsonl` was not used.
- **Retained split manifest:** **unavailable.** No list of retained pair or paper IDs was saved.

## 3. Preprocessing, vocabulary, denominators and outputs

- **Text:** `title + " . " + abstract` for each paper.
- **Vectoriser:** scikit-learn `TfidfVectorizer(min_df=3, ngram_range=(1, 2), sublinear_tf=True)`.
  - Pilot 1: `max_features=50000`, fitted on both papers of all 7,830 retained training pairs.
  - Pilot 2: `max_features=200000`, re-fitted for each training size on both papers of the first min(n, 60,000) shuffled training pairs.
  - The vocabulary was never fitted on test data.
- **Features:**
  - Both pilots use the TF-IDF vector of paper A minus that of paper B.
  - Pilot 1's main model also adds the date difference in years, the difference in log abstract length and the difference in title length.
  - Every training pair also appears reversed, with its features negated and its label flipped.
- **Classifier:** `LogisticRegression` with an intercept and no `random_state` set.
  - Pilot 1: default solver; C ∈ {0.3, 1, 3}; `max_iter=3000`. Its words-only model uses C = 1.
  - Pilot 2: `solver="liblinear"`, C = 1, `max_iter=2000`.
- **Metric and denominators:**
  - Accuracy is the share of pairs where the prediction equals `correct_answer`, scored in the given order only.
  - Denominators: 1,000 (main test) and 904 (future-year). By field on the main test: Physics 340, Computer Science 248, Others 208, Mathematics 204.
  - For the simple rules, a tie counts as 0.5.
- **Predictions, fitted models and vocabularies:** **unavailable.** Only the printed accuracies and word lists were kept.

## Caveats

1. **Different metric.** The paper counts a pair as correct only if the prediction is the same in both orders (A, B) and (B, A). The pilot scores one order. Because the model works on the difference between the two papers and was trained on both orders, the two metrics should be close for it, but this was not measured.
2. **Choices made on the test sets.**
   - There was no validation split.
   - Pilot 1 reports three values of C, and pilot 2 reports three training sizes, all on the test sets.
   - The headline numbers come from the largest training size. C = 1 was fixed in pilot 2, not tuned.
3. **Non-random training sample.** The slice is the start of the file, not a random sample. It is 83% Physics, while the main test set is 34% Physics.
4. **Pilot 1 trains on an evaluation split.** Its numbers (72.1 / 76.4) are reported only as a smaller check.
5. **One run per setting.** No repeated seeds or intervals were computed. As a rough guide, the sampling standard error of an accuracy near 75% is about 1.4 points at n = 1,000 and n = 904. That is roughly the same size as the gaps above.
6. **Less training data.** The judge was trained on 720,341 pairs; this baseline saw at most 222,713.

## Reproducing (not part of the checkpoint)

Put the files from `OpenMOSS-Team/SciJudgeBench` at revision `1a4cb0f3` in a working folder: `test.jsonl`, `test_ood_year.jsonl`, `test_extended.jsonl`, and the first 1,200,000,001 bytes of `train.jsonl` saved as `train_part.jsonl`. Then run `python pilot.py` and `python pilot2.py` from that folder. The download commands are in `artifacts/run_logs/`. Pilot 2 needs about 2 minutes of CPU time and several GB of memory. It needs no GPU and no model inference.
