# Record manifest

This file lists every record the checkpoint asks for. Each one is either **available** and points to its file, or **unavailable** and was not recreated.

| Requested record | Status | Where, or why not |
|---|---|---|
| Code revision | Available as file hashes; there was no version control at run time | `checksums.sha256`, `original_scripts/` |
| Paper version | Available | arXiv 2603.14473v3, Tables 3 and 4 (`../README.md`, section 1) |
| Dataset revision | Commit not recorded at download; test files matched by hash to `main` = `1a4cb0f3` | `checksums.sha256`, `../README.md` section 1 |
| Training-slice revision | Partly available: the byte range is known; the slice was not hash-checked against the remote file | `run_logs/04_download_train_slice.log` |
| Test paper IDs | Available in the public test files; no separate list was saved | `test.jsonl` and `test_ood_year.jsonl`, fields `paper_a_arxiv_id` and `paper_b_arxiv_id` |
| Retained training split manifest | **Unavailable** | Not saved |
| Training-pair counts | Available | Pilot 1: 7,830 of 8,830. Pilot 2: 25,000, 100,000 and 222,713 (`run_logs/03_pilot1.log`, `run_logs/05_pilot2.log`) |
| Seed | Available | Pilot 2: Python `random.seed(0)` for the shuffle. No `random_state` was set for scikit-learn. Pilot 1 has no shuffle. |
| Overlap-removal method | Available | Exact arXiv-ID match against both test sets (scripts, lines that build `held`) |
| Where preprocessing and vocabulary were fitted | Available | Training pairs only (`../README.md`, section 3) |
| Evaluation denominators | Available | 1,000 and 904; by field 340, 248, 208 and 204 |
| Printed outputs | Available | `run_logs/`, `../pilot.ipynb` |
| Per-pair predictions | **Unavailable** | Not saved |
| Fitted models and vocabularies | **Unavailable** | Not saved |
| Label balance of the future-year set | **Unavailable** | Not printed |
| Swap-consistent accuracy of the baseline | **Unavailable** | Not computed |
