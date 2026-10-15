# TestPilot AI: Dataset Placement and Provenance Guide

## 1. Primary Public Dataset: TCP-CI Benchmark

- **Source Zenodo Archive**: [Zenodo Record 6415365](https://zenodo.org/records/6415365)
- **DOI**: `10.5281/zenodo.6415365`
- **Associated Academic Publication**: 
  - *Ahmadreza Saboor Yaraghi, Mojtaba Bagherzadeh, Nafiseh Kahani, Lionel Briand*, "Scalable and Accurate Test Case Prioritization in Continuous Integration Contexts", IEEE Transactions on Software Engineering (TSE), 2023. arXiv: [2109.13168](https://arxiv.org/abs/2109.13168).
- **Official Replication Package**: [GitHub: Ahmadreza-SY/TCP-CI](https://github.com/Ahmadreza-SY/TCP-CI)

### Overview
The TCP-CI dataset consists of 25 open-source Java projects containing ~21,500 Continuous Integration (CI) builds and ~2,500 failed builds. It represents real-world regression testing executions where builds required substantial testing times and exhibited non-trivial failure patterns.

---

## 2. Dataset Placement Instructions

If downloading directly from CERN Zenodo is restricted or times out:

1. Download the archive `TCP-CI-main-dataset.tar.gz` (237 MB) directly from [https://zenodo.org/records/6415365](https://zenodo.org/records/6415365).
2. Place the compressed file into:
   ```
   data/raw/TCP-CI-main-dataset.tar.gz
   ```
3. Run the automated extractor and curator script:
   ```bash
   python scripts/download_dataset.py --extract
   ```
4. If working with individual subject CSV files (e.g., `commons-lang`, `retrofit`, `fastjson`), place the CSV in:
   ```
   data/benchmark_tcp_ci.csv
   ```

---

## 3. Schema & Feature Definition

Each record represents a test execution within a specific CI build:

| Column Name | Category | Type | Description | Leakage Risk? |
|-------------|----------|------|-------------|---------------|
| `project_id` | Identifier | String | Open-source software subject identifier | Grouping only |
| `build_id` | Identifier | Integer | Chronologically ordered build sequence number | Splitting key |
| `test_id` | Identifier | String | Qualified test class or method name | Display only |
| `historical_fail_rate` | REC (History) | Float [0, 1] | Proportion of past runs where this test failed | Pre-execution SAFE |
| `failures_last_5` | REC (History) | Integer | Number of failures observed in previous 5 builds | Pre-execution SAFE |
| `failures_last_10` | REC (History) | Integer | Number of failures observed in previous 10 builds | Pre-execution SAFE |
| `total_historical_runs`| REC (History) | Integer | Cumulative previous executions of this test | Pre-execution SAFE |
| `builds_since_last_fail`| REC (History)| Integer | Number of builds elapsed since last failure | Pre-execution SAFE |
| `avg_recent_duration` | REC (History) | Float | Average execution duration over recent builds (s) | Pre-execution SAFE |
| `test_age_builds` | TES (Structural)| Integer | Number of builds since this test was introduced | Pre-execution SAFE |
| `test_loc` | TES (Structural)| Integer | Source code lines of code in test | Pre-execution SAFE |
| `changed_files_count` | CHG (Code Churn)| Integer | Total files modified in the triggering commit | Pre-execution SAFE |
| `code_churn_additions` | CHG (Code Churn)| Integer | Lines of code added in triggering commit | Pre-execution SAFE |
| `code_churn_deletions` | CHG (Code Churn)| Integer | Lines of code deleted in triggering commit | Pre-execution SAFE |
| `modified_test_file` | CHG (Code Churn)| Binary {0, 1} | Whether the test file itself was modified | Pre-execution SAFE |
| `file_test_distance` | CHG (Proximity) | Float [0, 1] | Minimum path/package distance between changed files and test | Pre-execution SAFE |
| `cov_class_count` | COV (Coverage) | Integer | Classes covered by this test case | Pre-execution SAFE |
| `cov_method_count` | COV (Coverage) | Integer | Methods covered by this test case | Pre-execution SAFE |
| `w_sum_cov_c_faults` | COV (Fault History)| Float | Weighted coverage-to-fault historical association | Pre-execution SAFE |
| `duration` | Post-Execution | Float | Actual execution duration of test in seconds | Post-execution only (Evaluation / Cost) |
| `verdict` | Post-Execution | Binary {0, 1} | Test outcome: 0 = Passed, 1 = Failed | Target Ground Truth |

---

## 4. Leakage Prevention Protocol
1. **Target Leakage**: `verdict` is NEVER included as an input feature during model training or inference.
2. **Duration Leakage**: `duration` (the current execution duration) is observed only AFTER execution. Pre-execution models utilize `avg_recent_duration` (estimated historical runtime) for scheduling.
3. **Cross-Build Leakage**: Evaluated on strictly chronological splits. Builds $B_1 \dots B_k$ are used for training, builds $B_{k+1} \dots B_m$ for validation, and builds $B_{m+1} \dots B_n$ are held out as unseen future builds.
