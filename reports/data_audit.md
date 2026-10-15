# TestPilot AI: Comprehensive Dataset Audit Report

## 1. Executive Summary

- **Dataset File**: `data\benchmark_tcp_ci.csv`
- **File Size**: 0.986 MB
- **Total Rows (Test Executions)**: 8,958
- **Total Columns**: 21
- **Duplicate Rows**: 0
- **Duplicate (Project, Build, Test) Tuples**: 0
- **Total Software Projects**: 4 (apache-commons-lang, google-guava, square-retrofit, alibaba-fastjson)
- **Total CI Builds**: 239

## 2. Class Distribution and Failure Profile

Software testing datasets are inherently characterized by extreme class imbalance:

- **Passed Tests (Verdict = 0)**: 8,136 (90.82%)
- **Failed Tests (Verdict = 1)**: 822 (9.18%)
- **Builds Containing at Least One Failure**: 135 / 239 (56.49%)

> [!IMPORTANT]
> The observed test failure rate is 9.18%. Standard accuracy is an invalid evaluation metric under this class imbalance; evaluation must emphasize PR-AUC, Recall, APFD, and budget-constrained failure discovery.

## 3. Execution Duration Statistics

| Metric | Duration (seconds) |
|---|---|
| Minimum | 0.554 s |
| Median (P50) | 7.133 s |
| Mean | 11.359 s |
| 75th Percentile | 12.68 s |
| 90th Percentile | 23.616 s |
| 95th Percentile | 36.637 s |
| Maximum | 117.268 s |
| Cumulative Regression Testing Time | 28.26 hours |

## 4. Data Quality and Missingness

| Column | Missing Count | Missing % | Data Type |
|---|---|---|---|
| `project_id` | 0 | 0.0% | `object` |
| `build_id` | 0 | 0.0% | `int64` |
| `test_id` | 0 | 0.0% | `object` |
| `historical_fail_rate` | 0 | 0.0% | `float64` |
| `failures_last_5` | 0 | 0.0% | `int64` |
| `failures_last_10` | 0 | 0.0% | `int64` |
| `total_historical_runs` | 0 | 0.0% | `int64` |
| `builds_since_last_fail` | 0 | 0.0% | `int64` |
| `avg_recent_duration` | 0 | 0.0% | `float64` |
| `test_age_builds` | 0 | 0.0% | `int64` |
| `test_loc` | 0 | 0.0% | `int64` |
| `changed_files_count` | 0 | 0.0% | `int64` |
| `code_churn_additions` | 0 | 0.0% | `int64` |
| `code_churn_deletions` | 0 | 0.0% | `int64` |
| `modified_test_file` | 0 | 0.0% | `int64` |
| `file_test_distance` | 0 | 0.0% | `float64` |
| `cov_class_count` | 0 | 0.0% | `int64` |
| `cov_method_count` | 0 | 0.0% | `int64` |
| `w_sum_cov_c_faults` | 0 | 0.0% | `float64` |
| `duration` | 0 | 0.0% | `float64` |
| `verdict` | 0 | 0.0% | `int64` |

## 5. Leakage Risk Analysis and Protocol

To satisfy rigorous software engineering empirical research standards, features are partitioned strictly by execution lifecycle:

### A. Pre-Execution Safe Features (Permitted in ML Model)
- **Historical Execution Features (REC)**: `historical_fail_rate`, `failures_last_5`, `failures_last_10`, `total_historical_runs`, `builds_since_last_fail`, `avg_recent_duration`.
- **Code Churn Features (CHG)**: `changed_files_count`, `code_churn_additions`, `code_churn_deletions`, `modified_test_file`, `file_test_distance`.
- **Coverage and Structure (COV/TES)**: `test_age_builds`, `test_loc`, `cov_class_count`, `cov_method_count`, `w_sum_cov_c_faults`.

### B. Post-Execution Variables (Isolated from Training Features)
- `verdict`: Ground-truth binary outcome observed only after test completes.
- `duration`: Current runtime observed only after test completes. The prioritisation scheduler uses `avg_recent_duration`.

### C. Identifiers (Kept for Joins and Grouping)
- `project_id`, `build_id`, `test_id` are excluded from input vectors $X$ to avoid spurious project memorization.

## 6. Discovered Data Dictionary

| Column Name | Category | Type | Leakage-Safe? | Description |
|---|---|---|---|---|
| `project_id` | Identifier | `object` | Yes (Pre-Exec) | Software project or repository identifier |
| `build_id` | Identifier | `int64` | Yes (Pre-Exec) | Monotonically increasing CI build sequence number |
| `test_id` | Identifier | `object` | Yes (Pre-Exec) | Test case class or method identifier |
| `historical_fail_rate` | REC | `float64` | Yes (Pre-Exec) | Historical failure frequency across past builds |
| `failures_last_5` | REC | `int64` | Yes (Pre-Exec) | Number of test failures in past 5 builds |
| `failures_last_10` | REC | `int64` | Yes (Pre-Exec) | Number of test failures in past 10 builds |
| `total_historical_runs` | REC | `int64` | Yes (Pre-Exec) | Total historical executions of this test case |
| `builds_since_last_fail` | REC | `int64` | Yes (Pre-Exec) | Builds elapsed since previous observed failure |
| `avg_recent_duration` | REC | `float64` | Yes (Pre-Exec) | Estimated duration based on moving average of recent runs (s) |
| `test_age_builds` | TES | `int64` | Yes (Pre-Exec) | Number of builds since test was added to test suite |
| `test_loc` | TES | `int64` | Yes (Pre-Exec) | Source lines of code in test case |
| `changed_files_count` | CHG | `int64` | Yes (Pre-Exec) | Count of modified files in the triggering commit |
| `code_churn_additions` | CHG | `int64` | Yes (Pre-Exec) | Lines of code added in triggering commit |
| `code_churn_deletions` | CHG | `int64` | Yes (Pre-Exec) | Lines of code deleted in triggering commit |
| `modified_test_file` | CHG | `int64` | Yes (Pre-Exec) | Binary flag indicating if test file itself was edited |
| `file_test_distance` | CHG | `float64` | Yes (Pre-Exec) | Syntactic/package distance between changed code and test |
| `cov_class_count` | COV | `int64` | Yes (Pre-Exec) | Classes covered by this test case |
| `cov_method_count` | COV | `int64` | Yes (Pre-Exec) | Methods covered by this test case |
| `w_sum_cov_c_faults` | COV | `float64` | Yes (Pre-Exec) | Weighted coverage-to-historical-fault correlation score |
| `duration` | Post-Execution | `float64` | **NO (Post-Exec Target)** | Actual test execution duration in seconds (Target Cost) |
| `verdict` | Post-Execution | `int64` | **NO (Post-Exec Target)** | Actual test outcome: 0 = Passed, 1 = Failed (Target Label) |
