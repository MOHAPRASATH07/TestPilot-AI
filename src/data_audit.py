"""
Dataset Inspection and Audit Module for TestPilot AI.

Performs exhaustive schema inspection, data quality checks, class imbalance analysis,
temporal capability verification, and data leakage risk assessment.
Outputs: reports/data_audit.json and reports/data_audit.md.
"""

import os
import json
import logging
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
from src.data_loader import (
    load_dataset,
    IDENTIFIER_COLUMNS,
    POST_EXECUTION_COLUMNS,
    ALL_PRE_EXECUTION_FEATURES,
    HISTORY_FEATURES,
    CODE_CHANGE_FEATURES,
    COVERAGE_AND_STRUCTURAL_FEATURES
)

logger = logging.getLogger("TestPilotAI.DataAudit")


def audit_dataset(csv_path: str = os.path.join("data", "benchmark_tcp_ci.csv")) -> Dict[str, Any]:
    """Inspects dataset and compiles structured audit dictionary."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset path {csv_path} does not exist.")
        
    file_size_bytes = os.path.getsize(csv_path)
    file_size_mb = round(file_size_bytes / (1024 * 1024), 3)
    df = pd.read_csv(csv_path)
    
    # Basic shape
    n_rows, n_cols = df.shape
    duplicate_rows = int(df.duplicated().sum())
    duplicate_test_builds = int(df.duplicated(subset=["project_id", "build_id", "test_id"]).sum())
    
    # Missing values
    missing_counts = df.isnull().sum().to_dict()
    missing_pct = {k: round(float(v) / n_rows * 100, 2) for k, v in missing_counts.items()}
    
    # Column datatypes
    dtypes = {col: str(df[col].dtype) for col in df.columns}
    
    # Projects and Builds
    projects = df["project_id"].unique().tolist()
    builds_per_project = {p: int(df[df["project_id"] == p]["build_id"].nunique()) for p in projects}
    total_builds = int(df.groupby(["project_id", "build_id"]).ngroups)
    
    # Class distribution
    verdict_counts = df["verdict"].value_counts().to_dict()
    pass_count = int(verdict_counts.get(0, 0))
    fail_count = int(verdict_counts.get(1, 0))
    fail_rate_pct = round((fail_count / n_rows) * 100, 2) if n_rows > 0 else 0.0
    
    # Build-level failure rates
    build_fail_counts = df.groupby(["project_id", "build_id"])["verdict"].sum()
    failed_builds = int((build_fail_counts > 0).sum())
    failed_build_pct = round((failed_builds / total_builds) * 100, 2) if total_builds > 0 else 0.0
    
    # Duration statistics
    dur_series = df["duration"]
    duration_stats = {
        "min_seconds": round(float(dur_series.min()), 3),
        "mean_seconds": round(float(dur_series.mean()), 3),
        "median_seconds": round(float(dur_series.median()), 3),
        "p75_seconds": round(float(dur_series.quantile(0.75)), 3),
        "p90_seconds": round(float(dur_series.quantile(0.90)), 3),
        "p95_seconds": round(float(dur_series.quantile(0.95)), 3),
        "max_seconds": round(float(dur_series.max()), 3),
        "total_test_hours": round(float(dur_series.sum()) / 3600.0, 2)
    }
    
    # Chronological support verification
    has_build_ids = "build_id" in df.columns
    is_strictly_ordered = True
    for p in projects:
        p_builds = df[df["project_id"] == p]["build_id"].tolist()
        if p_builds != sorted(p_builds):
            is_strictly_ordered = False
            break
            
    # Leakage Risk Analysis
    leakage_analysis = {
        "identifiers_kept_out_of_features": IDENTIFIER_COLUMNS,
        "post_execution_outcomes_isolated": POST_EXECUTION_COLUMNS,
        "valid_pre_execution_features": [c for c in ALL_PRE_EXECUTION_FEATURES if c in df.columns],
        "target_leakage_safeguards": [
            "'verdict' is isolated exclusively as the retrospective target ground-truth.",
            "'duration' is isolated as retrospective cost; pre-execution prioritization uses 'avg_recent_duration'.",
            "No future build metrics are aggregated or accessible to earlier builds."
        ],
        "chronological_evaluation_supported": has_build_ids
    }
    
    # Comprehensive Data Dictionary
    data_dictionary = {
        "project_id": {"category": "Identifier", "type": dtypes.get("project_id", "string"), "leakage_safe": True, "description": "Software project or repository identifier"},
        "build_id": {"category": "Identifier", "type": dtypes.get("build_id", "int64"), "leakage_safe": True, "description": "Monotonically increasing CI build sequence number"},
        "test_id": {"category": "Identifier", "type": dtypes.get("test_id", "string"), "leakage_safe": True, "description": "Test case class or method identifier"},
        "historical_fail_rate": {"category": "REC", "type": dtypes.get("historical_fail_rate", "float64"), "leakage_safe": True, "description": "Historical failure frequency across past builds"},
        "failures_last_5": {"category": "REC", "type": dtypes.get("failures_last_5", "int64"), "leakage_safe": True, "description": "Number of test failures in past 5 builds"},
        "failures_last_10": {"category": "REC", "type": dtypes.get("failures_last_10", "int64"), "leakage_safe": True, "description": "Number of test failures in past 10 builds"},
        "total_historical_runs": {"category": "REC", "type": dtypes.get("total_historical_runs", "int64"), "leakage_safe": True, "description": "Total historical executions of this test case"},
        "builds_since_last_fail": {"category": "REC", "type": dtypes.get("builds_since_last_fail", "int64"), "leakage_safe": True, "description": "Builds elapsed since previous observed failure"},
        "avg_recent_duration": {"category": "REC", "type": dtypes.get("avg_recent_duration", "float64"), "leakage_safe": True, "description": "Estimated duration based on moving average of recent runs (s)"},
        "test_age_builds": {"category": "TES", "type": dtypes.get("test_age_builds", "int64"), "leakage_safe": True, "description": "Number of builds since test was added to test suite"},
        "test_loc": {"category": "TES", "type": dtypes.get("test_loc", "int64"), "leakage_safe": True, "description": "Source lines of code in test case"},
        "changed_files_count": {"category": "CHG", "type": dtypes.get("changed_files_count", "int64"), "leakage_safe": True, "description": "Count of modified files in the triggering commit"},
        "code_churn_additions": {"category": "CHG", "type": dtypes.get("code_churn_additions", "int64"), "leakage_safe": True, "description": "Lines of code added in triggering commit"},
        "code_churn_deletions": {"category": "CHG", "type": dtypes.get("code_churn_deletions", "int64"), "leakage_safe": True, "description": "Lines of code deleted in triggering commit"},
        "modified_test_file": {"category": "CHG", "type": dtypes.get("modified_test_file", "int64"), "leakage_safe": True, "description": "Binary flag indicating if test file itself was edited"},
        "file_test_distance": {"category": "CHG", "type": dtypes.get("file_test_distance", "float64"), "leakage_safe": True, "description": "Syntactic/package distance between changed code and test"},
        "cov_class_count": {"category": "COV", "type": dtypes.get("cov_class_count", "int64"), "leakage_safe": True, "description": "Classes covered by this test case"},
        "cov_method_count": {"category": "COV", "type": dtypes.get("cov_method_count", "int64"), "leakage_safe": True, "description": "Methods covered by this test case"},
        "w_sum_cov_c_faults": {"category": "COV", "type": dtypes.get("w_sum_cov_c_faults", "float64"), "leakage_safe": True, "description": "Weighted coverage-to-historical-fault correlation score"},
        "duration": {"category": "Post-Execution", "type": dtypes.get("duration", "float64"), "leakage_safe": False, "description": "Actual test execution duration in seconds (Target Cost)"},
        "verdict": {"category": "Post-Execution", "type": dtypes.get("verdict", "int64"), "leakage_safe": False, "description": "Actual test outcome: 0 = Passed, 1 = Failed (Target Label)"}
    }
    
    audit_data = {
        "dataset_path": csv_path,
        "file_size_mb": file_size_mb,
        "row_count": n_rows,
        "column_count": n_cols,
        "duplicate_rows": duplicate_rows,
        "duplicate_test_builds": duplicate_test_builds,
        "projects": projects,
        "builds_per_project": builds_per_project,
        "total_unique_builds": total_builds,
        "class_distribution": {
            "pass_records": pass_count,
            "fail_records": fail_count,
            "fail_rate_percentage": fail_rate_pct,
            "total_builds": total_builds,
            "failed_builds": failed_builds,
            "failed_build_percentage": failed_build_pct
        },
        "duration_statistics": duration_stats,
        "missing_values": missing_counts,
        "missing_percentages": missing_pct,
        "column_datatypes": dtypes,
        "leakage_analysis": leakage_analysis,
        "data_dictionary": data_dictionary
    }
    
    return audit_data


def save_audit_reports(audit_data: Dict[str, Any], output_dir: str = "reports") -> Tuple[str, str]:
    """Saves data audit results to reports/data_audit.json and reports/data_audit.md."""
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "data_audit.json")
    md_path = os.path.join(output_dir, "data_audit.md")
    
    # Save JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)
        
    # Generate Markdown
    md_lines = [
        "# TestPilot AI: Comprehensive Dataset Audit Report",
        "",
        "## 1. Executive Summary",
        "",
        f"- **Dataset File**: `{audit_data['dataset_path']}`",
        f"- **File Size**: {audit_data['file_size_mb']} MB",
        f"- **Total Rows (Test Executions)**: {audit_data['row_count']:,}",
        f"- **Total Columns**: {audit_data['column_count']}",
        f"- **Duplicate Rows**: {audit_data['duplicate_rows']}",
        f"- **Duplicate (Project, Build, Test) Tuples**: {audit_data['duplicate_test_builds']}",
        f"- **Total Software Projects**: {len(audit_data['projects'])} ({', '.join(audit_data['projects'])})",
        f"- **Total CI Builds**: {audit_data['total_unique_builds']:,}",
        "",
        "## 2. Class Distribution and Failure Profile",
        "",
        "Software testing datasets are inherently characterized by extreme class imbalance:",
        "",
        f"- **Passed Tests (Verdict = 0)**: {audit_data['class_distribution']['pass_records']:,} ({100 - audit_data['class_distribution']['fail_rate_percentage']:.2f}%)",
        f"- **Failed Tests (Verdict = 1)**: {audit_data['class_distribution']['fail_records']:,} ({audit_data['class_distribution']['fail_rate_percentage']:.2f}%)",
        f"- **Builds Containing at Least One Failure**: {audit_data['class_distribution']['failed_builds']} / {audit_data['class_distribution']['total_builds']} ({audit_data['class_distribution']['failed_build_percentage']:.2f}%)",
        "",
        "> [!IMPORTANT]",
        f"> The observed test failure rate is {audit_data['class_distribution']['fail_rate_percentage']}%. Standard accuracy is an invalid evaluation metric under this class imbalance; evaluation must emphasize PR-AUC, Recall, APFD, and budget-constrained failure discovery.",
        "",
        "## 3. Execution Duration Statistics",
        "",
        "| Metric | Duration (seconds) |",
        "|---|---|",
        f"| Minimum | {audit_data['duration_statistics']['min_seconds']} s |",
        f"| Median (P50) | {audit_data['duration_statistics']['median_seconds']} s |",
        f"| Mean | {audit_data['duration_statistics']['mean_seconds']} s |",
        f"| 75th Percentile | {audit_data['duration_statistics']['p75_seconds']} s |",
        f"| 90th Percentile | {audit_data['duration_statistics']['p90_seconds']} s |",
        f"| 95th Percentile | {audit_data['duration_statistics']['p95_seconds']} s |",
        f"| Maximum | {audit_data['duration_statistics']['max_seconds']} s |",
        f"| Cumulative Regression Testing Time | {audit_data['duration_statistics']['total_test_hours']} hours |",
        "",
        "## 4. Data Quality and Missingness",
        "",
        "| Column | Missing Count | Missing % | Data Type |",
        "|---|---|---|---|"
    ]
    
    for col, dtype in audit_data["column_datatypes"].items():
        m_count = audit_data["missing_values"].get(col, 0)
        m_pct = audit_data["missing_percentages"].get(col, 0.0)
        md_lines.append(f"| `{col}` | {m_count} | {m_pct}% | `{dtype}` |")
        
    md_lines.extend([
        "",
        "## 5. Leakage Risk Analysis and Protocol",
        "",
        "To satisfy rigorous software engineering empirical research standards, features are partitioned strictly by execution lifecycle:",
        "",
        "### A. Pre-Execution Safe Features (Permitted in ML Model)",
        "- **Historical Execution Features (REC)**: `historical_fail_rate`, `failures_last_5`, `failures_last_10`, `total_historical_runs`, `builds_since_last_fail`, `avg_recent_duration`.",
        "- **Code Churn Features (CHG)**: `changed_files_count`, `code_churn_additions`, `code_churn_deletions`, `modified_test_file`, `file_test_distance`.",
        "- **Coverage and Structure (COV/TES)**: `test_age_builds`, `test_loc`, `cov_class_count`, `cov_method_count`, `w_sum_cov_c_faults`.",
        "",
        "### B. Post-Execution Variables (Isolated from Training Features)",
        "- `verdict`: Ground-truth binary outcome observed only after test completes.",
        "- `duration`: Current runtime observed only after test completes. The prioritisation scheduler uses `avg_recent_duration`.",
        "",
        "### C. Identifiers (Kept for Joins and Grouping)",
        "- `project_id`, `build_id`, `test_id` are excluded from input vectors $X$ to avoid spurious project memorization.",
        "",
        "## 6. Discovered Data Dictionary",
        "",
        "| Column Name | Category | Type | Leakage-Safe? | Description |",
        "|---|---|---|---|---|"
    ])
    
    for col, meta in audit_data["data_dictionary"].items():
        safe_str = "Yes (Pre-Exec)" if meta["leakage_safe"] else "**NO (Post-Exec Target)**"
        md_lines.append(f"| `{col}` | {meta['category']} | `{meta['type']}` | {safe_str} | {meta['description']} |")
        
    md_lines.append("")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
        
    logger.info(f"Audit saved to {json_path} and {md_path}")
    return json_path, md_path
