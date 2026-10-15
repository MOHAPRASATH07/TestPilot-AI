"""
CLI runner for Dataset Audit.
Executes audit_dataset() and outputs reports/data_audit.json and reports/data_audit.md.
"""

import os
import sys
import logging

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_audit import audit_dataset, save_audit_reports

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    print("=" * 60)
    print("TestPilot AI: Executing Dataset Inspection and Audit")
    print("=" * 60)
    audit_data = audit_dataset()
    json_path, md_path = save_audit_reports(audit_data)
    print(f"\n[SUCCESS] Audit complete.")
    print(f"  JSON output: {json_path}")
    print(f"  Markdown output: {md_path}")
    print(f"  Total records: {audit_data['row_count']:,}")
    print(f"  Failure rate: {audit_data['class_distribution']['fail_rate_percentage']:.2f}%")
    print(f"  Total builds: {audit_data['total_unique_builds']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
