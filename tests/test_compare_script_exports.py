from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


def test_compare_script_can_export_report_files(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "compare_acos.py"
    export_dir = tmp_path / "exports"

    completed = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--instance-set",
            "demo",
            "--seeds",
            "7",
            "8",
            "--iterations",
            "5",
            "--colony-size",
            "4",
            "--export-dir",
            str(export_dir),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

    summary_csv = export_dir / "comparison_summary.csv"
    runs_csv = export_dir / "comparison_runs.csv"
    markdown_summary = export_dir / "comparison_summary.md"

    assert summary_csv.exists()
    assert runs_csv.exists()
    assert markdown_summary.exists()
    assert "Exported summary CSV" in completed.stdout

    with summary_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["instance"] == "demo"

    with runs_csv.open(newline="", encoding="utf-8") as handle:
        run_rows = list(csv.DictReader(handle))
    assert len(run_rows) == 2
    assert {row["seed"] for row in run_rows} == {"7", "8"}

    markdown_text = markdown_summary.read_text(encoding="utf-8")
    assert "# Benchmark Summary" in markdown_text
    assert "| Instance | Capacity | Items |" in markdown_text
