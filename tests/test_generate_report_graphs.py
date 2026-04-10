from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_generate_report_graphs_creates_svg_outputs(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    results_dir = root / "results"
    compare_script = root / "scripts" / "compare_acos.py"
    graph_script = root / "scripts" / "generate_report_graphs.py"

    subprocess.run(
        [
            sys.executable,
            str(compare_script),
            "--instance-set",
            "demo",
            "--seeds",
            "7",
            "8",
            "--iterations",
            "5",
            "--colony-size",
            "4",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    completed = subprocess.run(
        [sys.executable, str(graph_script)],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

    mean_value_svg = results_dir / "mean_value_comparison.svg"
    runtime_svg = results_dir / "runtime_comparison.svg"
    runtime_delta_svg = results_dir / "runtime_delta_per_seed.svg"

    assert mean_value_svg.exists()
    assert runtime_svg.exists()
    assert runtime_delta_svg.exists()
    assert "Generated" in completed.stdout
    assert "<svg" in mean_value_svg.read_text(encoding="utf-8")
