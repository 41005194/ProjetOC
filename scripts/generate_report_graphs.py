from __future__ import annotations

import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
SUMMARY_CSV = RESULTS_DIR / "comparison_summary.csv"
RUNS_CSV = RESULTS_DIR / "comparison_runs.csv"


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>',
        "text { font-family: Arial, sans-serif; fill: #1f2937; }",
        ".title { font-size: 20px; font-weight: bold; }",
        ".label { font-size: 12px; }",
        ".small { font-size: 11px; }",
        '</style>',
    ]


def write_grouped_bar_chart(
    *,
    title: str,
    subtitle: str,
    labels: list[str],
    baseline_values: list[float],
    ml_values: list[float],
    output_path: Path,
    value_suffix: str,
) -> None:
    width = 980
    height = 520
    margin_left = 90
    margin_right = 40
    margin_top = 85
    margin_bottom = 110
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom
    max_value = max(baseline_values + ml_values + [1.0])
    group_width = chart_width / max(len(labels), 1)
    bar_width = min(36, group_width * 0.28)

    lines = _svg_header(width, height)
    lines.append('<rect x="0" y="0" width="100%" height="100%" fill="#f8fafc"/>')
    lines.append(f'<text class="title" x="{margin_left}" y="35">{title}</text>')
    lines.append(f'<text class="label" x="{margin_left}" y="58">{subtitle}</text>')

    axis_y = margin_top + chart_height
    lines.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{axis_y}" stroke="#475569" stroke-width="1.5"/>'
    )
    lines.append(
        f'<line x1="{margin_left}" y1="{axis_y}" x2="{margin_left + chart_width}" y2="{axis_y}" stroke="#475569" stroke-width="1.5"/>'
    )

    for step in range(6):
        value = max_value * step / 5
        y = axis_y - (value / max_value) * chart_height
        lines.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + chart_width}" y2="{y:.2f}" stroke="#e2e8f0" stroke-width="1"/>'
        )
        lines.append(
            f'<text class="small" x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end">{value:.2f}</text>'
        )

    for index, label in enumerate(labels):
        group_x = margin_left + index * group_width + group_width / 2
        baseline_value = baseline_values[index]
        ml_value = ml_values[index]
        baseline_height = (baseline_value / max_value) * chart_height
        ml_height = (ml_value / max_value) * chart_height
        baseline_x = group_x - bar_width - 6
        ml_x = group_x + 6
        baseline_y = axis_y - baseline_height
        ml_y = axis_y - ml_height

        lines.append(
            f'<rect x="{baseline_x:.2f}" y="{baseline_y:.2f}" width="{bar_width:.2f}" height="{baseline_height:.2f}" fill="#2563eb"/>'
        )
        lines.append(
            f'<rect x="{ml_x:.2f}" y="{ml_y:.2f}" width="{bar_width:.2f}" height="{ml_height:.2f}" fill="#f97316"/>'
        )
        lines.append(
            f'<text class="small" x="{baseline_x + bar_width / 2:.2f}" y="{baseline_y - 6:.2f}" text-anchor="middle">{baseline_value:.2f}{value_suffix}</text>'
        )
        lines.append(
            f'<text class="small" x="{ml_x + bar_width / 2:.2f}" y="{ml_y - 6:.2f}" text-anchor="middle">{ml_value:.2f}{value_suffix}</text>'
        )
        lines.append(
            f'<text class="label" x="{group_x:.2f}" y="{axis_y + 24}" text-anchor="middle">{label}</text>'
        )

    legend_x = width - 210
    legend_y = 35
    lines.append(f'<rect x="{legend_x}" y="{legend_y}" width="16" height="16" fill="#2563eb"/>')
    lines.append(f'<text class="label" x="{legend_x + 24}" y="{legend_y + 13}">Baseline ACO</text>')
    lines.append(f'<rect x="{legend_x}" y="{legend_y + 26}" width="16" height="16" fill="#f97316"/>')
    lines.append(f'<text class="label" x="{legend_x + 24}" y="{legend_y + 39}">ACO + ML</text>')

    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_runtime_delta_chart(rows: list[dict[str, str]], output_path: Path) -> None:
    width = 1100
    height = 540
    margin_left = 90
    margin_right = 40
    margin_top = 85
    margin_bottom = 110
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom

    labels = [f'{row["instance"]}:{row["seed"]}' for row in rows]
    values = [float(row["runtime_delta_seconds"]) for row in rows]
    max_abs = max(max((abs(value) for value in values), default=0.001), 0.001)
    zero_y = margin_top + chart_height / 2
    group_width = chart_width / max(len(values), 1)
    bar_width = min(20, group_width * 0.7)

    lines = _svg_header(width, height)
    lines.append('<rect x="0" y="0" width="100%" height="100%" fill="#f8fafc"/>')
    lines.append(f'<text class="title" x="{margin_left}" y="35">Delta de temps par seed</text>')
    lines.append(
        f'<text class="label" x="{margin_left}" y="58">Valeur positive: la variante ML est plus lente. Valeur negative: la variante ML est plus rapide.</text>'
    )

    lines.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + chart_height}" stroke="#475569" stroke-width="1.5"/>'
    )
    lines.append(
        f'<line x1="{margin_left}" y1="{zero_y:.2f}" x2="{margin_left + chart_width}" y2="{zero_y:.2f}" stroke="#334155" stroke-width="1.5"/>'
    )

    for factor in (-1, -0.5, 0.5, 1):
        value = max_abs * factor
        y = zero_y - (value / max_abs) * (chart_height / 2)
        lines.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + chart_width}" y2="{y:.2f}" stroke="#e2e8f0" stroke-width="1"/>'
        )
        lines.append(
            f'<text class="small" x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end">{value:.4f}</text>'
        )

    for index, value in enumerate(values):
        x = margin_left + index * group_width + (group_width - bar_width) / 2
        height_pixels = abs(value) / max_abs * (chart_height / 2)
        y = zero_y - height_pixels if value >= 0 else zero_y
        color = "#f97316" if value >= 0 else "#16a34a"
        lines.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{height_pixels:.2f}" fill="{color}"/>'
        )
        if index % 2 == 0:
            lines.append(
                f'<text class="small" x="{x + bar_width / 2:.2f}" y="{margin_top + chart_height + 20}" text-anchor="middle" transform="rotate(45 {x + bar_width / 2:.2f},{margin_top + chart_height + 20})">{labels[index]}</text>'
            )

    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not SUMMARY_CSV.exists() or not RUNS_CSV.exists():
        raise FileNotFoundError(
            "Missing benchmark CSV files. Run `python scripts/compare_acos.py --instance-set report` first."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = load_csv_rows(SUMMARY_CSV)
    run_rows = load_csv_rows(RUNS_CSV)

    labels = [row["instance"] for row in summary_rows]
    baseline_mean_values = [float(row["baseline_mean"]) for row in summary_rows]
    ml_mean_values = [float(row["ml_mean"]) for row in summary_rows]
    baseline_runtimes = [float(row["baseline_total_runtime_seconds"]) for row in summary_rows]
    ml_runtimes = [float(row["ml_total_runtime_seconds"]) for row in summary_rows]

    write_grouped_bar_chart(
        title="Comparaison des valeurs moyennes",
        subtitle="Moyenne de la meilleure valeur obtenue sur 10 seeds pour chaque instance.",
        labels=labels,
        baseline_values=baseline_mean_values,
        ml_values=ml_mean_values,
        output_path=RESULTS_DIR / "mean_value_comparison.svg",
        value_suffix="",
    )
    write_grouped_bar_chart(
        title="Comparaison des temps d'execution",
        subtitle="Temps total cumule sur l'ensemble des seeds pour chaque instance.",
        labels=labels,
        baseline_values=baseline_runtimes,
        ml_values=ml_runtimes,
        output_path=RESULTS_DIR / "runtime_comparison.svg",
        value_suffix="s",
    )
    write_runtime_delta_chart(run_rows, RESULTS_DIR / "runtime_delta_per_seed.svg")

    print(f"Generated {RESULTS_DIR / 'mean_value_comparison.svg'}")
    print(f"Generated {RESULTS_DIR / 'runtime_comparison.svg'}")
    print(f"Generated {RESULTS_DIR / 'runtime_delta_per_seed.svg'}")


if __name__ == "__main__":
    main()
