"""Plot mean fidelity values from saved top-k evaluation JSON files."""

import argparse
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt


def load_topk_evaluation(json_file):
    """Load percentage and mean fidelity values from one top-k JSON file."""
    data = json.loads(json_file.read_text(encoding="utf-8"))
    evaluations = data.get("evaluations", [])
    if not evaluations:
        raise ValueError(f"No evaluations found in {json_file}.")

    percentages = [evaluation["percentage"] for evaluation in evaluations]
    fidelity_plus = [evaluation["mean_fidelity_plus"] for evaluation in evaluations]
    fidelity_minus = [evaluation["mean_fidelity_minus"] for evaluation in evaluations]
    return percentages, fidelity_plus, fidelity_minus


def _plot_metadata(json_file):
    algorithm = json_file.parent.name
    match = re.search(r"_topk_(\d+)$", json_file.stem)
    if match is None:
        raise ValueError(f"Could not determine top-k step size from {json_file}.")
    return algorithm, int(match.group(1))


def plot_topk_fidelity(json_file):
    """Create a fidelity plot next to one top-k evaluation JSON file."""
    percentages, fidelity_plus, fidelity_minus = load_topk_evaluation(json_file)
    algorithm, step = _plot_metadata(json_file)

    figure, axis = plt.subplots(figsize=(9, 6))
    axis.plot(
        percentages,
        fidelity_plus,
        marker="o",
        linewidth=2,
        label="Mean fidelity+",
    )
    axis.plot(
        percentages,
        fidelity_minus,
        marker="o",
        linewidth=2,
        label="Mean fidelity-",
    )
    axis.set_xlabel("Retained edges (%)")
    axis.set_ylabel("Mean fidelity")
    axis.set_title(f"{algorithm} fidelity (step size {step})")
    axis.set_xlim(0, 100)
    axis.set_ylim(0, 1)
    axis.set_xticks(range(0, 101, 10))
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()

    output_file = json_file.with_suffix(".png")
    figure.savefig(output_file, dpi=200)
    plt.close(figure)
    return output_file


def plot_all_topk_fidelity(output_root):
    """Plot every top-k fidelity JSON below the output root."""
    json_files = sorted(output_root.glob("*/fidelity_*_topk_*.json"))
    if not json_files:
        raise FileNotFoundError(
            f"No top-k fidelity JSON files found below {output_root}."
        )

    output_files = []
    for json_file in json_files:
        output_files.append(plot_topk_fidelity(json_file))
    return output_files


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot mean fidelity against retained top-k edge percentages."
    )
    parser.add_argument(
        "--input-root",
        default="output/explanations",
        type=Path,
        help="Root directory containing algorithm-specific fidelity JSON files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    for output_file in plot_all_topk_fidelity(args.input_root):
        print(f"Saved top-k fidelity plot to: {output_file}")


if __name__ == "__main__":
    main()
