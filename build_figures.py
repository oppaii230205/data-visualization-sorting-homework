from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns

INPUT_TYPE_LABELS: Dict[str, str] = {
    "random": "Random",
    "reverse": "Reversed",
    "nearly-sorted": "Nearly Sorted",
    "many_duplicates": "Many Duplicates",
}
INPUT_TYPE_ORDER: List[str] = list(INPUT_TYPE_LABELS.keys())
INPUT_TYPE_LABEL_ORDER: List[str] = [INPUT_TYPE_LABELS[k] for k in INPUT_TYPE_ORDER]

ALGORITHM_ORDER: List[str] = ["Insertion Sort", "Quick Sort", "Counting Sort"]
ALGORITHM_MARKERS: Dict[str, str] = {
    "Insertion Sort": "o",
    "Quick Sort": "s",
    "Counting Sort": "D",
}

PALETTE: Dict[str, str] = {
    "Insertion Sort": "#d1495b",
    "Quick Sort": "#00798c",
    "Counting Sort": "#edae49",
}


def normalize_algorithm_name(name: str) -> str:
    cleaned = str(name).strip().lower()
    mapping = {
        "insertion sort": "Insertion Sort",
        "quick sort": "Quick Sort",
        "counting sort": "Counting Sort",
    }
    return mapping.get(cleaned, str(name).strip())


def find_row_index(df: pd.DataFrame, keyword: str) -> int:
    keyword_lower = keyword.lower()
    for idx in range(len(df)):
        value = str(df.iat[idx, 0]).strip().lower()
        if keyword_lower in value:
            return idx
    raise ValueError(f"Cannot find row containing keyword '{keyword}'.")


def parse_result_file(file_path: Path) -> pd.DataFrame:
    raw = pd.read_excel(file_path, header=None)
    data_size_row = find_row_index(raw, "data size")
    metric_row = find_row_index(raw, "resulting")

    records = []
    input_type = file_path.stem

    # Columns are arranged in pairs: [time_col, comparison_col].
    for col in range(1, raw.shape[1], 2):
        n_value = pd.to_numeric(raw.iat[data_size_row, col], errors="coerce")
        if pd.isna(n_value):
            continue

        metric_cell = str(raw.iat[metric_row, col]).strip().lower()
        if "running" not in metric_cell:
            continue

        comp_col = col + 1
        for row in range(metric_row + 1, raw.shape[0]):
            algorithm = raw.iat[row, 0]
            if pd.isna(algorithm):
                continue

            algorithm_name = normalize_algorithm_name(algorithm)
            if algorithm_name not in ALGORITHM_ORDER:
                continue

            time_ms = pd.to_numeric(raw.iat[row, col], errors="coerce")
            comparisons = None
            if comp_col < raw.shape[1]:
                comparisons = pd.to_numeric(raw.iat[row, comp_col], errors="coerce")

            if pd.isna(time_ms):
                continue

            records.append(
                {
                    "input_type": input_type,
                    "algorithm": algorithm_name,
                    "n": int(n_value),
                    "time_ms": float(time_ms),
                    "comparisons": None if pd.isna(comparisons) else float(comparisons),
                }
            )

    if not records:
        raise ValueError(f"No valid records parsed from file: {file_path}")

    return pd.DataFrame.from_records(records)


def load_all_results(results_dir: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for input_type in INPUT_TYPE_LABELS:
        file_path = results_dir / f"{input_type}.xlsx"
        if not file_path.exists():
            raise FileNotFoundError(f"Missing required result file: {file_path}")
        frames.append(parse_result_file(file_path))

    df = pd.concat(frames, ignore_index=True)
    df["input_type_label"] = df["input_type"].map(INPUT_TYPE_LABELS)
    df["input_type"] = pd.Categorical(df["input_type"], INPUT_TYPE_ORDER, ordered=True)
    df["input_type_label"] = pd.Categorical(
        df["input_type_label"], INPUT_TYPE_LABEL_ORDER, ordered=True
    )
    df["algorithm"] = pd.Categorical(df["algorithm"], ALGORITHM_ORDER, ordered=True)
    return df.sort_values(["input_type", "algorithm", "n"]).reset_index(drop=True)


def setup_plot_theme() -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=1.05)


def save_figure_outputs(fig: plt.Figure, output_path: Path) -> None:
    fig.savefig(output_path, dpi=320, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")


def save_time_facet_line(df: pd.DataFrame, output_path: Path, yscale: str) -> None:
    g = sns.relplot(
        data=df,
        x="n",
        y="time_ms",
        hue="algorithm",
        style="algorithm",
        markers=ALGORITHM_MARKERS,
        dashes=False,
        kind="line",
        col="input_type_label",
        col_order=INPUT_TYPE_LABEL_ORDER,
        col_wrap=2,
        linewidth=2.4,
        height=4.1,
        aspect=1.25,
        palette=PALETTE,
        markeredgecolor="black",
        markeredgewidth=0.6,
        legend=False,
    )

    g.set_axis_labels("", "")
    g.set_titles("{col_name}")

    y_label = "Execution time (ms)" if yscale == "linear" else "Execution time (ms, log scale)"

    for idx, ax in enumerate(g.axes.flat):
        row_idx = idx // 2
        col_idx = idx % 2

        ax.set_xscale("log")
        ax.set_xticks([100, 1000, 10000, 100000])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.tick_params(axis="x", rotation=20)

        if row_idx == 1:
            ax.set_xlabel("Input size (n)", labelpad=6)
        else:
            ax.set_xlabel("")

        if col_idx == 0 and row_idx == 0:
            ax.set_ylabel(y_label, labelpad=6)
        else:
            ax.set_ylabel("")

        if yscale == "log":
            ax.set_yscale("log")
        ax.grid(True, which="major", linestyle="--", alpha=0.35)

    handles = [
        Line2D(
            [0],
            [0],
            color=PALETTE[name],
            marker=ALGORITHM_MARKERS[name],
            linewidth=2.4,
            markersize=8,
            markeredgecolor="black",
            markeredgewidth=0.6,
            label=name,
        )
        for name in ALGORITHM_ORDER
    ]
    g.figure.legend(
        handles,
        ALGORITHM_ORDER,
        title="Algorithm",
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.98),
        frameon=True,
    )

    scale_label = "Linear Y" if yscale == "linear" else "Log Y"
    g.figure.suptitle(
        f"Sorting Performance by Data Type ({scale_label})",
        fontsize=18,
        fontweight="bold",
        y=1.07,
    )

    g.figure.subplots_adjust(top=0.82, bottom=0.12, left=0.08, right=0.97, hspace=0.24, wspace=0.1)
    save_figure_outputs(g.figure, output_path)
    plt.close(g.figure)


def save_grouped_bar_at_max_n(df: pd.DataFrame, output_path: Path) -> None:
    max_n = int(df["n"].max())
    subset = df[df["n"] == max_n].copy()

    plt.figure(figsize=(13.5, 6.6))
    ax = sns.barplot(
        data=subset,
        x="input_type_label",
        y="time_ms",
        hue="algorithm",
        order=INPUT_TYPE_LABEL_ORDER,
        hue_order=ALGORITHM_ORDER,
        palette=PALETTE,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_title(
        f"Execution Time at n = {max_n} Across Data Types",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xlabel("Input type")
    ax.set_ylabel("Execution time (ms)")
    ax.set_yscale("log")
    ax.grid(True, which="major", axis="y", linestyle="--", alpha=0.35)
    ax.legend(title="Algorithm", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)

    fig = plt.gcf()
    plt.tight_layout(rect=[0.0, 0.0, 0.86, 1.0])
    save_figure_outputs(fig, output_path)
    plt.close()


def save_comparison_chart_if_available(df: pd.DataFrame, output_path: Path) -> bool:
    if df["comparisons"].isna().all():
        return False

    plt.figure(figsize=(11.8, 6.0))
    ax = sns.lineplot(
        data=df,
        x="n",
        y="comparisons",
        hue="algorithm",
        style="algorithm",
        markers=ALGORITHM_MARKERS,
        dashes=False,
        linewidth=2.4,
        palette=PALETTE,
        hue_order=ALGORITHM_ORDER,
        style_order=ALGORITHM_ORDER,
        errorbar=None,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([100, 1000, 10000, 100000])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_title("Comparison Counts (Aggregated Across Input Types)", fontsize=16, fontweight="bold")
    ax.set_xlabel("Input size (n)")
    ax.set_ylabel("Comparisons (log scale)")
    ax.grid(True, which="major", linestyle="--", alpha=0.35)
    ax.legend(title="Algorithm", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)

    fig = plt.gcf()
    plt.tight_layout(rect=[0.0, 0.0, 0.84, 1.0])
    save_figure_outputs(fig, output_path)
    plt.close()
    return True


def export_summary_tables(df: pd.DataFrame, output_dir: Path) -> None:
    tidy_path = output_dir / "tidy_results.csv"
    summary_path = output_dir / "summary_time_ms.csv"

    df.to_csv(tidy_path, index=False)

    summary = (
        df.groupby(["input_type", "algorithm", "n"], observed=True)["time_ms"]
        .agg(["mean", "median", "min", "max"])
        .reset_index()
    )
    summary.to_csv(summary_path, index=False)


def build_figures(results_dir: Path, output_dir: Path) -> None:
    setup_plot_theme()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_all_results(results_dir)
    export_summary_tables(df, output_dir)

    save_time_facet_line(df, output_dir / "figure_01_time_linear.png", yscale="linear")
    save_time_facet_line(df, output_dir / "figure_02_time_logy.png", yscale="log")
    save_grouped_bar_at_max_n(df, output_dir / "figure_03_time_bar_nmax_logy.png")

    has_comparison = save_comparison_chart_if_available(
        df,
        output_dir / "figure_04_comparisons_loglog.png",
    )

    print("[OK] Pipeline completed.")
    print(f"[OK] Input folder:  {results_dir}")
    print(f"[OK] Output folder: {output_dir}")
    print("[OK] Created files:")
    print("     - tidy_results.csv")
    print("     - summary_time_ms.csv")
    print("     - figure_01_time_linear.png")
    print("     - figure_02_time_logy.png")
    print("     - figure_03_time_bar_nmax_logy.png")
    if has_comparison:
        print("     - figure_04_comparisons_loglog.png")
    print("[OK] Vector versions also exported for each figure: .pdf and .svg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build publication-ready sorting figures from existing Excel result files."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Folder containing random.xlsx, reverse.xlsx, nearly-sorted.xlsx, many_duplicates.xlsx",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Folder to save generated figures and summary tables",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_figures(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
