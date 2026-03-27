from __future__ import annotations

import argparse
import math
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

INPUT_MARKERS: Dict[str, str] = {
    "Random": "o",
    "Reversed": "s",
    "Nearly Sorted": "^",
    "Many Duplicates": "D",
}

INPUT_PALETTE: Dict[str, str] = {
    "Random": "#4f5d75",
    "Reversed": "#ef8354",
    "Nearly Sorted": "#2a9d8f",
    "Many Duplicates": "#bc6c25",
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
    return df.sort_values(["algorithm", "input_type", "n"]).reset_index(drop=True)


def setup_plot_theme() -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=1.02)


def save_figure_outputs(fig: plt.Figure, output_path: Path) -> None:
    fig.savefig(output_path, dpi=320, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")


def estimate_k_for_counting(n: int, input_type: str, k_mode: str, k_fixed: int) -> int:
    if k_mode == "fixed":
        return int(k_fixed)

    if input_type == "many_duplicates":
        return max(2, n // 50)

    return n


def insertion_avg_comparisons(n: int) -> float:
    # E[C_n] for insertion sort on random permutation (distinct keys).
    return (n * (n - 1)) / 4.0 + (n - 1)


def harmonic_prefix(max_n: int) -> List[float]:
    h = [0.0] * (max_n + 1)
    for i in range(1, max_n + 1):
        h[i] = h[i - 1] + 1.0 / i
    return h


def quicksort_avg_comparisons(n: int, harmonics: List[float]) -> float:
    # E[C_n] = 2(n+1)H_n - 4n for randomized quicksort comparisons.
    return 2.0 * (n + 1) * harmonics[n] - 4.0 * n


def theoretical_terms(
    algorithm: str,
    n: int,
    input_type: str,
    harmonics: List[float],
    k_mode: str,
    k_fixed: int,
) -> Dict[str, float | int | str]:
    if algorithm == "Insertion Sort":
        return {
            "g_of_n": float(n * n),
            "g_label": "n^2",
            "exact_comparison_model": insertion_avg_comparisons(n),
            "k_value": 0,
        }

    if algorithm == "Quick Sort":
        return {
            "g_of_n": float(n * math.log2(n)),
            "g_label": "n log2 n",
            "exact_comparison_model": quicksort_avg_comparisons(n, harmonics),
            "k_value": 0,
        }

    if algorithm == "Counting Sort":
        k_value = estimate_k_for_counting(n, input_type, k_mode, k_fixed)
        return {
            "g_of_n": float(n + k_value),
            "g_label": "n + k",
            "exact_comparison_model": float(n + k_value),
            "k_value": int(k_value),
        }

    raise ValueError(f"Unsupported algorithm: {algorithm}")


def build_ratio_table(df: pd.DataFrame, k_mode: str, k_fixed: int) -> pd.DataFrame:
    max_n = int(df["n"].max())
    harmonics = harmonic_prefix(max_n)

    rows: List[Dict[str, object]] = []
    for rec in df.to_dict(orient="records"):
        algorithm = str(rec["algorithm"])
        input_type = str(rec["input_type"])
        n = int(rec["n"])
        time_ms = float(rec["time_ms"])

        terms = theoretical_terms(algorithm, n, input_type, harmonics, k_mode, k_fixed)
        g_of_n = float(terms["g_of_n"])
        exact_comp = float(terms["exact_comparison_model"])

        comparisons = rec["comparisons"]
        comp_ratio = None
        if comparisons is not None and not pd.isna(comparisons):
            comp_ratio = float(comparisons) / exact_comp

        rows.append(
            {
                **rec,
                "k_used": int(terms["k_value"]),
                "g_of_n": g_of_n,
                "g_label": str(terms["g_label"]),
                "exact_comparison_model": exact_comp,
                "time_ratio": time_ms / g_of_n,
                "comparison_ratio": comp_ratio,
            }
        )

    ratio_df = pd.DataFrame(rows)
    ratio_df["algorithm"] = pd.Categorical(
        ratio_df["algorithm"], ALGORITHM_ORDER, ordered=True
    )
    ratio_df["input_type"] = pd.Categorical(
        ratio_df["input_type"], INPUT_TYPE_ORDER, ordered=True
    )
    ratio_df["input_type_label"] = pd.Categorical(
        ratio_df["input_type_label"], INPUT_TYPE_LABEL_ORDER, ordered=True
    )
    return ratio_df.sort_values(["algorithm", "input_type", "n"]).reset_index(drop=True)


def save_time_ratio_facets(
    df_ratio: pd.DataFrame, output_path: Path, yscale: str
) -> None:
    g = sns.relplot(
        data=df_ratio,
        x="n",
        y="time_ratio",
        hue="input_type_label",
        style="input_type_label",
        markers=INPUT_MARKERS,
        dashes=False,
        kind="line",
        col="algorithm",
        col_order=ALGORITHM_ORDER,
        linewidth=2.2,
        height=4.0,
        aspect=1.15,
        palette=INPUT_PALETTE,
        markeredgecolor="black",
        markeredgewidth=0.55,
        legend=False,
        facet_kws={"sharey": False},
    )

    g.set_axis_labels("", "")
    g.set_titles("{col_name}")

    y_label = "Empirical time / g(n)"
    if yscale == "log":
        y_label = "Empirical time / g(n) (log scale)"

    for idx, ax in enumerate(g.axes.flat):
        ax.set_xscale("log")
        ax.set_xticks([100, 1000, 10000, 100000])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.tick_params(axis="x", rotation=20)

        if idx == 0:
            # ── FIX: smaller fontsize for y-label so it doesn't crowd the plot ──
            ax.set_ylabel(y_label, labelpad=6, fontsize=11)
        else:
            ax.set_ylabel("")
        ax.set_xlabel("Input size (n)", labelpad=6)

        if yscale == "log":
            ax.set_yscale("log")

        ax.axhline(1.0, color="#2b2d42", linestyle=":", linewidth=1.2, alpha=0.8)
        ax.grid(True, which="major", linestyle="--", alpha=0.35)

    handles = [
        Line2D(
            [0],
            [0],
            color=INPUT_PALETTE[label],
            marker=INPUT_MARKERS[label],
            linewidth=2.2,
            markersize=8,
            markeredgecolor="black",
            markeredgewidth=0.55,
            label=label,
        )
        for label in INPUT_TYPE_LABEL_ORDER
    ]

    # layout: suptitle (top) → legend (middle) → panel col titles (bottom of header)
    g.figure.subplots_adjust(top=0.62, bottom=0.17, left=0.08, right=0.98, wspace=0.15)

    g.figure.suptitle(
        "Empirical Ratio T(n)/g(n) by Algorithm",
        fontsize=17,
        fontweight="bold",
        y=1.08,
    )

    # legend sits in the gap between suptitle and the panel col-title row
    g.figure.legend(
        handles,
        INPUT_TYPE_LABEL_ORDER,
        title="Input Type",
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.95),
        frameon=True,
        fontsize=10,
        title_fontsize=10,
    )
    save_figure_outputs(g.figure, output_path)
    plt.close(g.figure)


def save_comparison_ratio_plot(
    df_ratio: pd.DataFrame, output_path: Path, yscale: str
) -> bool:
    subset = df_ratio[df_ratio["comparison_ratio"].notna()].copy()
    if subset.empty:
        return False

    g = sns.relplot(
        data=subset,
        x="n",
        y="comparison_ratio",
        hue="input_type_label",
        style="input_type_label",
        markers=INPUT_MARKERS,
        dashes=False,
        kind="line",
        col="algorithm",
        col_order=ALGORITHM_ORDER,
        linewidth=2.2,
        height=4.0,
        aspect=1.15,
        palette=INPUT_PALETTE,
        markeredgecolor="black",
        markeredgewidth=0.55,
        legend=False,
        facet_kws={"sharey": False},
    )

    g.set_axis_labels("", "")
    g.set_titles("{col_name}")

    y_label = "Empirical comparisons / exact model"
    if yscale == "log":
        y_label = "Empirical comparisons / exact model (log scale)"

    for idx, ax in enumerate(g.axes.flat):
        ax.set_xscale("log")
        ax.set_xticks([100, 1000, 10000, 100000])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.tick_params(axis="x", rotation=20)

        if idx == 0:
            # ── FIX: smaller fontsize for y-label so it doesn't crowd the plot ──
            ax.set_ylabel(y_label, labelpad=6, fontsize=11)
        else:
            ax.set_ylabel("")
        ax.set_xlabel("Input size (n)", labelpad=6)

        if yscale == "log":
            ax.set_yscale("log")

        ax.axhline(1.0, color="#2b2d42", linestyle=":", linewidth=1.2, alpha=0.8)
        ax.grid(True, which="major", linestyle="--", alpha=0.35)

    handles = [
        Line2D(
            [0],
            [0],
            color=INPUT_PALETTE[label],
            marker=INPUT_MARKERS[label],
            linewidth=2.2,
            markersize=8,
            markeredgecolor="black",
            markeredgewidth=0.55,
            label=label,
        )
        for label in INPUT_TYPE_LABEL_ORDER
    ]

    # ── FIX: move legend above the panel titles (higher bbox_to_anchor y) ──
    g.figure.legend(
        handles,
        INPUT_TYPE_LABEL_ORDER,
        title="Input Type",
        loc="center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.96),
        frameon=True,
        fontsize=10,
        title_fontsize=10,
    )

    g.figure.suptitle(
        "Comparison Ratio: Measured / Exact Model",
        fontsize=17,
        fontweight="bold",
        y=1.10,
    )

    # ── FIX: increase top margin so panel titles sit below legend comfortably ──
    g.figure.subplots_adjust(top=0.82, bottom=0.17, left=0.08, right=0.98, wspace=0.15)
    save_figure_outputs(g.figure, output_path)
    plt.close(g.figure)
    return True


def export_ratio_tables(df_ratio: pd.DataFrame, output_dir: Path) -> None:
    tidy_path = output_dir / "theory_ratio_tidy.csv"
    summary_path = output_dir / "theory_ratio_summary.csv"

    df_ratio.to_csv(tidy_path, index=False)

    summary = (
        df_ratio.groupby(["algorithm", "input_type", "n", "g_label"], observed=True)[
            ["time_ratio", "comparison_ratio"]
        ]
        .agg(["mean", "median", "min", "max"])
        .reset_index()
    )

    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else str(col)
        for col in summary.columns
    ]
    summary.to_csv(summary_path, index=False)


def build_theory_validation(
    results_dir: Path,
    output_dir: Path,
    ratio_yscale: str,
    comparison_yscale: str,
    k_mode: str,
    k_fixed: int,
) -> None:
    setup_plot_theme()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_all_results(results_dir)
    df_ratio = build_ratio_table(df, k_mode=k_mode, k_fixed=k_fixed)

    export_ratio_tables(df_ratio, output_dir)

    save_time_ratio_facets(
        df_ratio,
        output_dir / f"figure_05_time_ratio_over_theory_{ratio_yscale}.png",
        yscale=ratio_yscale,
    )

    has_comparison = save_comparison_ratio_plot(
        df_ratio,
        output_dir / f"figure_06_comparison_ratio_exact_{comparison_yscale}.png",
        yscale=comparison_yscale,
    )

    print("[OK] Theory validation pipeline completed.")
    print(f"[OK] Input folder:  {results_dir}")
    print(f"[OK] Output folder: {output_dir}")
    print("[OK] Theoretical models:")
    print("     - Insertion Sort: g(n)=n^2, exact E[C_n]=n(n-1)/4 + (n-1)")
    print("     - Quick Sort:     g(n)=n log2 n, exact E[C_n]=2(n+1)H_n - 4n")
    print(f"     - Counting Sort:  g(n)=n+k, k mode={k_mode}, k_fixed={k_fixed}")
    print("[OK] Created files:")
    print("     - theory_ratio_tidy.csv")
    print("     - theory_ratio_summary.csv")
    print(f"     - figure_05_time_ratio_over_theory_{ratio_yscale}.png")
    if has_comparison:
        print(f"     - figure_06_comparison_ratio_exact_{comparison_yscale}.png")
    else:
        print("     - (comparison figure skipped: no comparison data available)")
    print("[OK] Vector versions also exported for each figure: .pdf and .svg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate sorting growth models by visualizing empirical ratio T(n)/g(n) "
            "with standard theoretical normalizers."
        )
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
        help="Folder to save generated ratio figures and summary tables",
    )
    parser.add_argument(
        "--ratio-yscale",
        choices=["linear", "log"],
        default="linear",
        help="Y-axis scale for time ratio plot",
    )
    parser.add_argument(
        "--comparison-yscale",
        choices=["linear", "log"],
        default="linear",
        help="Y-axis scale for comparison ratio plot",
    )
    parser.add_argument(
        "--k-mode",
        choices=["from-generator", "fixed"],
        default="from-generator",
        help=(
            "How to choose k for Counting Sort model n+k. "
            "from-generator uses the known generator rule; fixed uses --k-fixed."
        ),
    )
    parser.add_argument(
        "--k-fixed",
        type=int,
        default=100000,
        help="Fixed k value used only when --k-mode fixed",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_theory_validation(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        ratio_yscale=args.ratio_yscale,
        comparison_yscale=args.comparison_yscale,
        k_mode=args.k_mode,
        k_fixed=args.k_fixed,
    )


if __name__ == "__main__":
    main()
