# ================================================================
#   visualise.py
# ================================================================

from pathlib import Path
import logging
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np

# ================================================================
#   Logging helper
# ================================================================
def setup_logging(output_dir: Path, logs_dir: Path = None, level=logging.INFO):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if logs_dir is None:
        logs_dir = out_dir / "logs"
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "pipeline.log"
    handlers = [logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()]
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=handlers)

# ================================================================
#   Seaborn theme
# ================================================================
sns.set_theme(style="whitegrid")

# ================================================================
#   Visuals
#   Sentiment Compound Score Over Time
#   Sentiment Distribution (Count Plot)
#   Average Sentiment Scores by Label
#   Pie chart
# ================================================================
def save_line(plot_data: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(12,6))
    plt.plot(plot_data["timestamp"], plot_data["compound"], marker="o", linewidth=2, markersize=6, alpha=0.8)
    plt.axhline(y=0, color="r", linestyle="--", alpha=0.3, label="Neutral Line")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.title("Sentiment Compound Score Over Time")
    plt.xlabel("Time")
    plt.ylabel("Compound Sentiment Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def save_count(df: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(6,4))
    sns.countplot(x="sentiment", data=df, hue="sentiment", palette="pastel",
                  order=["Positive","Neutral","Negative"], legend=False)
    plt.title("Sentiment Distribution (Count Plot)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def save_bar_avg(df: pd.DataFrame, out_path: Path):
    avg_scores = df.groupby("sentiment")[["neg","neu","pos"]].mean().reset_index()
    avg_melt = avg_scores.melt(id_vars="sentiment", value_vars=["neg","neu","pos"], var_name="Score Type", value_name="Average")
    plt.figure(figsize=(7,4))
    sns.barplot(x="sentiment", y="Average", hue="Score Type", data=avg_melt, palette="muted")
    plt.title("Average Sentiment Scores by Label")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def save_pie(df: pd.DataFrame, out_path: Path):
    logger = logging.getLogger(__name__)
    sentiment_counts = df["sentiment"].value_counts().reindex(["Positive","Neutral","Negative"])
    sentiment_counts = sentiment_counts.fillna(0).astype(float)
    sizes = sentiment_counts.values

    if not np.isfinite(sizes).all():
        logger.warning("Non-finite values in sentiment counts; coercing to zero.")
        sizes = np.nan_to_num(sizes, nan=0.0, posinf=0.0, neginf=0.0)

    total = sizes.sum()
    if total == 0:
        logger.warning("No sentiment counts available for pie chart; saving placeholder.")
        plt.figure(figsize=(6,6))
        plt.text(0.5, 0.5, "No data to display", ha="center", va="center", fontsize=12)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        return

    labels = sentiment_counts.index.tolist()
    plt.figure(figsize=(6,6))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140, wedgeprops={"edgecolor":"w"})
    plt.title("Sentiment Distribution (Pie Chart)")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# ================================================================
#   Run all visualisations (public)
# ================================================================
def run_visualisations(df: pd.DataFrame, out_dir: Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(__name__)

    required = {"timestamp", "compound", "sentiment"}
    missing = required - set(df.columns)
    if missing:
        logger.error("Analysed DataFrame is missing required columns: %s", missing)
        raise KeyError(f"Missing required columns for visualisation: {missing}")

    plot_df = df.copy()
    plot_df["timestamp"] = pd.to_datetime(plot_df["timestamp"], errors="coerce")
    if plot_df["timestamp"].isna().all():
        logger.warning("All timestamps are invalid; line plot will use index instead.")
        plot_df["timestamp"] = plot_df.index

    try:
        save_line(plot_df, out_dir / "sentiment_overtime.png")
        save_count(df, out_dir / "sentiment_distribution_count.png")
        save_bar_avg(df, out_dir / "average_sentiment_scores.png")
        save_pie(df, out_dir / "sentiment_distribution_pie.png")
        logger.info("saved plots to %s", out_dir)
    except Exception:
        logger.exception("Failed while generating visualisations")
        raise

# ================================================================
#   CLI + Main execution logic
#   Sources:
#       https://docs.python.org/3/library/argparse.html
#       https://realpython.com/command-line-interfaces-python-argparse/
#   Note:
#       This doesn't follow DRY principle so each file can be run independently
#       Allows for standalone execution for testing and modular use
# ================================================================
if __name__ == "__main__":
    import argparse
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    DEFAULT_OUTDIR = project_root / "output" / "main_outputs"
    DEFAULT_LOGDIR = project_root / "output" / "main_logging"
    DEFAULT_INPUT = DEFAULT_OUTDIR / "analysed.csv"

    parser = argparse.ArgumentParser(description="Generate visualisations from analysed CSV.")
    parser.add_argument("--input", "-i", default=str(DEFAULT_INPUT), help="Path to analysed CSV")
    parser.add_argument("--outdir", "-o", default=str(DEFAULT_OUTDIR), help="Output dir for plots")
    parser.add_argument("--logdir", "-L", default=str(DEFAULT_LOGDIR), help="Log directory")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Standard info or verbose logs?
    lvl = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(Path(args.outdir), logs_dir=Path(args.logdir), level=lvl)
    logger = logging.getLogger("visualise")

    # What's the path?
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = (Path.cwd() / input_path).resolve()
    if not input_path.exists():
        script_parent = Path(__file__).resolve().parent
        alt = (script_parent.parent / args.input).resolve()
        logger.debug("Primary input missing; trying script-based fallback: %s", alt)
        if alt.exists():
            input_path = alt
   # You're missing a file!
    if not input_path.exists():
        logger.error("Analysed CSV not found at resolved path(s): %s", input_path)
        raise FileNotFoundError(f"Analysed CSV not found: {input_path}. Run analyse.py or main.py first.")
    # Was it successful? If not, notify user!
    try:
        df = pd.read_csv(input_path)
        logger.info("Loaded analysed CSV from %s (shape=%s)", input_path, df.shape)
        run_visualisations(df, Path(args.outdir))
        logger.info("visualisation step complete")
    except Exception as e:
        logger.exception("visualisation failed: %s", e)
        raise
