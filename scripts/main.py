# ================================================================
#   main.py
# ================================================================

import argparse
import logging
from pathlib import Path
import pandas as pd

from ingest import stream_jsonl, setup_logging as ingest_setup_logging
from preprocess import df_preprocess, setup_logging as preprocess_setup_logging
from analyse import run_vader, setup_logging as analyse_setup_logging
from visualise import run_visualisations, setup_logging as visualise_setup_logging

# ================================================================
#   Pipeline orchestration
#               Sources:
#               https://docs.python.org/3/library/logging.html
#               https://realpython.com/python-logging/
# ================================================================
def setup_all_logs(outdir: Path, logdir: Path, verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    ingest_setup_logging(outdir, logs_dir=logdir, level=level)
    preprocess_setup_logging(outdir, logs_dir=logdir, level=level)
    analyse_setup_logging(outdir, logs_dir=logdir, level=level)
    visualise_setup_logging(outdir, logs_dir=logdir, level=level)
    logging.getLogger().setLevel(level)

# ================================================================
#   Summary report generator
# ================================================================
def generate_summary_report(df: pd.DataFrame, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    report_filename = outdir / "summary_report.txt"

    total_messages = len(df)
    sentiment_counts = df["sentiment"].value_counts().reindex(
        ["Positive", "Negative", "Neutral"]
    ).fillna(0).astype(int)
    avg_compound = df["compound"].mean() if "compound" in df.columns else float ("nan")

    with open(report_filename, "w", encoding="utf-8") as f:
        f.write("Sentiment Analysis Summary Report\n")
        f.write("="*40 + "\n")
        f.write(f"Total messages analysed: {total_messages}\n\n")
        f.write("Sentiment distribution:\n")
        for sentiment, count in sentiment_counts.items():
            f.write(f"  {sentiment}: {count} ({count/total_messages*100:.2f}%)\n")
        f.write(f"\nAverage compound score: {avg_compound:.4f}\n")
        f.write("="*40 + "\n")

# ================================================================
#   The main pipeline
# ================================================================

def run_pipeline(input_path: Path, outdir: Path, logdir: Path, simulate: bool = False, chunk_size: int = 1, delay: float = 0.0, verbose: bool = False):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logdir = Path(logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    setup_all_logs(outdir, logdir, verbose=verbose)
    logger = logging.getLogger("pipeline")
    csv_out = outdir / "sentiment_results.csv"

    if not simulate:
        logger.info("running pipeline (batch)")
        rows = [obj for obj in stream_jsonl(input_path, chunk_size=chunk_size, delay=0.0)]
        df = pd.DataFrame(rows)
        logger.info("loaded %d rows from ingest", len(df))
        df_processed = df_preprocess(df)
        df_analysed = run_vader(df_processed)

        generate_summary_report(df_analysed, outdir)

        df_analysed.to_csv(csv_out, index=False, encoding="utf-8")
        run_visualisations(df_analysed, outdir)
        logger.info("batch pipeline complete. outputs: %s", outdir)
    else:
        logger.info("running pipeline (simulated streaming)")
        buffer = []
        for obj in stream_jsonl(input_path, chunk_size=chunk_size, delay=delay):
            buffer.append(obj)
            if len(buffer) >= chunk_size:
                df_chunk = pd.DataFrame(buffer)
                df_proc = df_preprocess(df_chunk)
                df_analysed = run_vader(df_proc)
                if csv_out.exists():
                    df_analysed.to_csv(csv_out, mode="a", header=False, index=False, encoding="utf-8")
                else:
                    df_analysed.to_csv(csv_out, mode="w", header=True, index=False, encoding="utf-8")
                df_full = pd.read_csv(csv_out)
                run_visualisations(df_full, outdir)
                buffer = []

        if csv_out.exists():
            df_full = pd.read_csv(csv_out)
            generate_summary_report(df_full, outdir)

        logger.info("streaming pipeline finished. outputs: %s", outdir)

# ================================================================
#   CLI + Main execution logic
#   Sources:
#       https://docs.python.org/3/library/argparse.html
#       https://realpython.com/command-line-interfaces-python-argparse/
#   Notes:
#       This doesn't follow DRY principle so each file can be run independently
#       Allows for standalone execution for testing and modular use
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sentiment pipeline")
    project_root = Path(__file__).resolve().parent.parent
    default_outputs = project_root / "output" / "main_outputs"
    default_logs = project_root / "output" / "main_logging"
    default_input = project_root / "data" / "sample_stream_large.jsonl"

    parser.add_argument("-i", "--input", default=str(default_input), help="Path to sample_stream.jsonl")
    parser.add_argument("-o", "--outdir", default=str(default_outputs), help="Output directory (csv and plots)")
    parser.add_argument("-L", "--logdir", default=str(default_logs), help="Log directory")
    parser.add_argument("--simulate", action="store_true", help="Simulate streaming ingestion")
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    run_pipeline(
        Path(args.input),
        Path(args.outdir),
        Path(args.logdir),
        simulate=args.simulate,
        chunk_size=args.chunk_size,
        delay=args.delay,
        verbose=args.verbose
  )