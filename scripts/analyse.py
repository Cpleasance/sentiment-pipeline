# ================================================================
#   analyse.py
# ================================================================

import logging
from pathlib import Path
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer # will call sia later on

# ================================================================
#   Logging
#               Sources:
#               https://docs.python.org/3/library/logging.html
#               https://docs.python.org/3/howto/logging.html
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
#   VADER: ensure lexicon is there
# ================================================================
def ensure_vader():
    try:
        _ = nltk.data.find("sentiment/vader_lexicon")
    except LookupError:
        nltk.download("vader_lexicon")

# ================================================================
#   VADER analysis
#               Sources:
#                https://www.geeksforgeeks.org/python/python-sentiment-analysis-using-vader/
# ================================================================
def label_sentiment(compound: float) -> str:
    if compound >= 0.05:
        return "Positive"
    if compound <= -0.05:
        return "Negative"
    return "Neutral"

def run_vader(df: pd.DataFrame, text_col: str = "processed_text") -> pd.DataFrame:
    ensure_vader()
    sia = SentimentIntensityAnalyzer()
    df = df.copy()
    df[text_col] = df[text_col].fillna("").astype(str)
    df["vader_scores"] = df[text_col].apply(lambda x: sia.polarity_scores(x))
    df[["neg", "neu", "pos", "compound"]] = df["vader_scores"].apply(pd.Series)
    df["compound"] = df["compound"].fillna(0.0)
    df["sentiment"] = df["compound"].apply(label_sentiment)
    return df

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
    DEFAULT_INPUT = DEFAULT_OUTDIR / "processed.csv"
    DEFAULT_OUTPUT = DEFAULT_OUTDIR / "analysed.csv"

    parser = argparse.ArgumentParser(description="Apply VADER sentiment analysis")
    parser.add_argument("-i", "--input", default=str(DEFAULT_INPUT), help="Processed CSV input")
    parser.add_argument("-o", "--output", default=str(DEFAULT_OUTPUT), help="Analysed CSV output")
    parser.add_argument("-l", "--outdir", default=str(DEFAULT_OUTDIR), help="Output directory for outputs")
    parser.add_argument("-L", "--logdir", default=str(DEFAULT_LOGDIR), help="Log directory")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Standard info or verbose logs?
    logging_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(Path(args.outdir), logs_dir=Path(args.logdir), level=logging_level)
    logger = logging.getLogger("analyse")

    try:
        df = pd.read_csv(args.input)
        logger.info("loaded %d rows for analysis", len(df))
        df_result = run_vader(df)
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        df_result.to_csv(out, index=False, encoding="utf-8")
        logger.info("saved analysed csv to %s", out)
    except Exception as e:
        logger.exception("analysis failed: %s", e)
        raise
