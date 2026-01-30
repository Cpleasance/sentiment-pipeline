# ================================================================
#   preprocess.py
# ================================================================

import json
import string
from pathlib import Path
import logging
import pandas as pd
import nltk
from nltk.corpus import stopwords

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
#   NLTK: ensure stopwords is there
# ================================================================
def ensure_nltk_stopwords():
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")

# ================================================================
#   Preprocessing for VADER
# ================================================================
def preprocess_text_vader(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    punct_to_remove = ''.join([p for p in string.punctuation if p not in ["!", "?", "'"]])
    text = text.translate(str.maketrans("", "", punct_to_remove))
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords.words("english")]
    return " ".join(tokens)

def df_preprocess(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    ensure_nltk_stopwords()
    df = df.copy()
    if text_col not in df.columns:
        logging.getLogger(__name__).error("text column missing: %s", text_col)
        raise KeyError(text_col)
    df[text_col] = df[text_col].fillna("").astype(str)
    df["processed_text"] = df[text_col].apply(preprocess_text_vader)
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
    DEFAULT_INPUT = project_root / "data" / "sample_stream.jsonl"
    DEFAULT_OUTDIR = project_root / "output" / "main_outputs"
    DEFAULT_LOGDIR = project_root / "output" / "main_logging"

    parser = argparse.ArgumentParser(description="Preprocess text data")
    parser.add_argument("-i", "--input", default=str(DEFAULT_INPUT), help="Path to input file")
    parser.add_argument("--input-format", choices=["jsonl", "csv"], default="jsonl", help="Input format")
    parser.add_argument("-o", "--output", default=str(Path(DEFAULT_OUTDIR) / "processed.csv"), help="Path for processed data")
    parser.add_argument("-l", "--outdir", default=str(DEFAULT_OUTDIR), help="Output directory for outputs")
    parser.add_argument("-L", "--logdir", default=str(DEFAULT_LOGDIR), help="Log directory")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Standard info or verbose logs?
    logging_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(Path(args.outdir), logs_dir=Path(args.logdir), level=logging_level)
    logger = logging.getLogger("preprocess")

    try:
        if args.input_format == "jsonl":
            rows = []
            with Path(args.input).open("r", encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        logger.warning("skipping invalid json line %d", i)
                        continue
            df = pd.DataFrame(rows)
        else:
            df = pd.read_csv(args.input)
        logger.info("loaded %d rows for preprocessing", len(df))
        processed = df_preprocess(df)
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        processed.to_csv(out, index=False, encoding="utf-8")
        logger.info("saved processed csv to %s", out)
    except Exception as e:
        logger.exception("preprocess failed: %s", e)
        raise
