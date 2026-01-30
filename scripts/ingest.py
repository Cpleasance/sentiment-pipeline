# ================================================================
#   ingest.py
# ================================================================

import json
import time
from pathlib import Path
from typing import Generator, Dict
import logging

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
#   Read and stream .jsonl
#               Sources:
#               https://docs.python.org/3/library/json.html
#               https://realpython.com/python-generators/
# ================================================================
def read_jsonl(path: Path) -> Generator[Dict, None, None]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    logger = logging.getLogger(__name__)
    valid_count = 0
    # Parse the json file. If broken warn, skip and move!
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping invalid JSON on line %d", i)
                continue

            if not obj.get("text") or not obj.get("timestamp"):
                logger.warning("Skipping JSON on line %d: missing 'text' or 'timestamp'", i)
                continue

            valid_count += 1
            yield obj

    if valid_count == 0:
        logger.warning("No valid records found in file: %s", path)

    # Allow for live data
def stream_jsonl(path: Path, chunk_size: int = 1, delay: float = 0.0) -> Generator[Dict, None, None]:
    buf = []
    for obj in read_jsonl(path):
        buf.append(obj)
        if len(buf) >= chunk_size:
            for o in buf:
                yield o
            buf = []
            if delay > 0:
                time.sleep(delay)
    for o in buf:
        yield o

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
    DEFAULT_INPUT = project_root / "data" / "sample_stream_large.jsonl"
    DEFAULT_OUTDIR = project_root / "output" / "main_outputs"
    DEFAULT_LOGDIR = project_root / "output" / "main_logging"

    parser = argparse.ArgumentParser(description="Ingest jsonl")
    parser.add_argument("-i", "--input", default=str(DEFAULT_INPUT), help="Input jsonl file's path")
    parser.add_argument("-l", "--outdir", default=str(DEFAULT_OUTDIR), help="Output directory for outputs")
    parser.add_argument("-L", "--logdir", default=str(DEFAULT_LOGDIR), help="Log directory")
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--max", type=int, default=None, help="Max number of lines to read")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Standard info or verbose logs?
    logging_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(Path(args.outdir), logs_dir=Path(args.logdir), level=logging_level)
    logger = logging.getLogger("ingest")

    # Streaming jsonl files and if any issues log.
    count = 0
    try:
        for obj in stream_jsonl(Path(args.input), chunk_size=args.chunk_size, delay=args.delay):
            logger.info("ingested: %s", obj)
            count += 1
            if args.max and count >= args.max:
                break
        logger.info("finished ingesting %d messages", count)
    except Exception as e:
        logger.exception("failed to ingest: %s", e)
        raise
