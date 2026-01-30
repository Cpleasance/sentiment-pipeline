# Real-Time Sentiment Analysis Pipeline
![Python](https://img.shields.io/badge/python-3.8+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

This project provides a lightweight Python-based tool for monitoring public sentiment from customer feedback data. The solution was developed in two phases: an exploratory prototype notebook and a modular production pipeline.

> **⚠️ Important:** All files are located inside the `sentiment_pipeline_assignment` directory. You must navigate there before running any commands:
> ```bash
> cd sentiment_pipeline_assignment
> ```

## Quick Start

**Step 1:** Navigate to the project directory (if you have not done so already)
```bash
cd sentiment_pipeline_assignment
```

**Step 2:** Install dependencies (if not already installed)
```bash
pip install -r requirements.txt
```

**Step 3:** Run the pipeline
```bash
python scripts/main.py
```

Results will be saved to `output/main_outputs/` including CSV data, visualisations, and a summary report.

**Note:** Run each command separately, one at a time.

## Project Structure

```
sentiment_pipeline_assignment/
└── sentiment_pipeline_assignment/
    ├── data/
    │   ├── sample_stream.jsonl          # Small sample dataset (reccomended for testing)
    │   └── sample_stream_large.jsonl    # Larger dataset (default)
    ├── scripts/
    │   ├── prototype.ipynb              # Phase 1: Exploratory analysis
    │   ├── main.py                      # Phase 2: Pipeline orchestrator
    │   ├── ingest.py                    # Reads and streams JSONL data
    │   ├── preprocess.py                # Text cleaning and preparation
    │   ├── analyse.py                   # VADER sentiment analysis
    │   └── visualise.py                 # Plot generation
    ├── output/
    │   ├── prototype_outputs/           # Prototype results
    │   └── main_outputs/                # Pipeline results and plots
    ├── requirements.txt
    └── README.md
```


## Usage

**⚠️ Important:** All commands below assume you are in the `sentiment_pipeline_assignment` directory. If you're not there, run `cd sentiment_pipeline_assignment` first.

### Phase 1: Prototype Notebook

The prototype notebook (`scripts/prototype.ipynb`) demonstrates the exploratory development process. It processes a small sample dataset through all pipeline stages and generates visualisations.

To run:
1. Open `scripts/prototype.ipynb` in Jupyter Notebook or JupyterLab
2. Execute all cells in order
3. Results are saved to `output/prototype_outputs/`

### Phase 2: Modular Pipeline

The production pipeline can be run in two modes: batch processing or simulated streaming.

#### Basic Usage (Batch Mode)

Process all data at once:
```bash
python scripts/main.py
```

This will:
- Read data from `data/sample_stream_large.jsonl` (default)
- Preprocess and analyse the text
- Generate visualisations and a summary report
- Save outputs to `output/main_outputs/`

To use the smaller dataset:
```bash
python scripts/main.py -i data/sample_stream.jsonl
```

#### Simulated Streaming Mode

Process data in chunks to simulate real-time ingestion:
```bash
python scripts/main.py --simulate --chunk-size 10 --delay 0.5
```

This example processes 10 messages at a time with a 0.5 second delay between chunks.

#### Custom Input/Output

Specify different files and directories:
```bash
python scripts/main.py -i data/custom_data.jsonl -o output/custom_run -L logs/custom_logs
```

#### Verbose Logging

Enable detailed logging for debugging:
```bash
python scripts/main.py --verbose
```

### Running Individual Modules

Each module can be run independently. Make sure you're in the `sentiment_pipeline_assignment` directory first.

**Ingest data:**
```bash
python scripts/ingest.py -i data/sample_stream.jsonl --max 50
```

**Preprocess text:**
```bash
python scripts/preprocess.py -i data/sample_stream.jsonl -o output/processed.csv
```

**Analyse sentiment:**
```bash
python scripts/analyse.py -i output/processed.csv -o output/analysed.csv
```

**Generate visualisations:**
```bash
python scripts/visualise.py -i output/analysed.csv -o output/plots
```

## Pipeline Components

### 1. Data Ingestion (`ingest.py`)

Reads JSONL format data and validates each entry. Supports simulated streaming with configurable chunk sizes and delays.

### 2. Preprocessing (`preprocess.py`)

Cleans text data whilst preserving emotional indicators:
- Converts to lowercase
- Removes most punctuation (keeps !, ?, ')
- Filters stopwords using NLTK
- Tokenises text

The preprocessing approach was designed specifically for VADER, which relies on punctuation and capitalisation for intensity scoring.

### 3. Sentiment Analysis (`analyse.py`)

Uses VADER to score sentiment. Each message receives:
- Negative, neutral, and positive scores
- Compound score (-1 to +1)
- Sentiment label (Positive/Neutral/Negative)

Thresholds:
- Positive: compound ≥ 0.05
- Negative: compound ≤ -0.05
- Neutral: -0.05 < compound < 0.05

### 4. Visualisation (`visualise.py`)

Generates four visualisations:
- **Line plot**: Compound sentiment over time
- **Count plot**: Distribution of sentiment labels
- **Bar chart**: Average scores by sentiment category
- **Pie chart**: Proportion of each sentiment

## Outputs

The pipeline generates:

1. **sentiment_results.csv** - Full analysis results with all scores
2. **summary_report.txt** - Text summary with key metrics
3. **Visualisation PNGs**:
   - `sentiment_overtime.png`
   - `sentiment_distribution_count.png`
   - `average_sentiment_scores.png`
   - `sentiment_distribution_pie.png`
4. **pipeline.log** - Execution logs

## Design Decisions

### Why VADER?

VADER was chosen because it:
- Works well on social media text
- Handles emoticons and punctuation
- Doesn't require training data
- Provides interpretable scores

### Why was punctuation preserved?

Exclamation marks and question marks carry emotional information that VADER uses for intensity scoring. Removing them would reduce accuracy.

### Modular structure

Each module can run independently, making the pipeline:
- Easier to test and debug
- Reusable for different datasets
- Adaptable for future development

## Troubleshooting

**"can't open file" or "No such file or directory" errors:**
- Make sure you're in the `sentiment_pipeline_assignment` directory when running commands
- Check your current location with `pwd` (Linux) or `cd` (Windows)
- Navigate to the correct directory: `cd sentiment_pipeline_assignment`

**"File not found" errors for data files:**
- Check that file paths are correct relative to the `sentiment_pipeline_assignment` directory
- Verify data files exist: `ls data/` or `dir data/`
- The default input file is `sample_stream_large.jsonl` - use `-i data/sample_stream.jsonl` for the smaller file (great for testing)

**NLTK download errors (must be online to work):**
- The required NLTK data downloads automatically
- If downloads fail, manually run: `python -m nltk.downloader stopwords vader_lexicon`

**Memory issues with large files:**
- Use simulated streaming mode with smaller chunk sizes
- Process data in batches using the individual modules

## Requirements

- Python 3.8+
- pandas >= 2.0
- numpy >= 1.25
- matplotlib >= 3.8
- seaborn >= 0.12
- nltk >= 3.9.2

## Notes

- The pipeline processes data offline and doesn't require internet access (after installation)
- Log files accumulate over time - consider clearing the logs directory periodically

## Inspiration for this README.md file:
https://github.com/matiassingers/awesome-readme?tab=readme-ov-file

