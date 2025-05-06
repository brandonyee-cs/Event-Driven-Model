# Event-Driven Model Analysis

A comprehensive event-driven analysis framework for studying market reactions to corporate events, specifically focusing on FDA approvals and earnings announcements.

## Overview

This project implements an event-driven analysis system that examines market reactions to significant corporate events. It provides tools for analyzing:
- FDA approval events
- Earnings announcements
- Volatility patterns
- Return distributions
- Sharpe ratios
- Machine learning-based predictions

## Features

- **Event Analysis Pipeline**: Automated processing of event data and stock price information
- **Volatility Analysis**: Study of volatility patterns before and after events
- **Return Analysis**: Examination of mean returns and return distributions
- **Sharpe Ratio Analysis**: Risk-adjusted performance metrics
- **Quantile Analysis**: Distribution analysis of returns and volatility
- **Machine Learning Integration**: Optional ML-based prediction capabilities
- **Visualization Tools**: Comprehensive plotting and visualization capabilities

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`:
  - pandas >= 1.5.3
  - numpy >= 1.24.3
  - polars >= 1.0.0
  - matplotlib >= 3.7.1
  - seaborn >= 0.12.2
  - scikit-learn >= 1.2.2
  - xgboost >= 1.7.6
  - pyarrow >= 12.0.0
  - plotly
  - kaleido

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd ICBS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
ICBS/
├── src/
│   ├── event_processor.py   # Event Processing
│   └── models.py            # Models
├── results/
│   ├── results_fda/         # FDA analysis results
│   └── results_earnings/    # Earnings analysis results
├── runobs.py                # Main analysis script
├── requirements.txt         # Project dependencies
└── README.md                # This file
```

## Usage

The main analysis can be run using the `runobs.py` script:

```bash
python3 runobs.py
```

The script will:
1. Process FDA approval events
2. Analyze earnings announcements
3. Generate comprehensive analysis reports
4. Create visualizations in the results directory

## Configuration

Key parameters can be adjusted in the script:
- Analysis windows
- Volatility parameters
- Sharpe ratio calculations
- Quantile analysis settings
- Machine learning parameters

## Results

Analysis results are stored in the `results/` directory, organized by event type:
- `results_fda/`: FDA approval event analysis
- `results_earnings/`: Earnings announcement analysis