# Duration-Driven Returns: Event Study Framework

This repository contains the implementation code for the paper "Modeling Equilibrium Asset Pricing Around Events with Heterogeneous Beliefs, Dynamic Volatility, and a Two-Risk Uncertainty Framework" by Brandon Yee.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project provides empirical validation of an asset pricing model designed to analyze investor behavior around high-uncertainty events such as earnings announcements and FDA approvals. The framework distinguishes between two types of risk (directional news risk and impact uncertainty) and employs a three-phase volatility model to characterize price dynamics around events.

### Key Features

- **Two-Risk Framework**: Distinguishes between directional news risk (uncertainty about event outcomes) and impact uncertainty (uncertainty about market response magnitude)
- **Three-Phase Volatility Model**: Models volatility dynamics through pre-event, event, and post-event phases
- **GARCH Implementation**: Uses both standard GARCH and GJR-GARCH models for baseline volatility estimation
- **Return-to-Variance Ratio (RVR) Analysis**: Tests hypotheses about RVR dynamics around events
- **Event Analysis Pipeline**: Comprehensive pipeline for analyzing earnings announcements and FDA approvals

## Repository Structure

- `src/`: Core implementation code
  - `event_processor.py`: Main event processing and analysis pipeline
  - `models.py`: GARCH, GJR-GARCH, and other statistical models
- `*.py`: Analysis scripts
  - `runobs.py`: Runs observational analysis on event data
  - `runrvr.py`: Runs Return-to-Variance Ratio analysis
  - `testhyp1.py`: Tests Hypothesis 1 about RVR dynamics
  - `testhyp2.py`: Tests Hypothesis 2 about volatility innovations
- `results/`: Directory where analysis results are stored (created on first run)
  - `hypothesis1/`: Results for tests of Hypothesis 1
  - `hypothesis2/`: Results for tests of Hypothesis 2
  - `rvr_improved/`: Results for improved RVR analysis
  - `obs/`: Results for observational analysis

## Installation

1. Clone the repository:
```bash
git clone https://github.com/brandonyee-cs/Event-Driven-Model.git
cd Event-Driven-Model
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Requirements

The code requires the following packages:
- pandas >= 1.5.3
- numpy >= 1.24.3
- polars >= 1.0.0, < 2.0.0
- matplotlib >= 3.7.1
- seaborn >= 0.12.2
- scikit-learn >= 1.2.2, < 1.4.0
- xgboost >= 1.7.6, < 2.0.0
- statsmodels
- pyarrow >= 12.0.0

## Usage

### Basic Analysis

To run the basic observational analysis:

```bash
python runobs.py
```

This will analyze both earnings announcements and FDA approvals, generating visualizations of volatility patterns, returns, and Sharpe ratios around events.

### RVR Analysis

To run the Return-to-Variance Ratio analysis:

```bash
python runrvr.py
```

This implements the improved RVR methodology described in the paper.

### Hypothesis Testing

To test specific hypotheses from the paper:

```bash
python testhyp1.py  # Tests Hypothesis 1: RVR peaks during post-event rising phase
python testhyp2.py  # Tests Hypothesis 2: Volatility innovations predict returns
```

### Customizing Analysis

You can modify the analysis parameters in each script file:
- `WINDOW_DAYS`: Number of days to analyze before/after events
- `ANALYSIS_WINDOW`: Range of days (relative to event) for analysis
- `GARCH_TYPE`: Type of GARCH model ('garch' or 'gjr')
- `OPTIMISTIC_BIAS`: Bias parameter for post-event expected returns

## Key Hypotheses Tested

1. **Hypothesis 1**: Return-to-Variance Ratio peaks during the post-event rising phase due to high volatility and biased expectations.
2. **Hypothesis 2**: GARCH-estimated conditional volatility innovations serve as an effective proxy for impact uncertainty.
   - **H2.1**: Pre-event volatility innovations predict subsequent returns
   - **H2.2**: Post-event volatility persistence extends elevated returns
   - **H2.3**: Asymmetric volatility response correlates with price adjustment

## Data Sources

The code is designed to work with:
- Earnings announcements from I/B/E/S database
- FDA approvals from public records
- Stock price data from CRSP Daily Stock File

Note: The repository does not include the raw data files due to licensing restrictions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```
@article{yee2025modeling,
  title={Modeling Equilibrium Asset Pricing Around Events with Heterogeneous Beliefs, Dynamic Volatility, and a Two-Risk Uncertainty Framework},
  author={Yee, Brandon},
  year={2025}
}
```

## Contact

Brandon Yee - brandonyee.nyc@gmail.com
