#!/bin/bash
#
# YAML Parser for Shell Scripts
# This utility script parses config.yaml and exports variables
#

# Check if the necessary tools are available
if ! command -v python &> /dev/null; then
    echo "Error: Python not found, required for YAML parsing"
    exit 1
fi

# Function to load YAML file and convert to shell variables
parse_yaml() {
    local yaml_file=$1
    local prefix=$2
    
    # Create a Python script to parse YAML and output shell variable assignments
    python3 -c "
import sys
import yaml
import os

# Try to load PyYAML
try:
    from yaml import safe_load
except ImportError:
    print('Error: PyYAML module not found. Install with: pip install pyyaml')
    sys.exit(1)

# Load YAML file
try:
    with open('$yaml_file', 'r') as file:
        config = safe_load(file)
except Exception as e:
    print(f'Error: Failed to load YAML file: {e}')
    sys.exit(1)

# Function to flatten the YAML structure into shell variables
def flatten(dictionary, parent_key='', sep='_'):
    items = []
    for key, value in dictionary.items():
        new_key = f'{parent_key}{sep}{key}' if parent_key else key
        
        if isinstance(value, dict):
            items.extend(flatten(value, new_key, sep=sep).items())
        elif isinstance(value, list):
            # For lists, create indexed variables and a count variable
            items.append((f'{new_key}_count', str(len(value))))
            for i, item in enumerate(value):
                items.append((f'{new_key}_{i}', str(item)))
        else:
            items.append((new_key, str(value)))
    return dict(items)

# Convert to shell variables with the given prefix
flat_config = flatten(config)
for key, value in flat_config.items():
    # Escape single quotes in values
    value = value.replace(\"'\", \"'\\\\''\")\
              .replace('\\\$(', '\\\\\\\$(') \
              .replace('`', '\\\\`')
    
    # Print variable assignment
    print(f\"{prefix}{key}='{value}'\")
"
}

# Parse the YAML file and export variables
if [ -f "config.yaml" ]; then
    eval "$(parse_yaml config.yaml "CONFIG_")"
    export $(parse_yaml config.yaml "CONFIG_" | sed 's/=.*//')
else
    echo "Error: config.yaml not found"
    exit 1
fi

# Function to get a stock file by index
get_stock_file() {
    local index=$1
    eval echo "\$CONFIG_paths_stock_files_$index"
}

# Function to build a space-separated list of stock files
get_stock_files_str() {
    local count=${CONFIG_paths_stock_files_count}
    local files=""
    for ((i=0; i<count; i++)); do
        local file=$(get_stock_file $i)
        files="$files $file"
    done
    echo $files
}

# Export the stock files as a space-separated string
export STOCK_FILES_STR=$(get_stock_files_str)

# Function to configure all parameters for a specific event type
# Usage: configure_event_params EVENT_TYPE
configure_event_params() {
    local event_type=$1
    
    # Get event-specific settings
    eval export EVENT_FILE="\$CONFIG_paths_${event_type}_event_file"
    eval export RESULTS_DIR="\$CONFIG_paths_results_${event_type}"
    eval export EVENT_DATE_COL="\$CONFIG_events_${event_type}_event_date_col"
    eval export TICKER_COL="\$CONFIG_events_${event_type}_ticker_col"
    eval export FILE_PREFIX="\$CONFIG_events_${event_type}_file_prefix"
    
    # Get parameters from the shared analysis section
    export WINDOW_DAYS="${CONFIG_analysis_window_days}"
    export VOL_WINDOW="${CONFIG_analysis_volatility_window}"
    export VOL_PRE_DAYS="${CONFIG_analysis_volatility_pre_days}"
    export VOL_POST_DAYS="${CONFIG_analysis_volatility_post_days}"
    export VOL_BASELINE_START="${CONFIG_analysis_volatility_baseline_window_start}"
    export VOL_BASELINE_END="${CONFIG_analysis_volatility_baseline_window_end}"
    export VOL_EVENT_START="${CONFIG_analysis_volatility_event_window_start}"
    export VOL_EVENT_END="${CONFIG_analysis_volatility_event_window_end}"
    export SHARPE_WINDOW="${CONFIG_analysis_sharpe_window}"
    export SHARPE_ANALYSIS_START="${CONFIG_analysis_sharpe_analysis_window_start}"
    export SHARPE_ANALYSIS_END="${CONFIG_analysis_sharpe_analysis_window_end}"
    export SHARPE_LOOKBACK="${CONFIG_analysis_sharpe_lookback}"
    export ML_WINDOW="${CONFIG_analysis_ml_window}"
    export RUN_ML="${CONFIG_analysis_ml_run}"
    export ML_TEST_SPLIT="${CONFIG_analysis_ml_test_split}"
}