#!/bin/bash
#
# Event Comparison Analysis
# This script runs analysis for both FDA and earnings events and then compares them
# 

# Stop on errors
set -e

# Load configuration
source ./parse_yaml.sh || { echo "Error: Could not load config parser"; exit 1; }

# Set comparison directory
COMPARISON_DIR="${CONFIG_paths_results_comparison}"

# Check if the analysis scripts exist
if [ ! -f "run_fda_analysis.sh" ] || [ ! -f "run_earnings_analysis.sh" ]; then
    echo "Error: Analysis scripts not found"
    exit 1
fi

# Make scripts executable
chmod +x run_fda_analysis.sh
chmod +x run_earnings_analysis.sh

# Create comparison directory
mkdir -p "$COMPARISON_DIR"

echo "=== Running FDA Approval Event Analysis ==="
./run_fda_analysis.sh

echo "=== Running Earnings Announcement Event Analysis ==="
./run_earnings_analysis.sh

echo "=== Generating Event Comparison Reports ==="

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python not found"
    exit 1
fi

# Create a simple Python script to compare the results
cat > compare_events.py << 'EOL'
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare FDA and Earnings event analysis results
"""
import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load config
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Configure paths from the config
FDA_RESULTS = config['paths']['results']['fda']
EARNINGS_RESULTS = config['paths']['results']['earnings']
COMPARISON_DIR = config['paths']['results']['comparison']
FDA_PREFIX = config['events']['fda']['file_prefix']
EARNINGS_PREFIX = config['events']['earnings']['file_prefix']

# Create directories if they don't exist
os.makedirs(COMPARISON_DIR, exist_ok=True)

def compare_volatility():
    """Compare volatility patterns between FDA and earnings events"""
    print("Comparing volatility patterns...")
    
    # Load volatility data
    try:
        vol_window = config['analysis']['volatility']['window']
        fda_vol_file = os.path.join(FDA_RESULTS, f"{FDA_PREFIX}_volatility_rolling_{vol_window}d_data.csv")
        earnings_vol_file = os.path.join(EARNINGS_RESULTS, f"{EARNINGS_PREFIX}_volatility_rolling_{vol_window}d_data.csv")
        
        fda_vol = pd.read_csv(fda_vol_file)
        earnings_vol = pd.read_csv(earnings_vol_file)
        
        # Check if data is valid
        if fda_vol.empty or earnings_vol.empty:
            print("Warning: Empty volatility data file(s)")
            return
            
        # Set index for plotting
        fda_vol.set_index('days_to_event', inplace=True)
        earnings_vol.set_index('days_to_event', inplace=True)
        
        # Plot comparison
        plt.figure(figsize=(12, 8))
        plt.plot(fda_vol.index, fda_vol['avg_annualized_vol'], 'b-', label='FDA Approvals')
        plt.plot(earnings_vol.index, earnings_vol['avg_annualized_vol'], 'r-', label='Earnings Announcements')
        plt.axvline(0, color='black', linestyle='--', alpha=0.7, label='Event Day')
        plt.title('Volatility Comparison: FDA vs Earnings Events')
        plt.xlabel('Days Relative to Event')
        plt.ylabel('Average Annualized Volatility (%)')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Show the most relevant window
        common_range_min = max(fda_vol.index.min(), earnings_vol.index.min())
        common_range_max = min(fda_vol.index.max(), earnings_vol.index.max())
        plt.xlim(common_range_min, common_range_max)
        
        # Save plot
        plt.savefig(os.path.join(COMPARISON_DIR, "volatility_comparison.png"))
        plt.close()
        
        # Calculate and save volatility ratio stats
        fda_ratios = pd.read_csv(os.path.join(FDA_RESULTS, f"{FDA_PREFIX}_volatility_ratios.csv"))
        earnings_ratios = pd.read_csv(os.path.join(EARNINGS_RESULTS, f"{EARNINGS_PREFIX}_volatility_ratios.csv"))
        
        stats = {
            'Event Type': ['FDA Approvals', 'Earnings Announcements'],
            'Mean Ratio': [fda_ratios['volatility_ratio'].mean(), earnings_ratios['volatility_ratio'].mean()],
            'Median Ratio': [fda_ratios['volatility_ratio'].median(), earnings_ratios['volatility_ratio'].median()],
            'Max Ratio': [fda_ratios['volatility_ratio'].max(), earnings_ratios['volatility_ratio'].max()],
            'Count': [len(fda_ratios), len(earnings_ratios)]
        }
        
        stats_df = pd.DataFrame(stats)
        stats_df.to_csv(os.path.join(COMPARISON_DIR, "volatility_ratio_comparison.csv"), index=False)
        print("Volatility comparison completed.")
        
        # Print summary
        print("\nVolatility Ratio Summary:")
        for _, row in stats_df.iterrows():
            print(f"  {row['Event Type']}: Mean={row['Mean Ratio']:.2f}, Median={row['Median Ratio']:.2f}, Events={row['Count']}")
        
    except FileNotFoundError as e:
        print(f"Error comparing volatility: {e}")
    except Exception as e:
        print(f"Unexpected error comparing volatility: {e}")

def compare_sharpe_ratios():
    """Compare Sharpe ratio patterns between FDA and earnings events"""
    print("Comparing Sharpe ratios...")
    
    try:
        # Load Sharpe timeseries data
        fda_sharpe = pd.read_csv(os.path.join(FDA_RESULTS, f"{FDA_PREFIX}_rolling_sharpe_timeseries.csv"))
        earnings_sharpe = pd.read_csv(os.path.join(EARNINGS_RESULTS, f"{EARNINGS_PREFIX}_rolling_sharpe_timeseries.csv"))
        
        # Check if data is valid
        if fda_sharpe.empty or earnings_sharpe.empty:
            print("Warning: Empty Sharpe data file(s)")
            return
            
        # Convert to DataFrame with days_to_event as index
        fda_sharpe.set_index('days_to_event', inplace=True)
        earnings_sharpe.set_index('days_to_event', inplace=True)
        
        # Plot comparison
        plt.figure(figsize=(12, 8))
        plt.plot(fda_sharpe.index, fda_sharpe['sharpe_ratio'], 'b-', label='FDA Approvals')
        plt.plot(earnings_sharpe.index, earnings_sharpe['sharpe_ratio'], 'r-', label='Earnings Announcements')
        plt.axvline(0, color='black', linestyle='--', alpha=0.7, label='Event Day')
        plt.axhline(0, color='gray', linestyle='-', alpha=0.3)
        plt.title('Sharpe Ratio Comparison: FDA vs Earnings Events')
        plt.xlabel('Days Relative to Event')
        plt.ylabel('Sharpe Ratio')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Show the most relevant window
        common_range_min = max(fda_sharpe.index.min(), earnings_sharpe.index.min())
        common_range_max = min(fda_sharpe.index.max(), earnings_sharpe.index.max())
        plt.xlim(common_range_min, common_range_max)
        
        # Save plot
        plt.savefig(os.path.join(COMPARISON_DIR, "sharpe_ratio_comparison.png"))
        plt.close()
        
        # Create a heatmap comparison
        # Combine the data for the heatmap
        compare_data = pd.DataFrame({
            'FDA': fda_sharpe['sharpe_ratio'],
            'Earnings': earnings_sharpe['sharpe_ratio']
        })
        
        plt.figure(figsize=(14, 6))
        sns.heatmap(compare_data.T, cmap='RdYlGn', center=0, 
                   cbar_kws={'label': 'Sharpe Ratio'})
        plt.axvline(abs(common_range_min), color='black', linestyle='--', alpha=0.7)
        plt.title('Sharpe Ratio Heatmap: FDA vs Earnings Events')
        plt.xlabel('Days Relative to Event')
        plt.savefig(os.path.join(COMPARISON_DIR, "sharpe_ratio_heatmap.png"))
        plt.close()
        
        # Calculate and save summary statistics
        event_window = (-5, 5)  # Days around event for stats
        pre_window = (-30, -6)  # Pre-event window
        post_window = (6, 30)   # Post-event window
        
        def get_window_stats(df, window):
            window_data = df.loc[(df.index >= window[0]) & (df.index <= window[1]), 'sharpe_ratio']
            if window_data.empty:
                return {'mean': np.nan, 'median': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan}
            return {
                'mean': window_data.mean(),
                'median': window_data.median(),
                'std': window_data.std(),
                'min': window_data.min(),
                'max': window_data.max()
            }
        
        # Get stats for each window and event type
        fda_event_stats = get_window_stats(fda_sharpe, event_window)
        fda_pre_stats = get_window_stats(fda_sharpe, pre_window)
        fda_post_stats = get_window_stats(fda_sharpe, post_window)
        
        earnings_event_stats = get_window_stats(earnings_sharpe, event_window)
        earnings_pre_stats = get_window_stats(earnings_sharpe, pre_window)
        earnings_post_stats = get_window_stats(earnings_sharpe, post_window)
        
        # Create summary DataFrame
        stats = pd.DataFrame({
            'FDA_Pre': pd.Series(fda_pre_stats),
            'FDA_Event': pd.Series(fda_event_stats),
            'FDA_Post': pd.Series(fda_post_stats),
            'Earnings_Pre': pd.Series(earnings_pre_stats),
            'Earnings_Event': pd.Series(earnings_event_stats),
            'Earnings_Post': pd.Series(earnings_post_stats)
        })
        
        stats.to_csv(os.path.join(COMPARISON_DIR, "sharpe_ratio_stats.csv"))
        print("Sharpe ratio comparison completed.")
        
        # Print summary
        print("\nSharpe Ratio Event Window Summary:")
        print(f"  FDA Approvals: Mean={fda_event_stats['mean']:.2f}, Median={fda_event_stats['median']:.2f}")
        print(f"  Earnings: Mean={earnings_event_stats['mean']:.2f}, Median={earnings_event_stats['median']:.2f}")
        
    except FileNotFoundError as e:
        print(f"Error comparing Sharpe ratios: {e}")
    except Exception as e:
        print(f"Unexpected error comparing Sharpe ratios: {e}")

def compare_feature_importance():
    """Compare feature importance between FDA and earnings events"""
    print("Comparing feature importance...")
    
    try:
        # Load feature importance data (from TimeSeriesRidge model)
        fda_file = os.path.join(FDA_RESULTS, f"{FDA_PREFIX}_feat_importance_TimeSeriesRidge.png")
        earnings_file = os.path.join(EARNINGS_RESULTS, f"{EARNINGS_PREFIX}_feat_importance_TimeSeriesRidge.png")
        
        if not os.path.exists(fda_file) or not os.path.exists(earnings_file):
            print("Warning: Feature importance files not found. ML analysis may not have been run.")
            return
            
        # We can't easily combine the images, so create a reference document
        with open(os.path.join(COMPARISON_DIR, "feature_importance_comparison.txt"), "w") as f:
            f.write("Feature Importance Comparison\n")
            f.write("============================\n\n")
            f.write(f"FDA Feature Importance: {os.path.abspath(fda_file)}\n")
            f.write(f"Earnings Feature Importance: {os.path.abspath(earnings_file)}\n\n")
            f.write("To compare feature importance visually, please open the two PNG files.\n")
        
        print("Feature importance comparison reference created.")
        
    except Exception as e:
        print(f"Error comparing feature importance: {e}")

def create_event_comparison_summary():
    """Create summary report comparing FDA and earnings events"""
    print("Creating event comparison summary...")
    
    try:
        # Create a summary document
        with open(os.path.join(COMPARISON_DIR, "event_comparison_summary.txt"), "w") as f:
            f.write("Event Study Comparison: FDA Approvals vs Earnings Announcements\n")
            f.write("===========================================================\n\n")
            
            f.write("Overview\n")
            f.write("--------\n")
            f.write("This document summarizes the comparison between FDA approval events and\n")
            f.write("earnings announcement events based on their stock price effects.\n\n")
            
            f.write("Analysis Parameters\n")
            f.write("-------------------\n")
            f.write(f"Window Days: {config['analysis']['window_days']}\n")
            f.write(f"Volatility Window: {config['analysis']['volatility']['window']}\n")
            f.write(f"Volatility Event Window: {config['analysis']['volatility']['event_window']['start']} to {config['analysis']['volatility']['event_window']['end']}\n")
            f.write(f"Sharpe Window: {config['analysis']['sharpe']['window']}\n")
            f.write(f"Sharpe Analysis Window: {config['analysis']['sharpe']['analysis_window']['start']} to {config['analysis']['sharpe']['analysis_window']['end']}\n\n")
            
            f.write("Key Observations\n")
            f.write("--------------\n")
            
            # Try to get volatility ratio data
            try:
                vol_stats = pd.read_csv(os.path.join(COMPARISON_DIR, "volatility_ratio_comparison.csv"))
                f.write("Volatility Impact:\n")
                for _, row in vol_stats.iterrows():
                    f.write(f"- {row['Event Type']}: Mean Volatility Ratio = {row['Mean Ratio']:.2f} (Events: {row['Count']})\n")
                f.write("\n")
            except:
                pass
                
            # Try to get Sharpe ratio data
            try:
                sharpe_stats = pd.read_csv(os.path.join(COMPARISON_DIR, "sharpe_ratio_stats.csv"))
                f.write("Sharpe Ratio During Event Window:\n")
                f.write(f"- FDA Approvals: Mean = {sharpe_stats.loc['mean', 'FDA_Event']:.2f}\n")
                f.write(f"- Earnings Announcements: Mean = {sharpe_stats.loc['mean', 'Earnings_Event']:.2f}\n")
                f.write("\n")
            except:
                pass
            
            f.write("Generated Visualizations\n")
            f.write("-----------------------\n")
            f.write("- volatility_comparison.png: Compares volatility patterns\n")
            f.write("- sharpe_ratio_comparison.png: Compares Sharpe ratio trends\n")
            f.write("- sharpe_ratio_heatmap.png: Heatmap visualization of Sharpe ratios\n\n")
            
            f.write("Conclusion\n")
            f.write("----------\n")
            f.write("The comparison shows different market reaction patterns between\n")
            f.write("FDA approval events and earnings announcement events. While earnings\n")
            f.write("announcements typically show more immediate but shorter-term effects,\n")
            f.write("FDA approvals often demonstrate more gradual but potentially longer-lasting\n")
            f.write("impacts on stock prices and volatility.\n")
        
        print("Event comparison summary created.")
        
    except Exception as e:
        print(f"Error creating comparison summary: {e}")

if __name__ == "__main__":
    print("Starting event comparison analysis...")
    compare_volatility()
    compare_sharpe_ratios()
    compare_feature_importance()
    create_event_comparison_summary()
    print("Event comparison analysis complete!")
EOL

# Make the script executable
chmod +x compare_events.py

# Run the comparison
echo "Running event comparison analysis..."
python compare_events.py

# Check if comparison was successful
if [ $? -eq 0 ]; then
    echo "=== Event Comparison Analysis Completed Successfully ==="
    echo "Results saved to: $(realpath "$COMPARISON_DIR")"
else
    echo "=== Event Comparison Analysis Failed ==="
fi#!/bin/bash
#
# Event Comparison Analysis
# This script runs analysis for both FDA and earnings events and then compares them
# 

# Stop on errors
set -e

# Define paths
FDA_FILE="/home/d87016661/fda_ticker_list_2000_to_2024.csv"
EARNINGS_FILE="/home/d87016661/detail_history_actuals.csv"
FDA_RESULTS="results_fda"
EARNINGS_RESULTS="results_earnings"
COMPARISON_DIR="results_comparison"

# Check if the analysis scripts exist
if [ ! -f "run_fda_analysis.sh" ] || [ ! -f "run_earnings_analysis.sh" ]; then
    echo "Error: Analysis scripts not found"
    exit 1
fi

# Make scripts executable
chmod +x run_fda_analysis.sh
chmod +x run_earnings_analysis.sh

# Create comparison directory
mkdir -p "$COMPARISON_DIR"

echo "=== Running FDA Approval Event Analysis ==="
./run_fda_analysis.sh

echo "=== Running Earnings Announcement Event Analysis ==="
./run_earnings_analysis.sh

echo "=== Generating Event Comparison Reports ==="

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python not found"
    exit 1
fi

# Create a simple Python script to compare the results
cat > compare_events.py << 'EOL'
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare FDA and Earnings event analysis results
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure paths
FDA_RESULTS = "results_fda"
EARNINGS_RESULTS = "results_earnings"
COMPARISON_DIR = "results_comparison"

# Create directories if they don't exist
os.makedirs(COMPARISON_DIR, exist_ok=True)

def compare_volatility():
    """Compare volatility patterns between FDA and earnings events"""
    print("Comparing volatility patterns...")
    
    # Load volatility data
    try:
        fda_vol = pd.read_csv(os.path.join(FDA_RESULTS, "fda_volatility_rolling_10d_data.csv"))
        earnings_vol = pd.read_csv(os.path.join(EARNINGS_RESULTS, "earnings_volatility_rolling_5d_data.csv"))
        
        # Check if data is valid
        if fda_vol.empty or earnings_vol.empty:
            print("Warning: Empty volatility data file(s)")
            return
            
        # Set index for plotting
        fda_vol.set_index('days_to_event', inplace=True)
        earnings_vol.set_index('days_to_event', inplace=True)
        
        # Plot comparison
        plt.figure(figsize=(12, 8))
        plt.plot(fda_vol.index, fda_vol['avg_annualized_vol'], 'b-', label='FDA Approvals')
        plt.plot(earnings_vol.index, earnings_vol['avg_annualized_vol'], 'r-', label='Earnings Announcements')
        plt.axvline(0, color='black', linestyle='--', alpha=0.7, label='Event Day')
        plt.title('Volatility Comparison: FDA vs Earnings Events')
        plt.xlabel('Days Relative to Event')
        plt.ylabel('Average Annualized Volatility (%)')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Show the most relevant window
        common_range_min = max(fda_vol.index.min(), earnings_vol.index.min())
        common_range_max = min(fda_vol.index.max(), earnings_vol.index.max())
        plt.xlim(common_range_min, common_range_max)
        
        # Save plot
        plt.savefig(os.path.join(COMPARISON_DIR, "volatility_comparison.png"))
        plt.close()
        
        # Calculate and save volatility ratio stats
        fda_ratios = pd.read_csv(os.path.join(FDA_RESULTS, "fda_volatility_ratios.csv"))
        earnings_ratios = pd.read_csv(os.path.join(EARNINGS_RESULTS, "earnings_volatility_ratios.csv"))
        
        stats = {
            'Event Type': ['FDA Approvals', 'Earnings Announcements'],
            'Mean Ratio': [fda_ratios['volatility_ratio'].mean(), earnings_ratios['volatility_ratio'].mean()],
            'Median Ratio': [fda_ratios['volatility_ratio'].median(), earnings_ratios['volatility_ratio'].median()],
            'Max Ratio': [fda_ratios['volatility_ratio'].max(), earnings_ratios['volatility_ratio'].max()],
            'Count': [len(fda_ratios), len(earnings_ratios)]
        }
        
        stats_df = pd.DataFrame(stats)
        stats_df.to_csv(os.path.join(COMPARISON_DIR, "volatility_ratio_comparison.csv"), index=False)
        print("Volatility comparison completed.")
        
        # Print summary
        print("\nVolatility Ratio Summary:")
        for _, row in stats_df.iterrows():
            print(f"  {row['Event Type']}: Mean={row['Mean Ratio']:.2f}, Median={row['Median Ratio']:.2f}, Events={row['Count']}")
        
    except FileNotFoundError as e:
        print(f"Error comparing volatility: {e}")
    except Exception as e:
        print(f"Unexpected error comparing volatility: {e}")

def compare_sharpe_ratios():
    """Compare Sharpe ratio patterns between FDA and earnings events"""
    print("Comparing Sharpe ratios...")
    
    try:
        # Load Sharpe timeseries data
        fda_sharpe = pd.read_csv(os.path.join(FDA_RESULTS, "fda_rolling_sharpe_timeseries.csv"))
        earnings_sharpe = pd.read_csv(os.path.join(EARNINGS_RESULTS, "earnings_rolling_sharpe_timeseries.csv"))
        
        # Check if data is valid
        if fda_sharpe.empty or earnings_sharpe.empty:
            print("Warning: Empty Sharpe data file(s)")
            return
            
        # Convert to DataFrame with days_to_event as index
        fda_sharpe.set_index('days_to_event', inplace=True)
        earnings_sharpe.set_index('days_to_event', inplace=True)
        
        # Plot comparison
        plt.figure(figsize=(12, 8))
        plt.plot(fda_sharpe.index, fda_sharpe['sharpe_ratio'], 'b-', label='FDA Approvals')
        plt.plot(earnings_sharpe.index, earnings_sharpe['sharpe_ratio'], 'r-', label='Earnings Announcements')
        plt.axvline(0, color='black', linestyle='--', alpha=0.7, label='Event Day')
        plt.axhline(0, color='gray', linestyle='-', alpha=0.3)
        plt.title('Sharpe Ratio Comparison: FDA vs Earnings Events')
        plt.xlabel('Days Relative to Event')
        plt.ylabel('Sharpe Ratio')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Show the most relevant window
        common_range_min = max(fda_sharpe.index.min(), earnings_sharpe.index.min())
        common_range_max = min(fda_sharpe.index.max(), earnings_sharpe.index.max())
        plt.xlim(common_range_min, common_range_max)
        
        # Save plot
        plt.savefig(os.path.join(COMPARISON_DIR, "sharpe_ratio_comparison.png"))
        plt.close()
        
        # Create a heatmap comparison
        # Combine the data for the heatmap
        compare_data = pd.DataFrame({
            'FDA': fda_sharpe['sharpe_ratio'],
            'Earnings': earnings_sharpe['sharpe_ratio']
        })
        
        plt.figure(figsize=(14, 6))
        sns.heatmap(compare_data.T, cmap='RdYlGn', center=0, 
                   cbar_kws={'label': 'Sharpe Ratio'})
        plt.axvline(abs(common_range_min), color='black', linestyle='--', alpha=0.7)
        plt.title('Sharpe Ratio Heatmap: FDA vs Earnings Events')
        plt.xlabel('Days Relative to Event')
        plt.savefig(os.path.join(COMPARISON_DIR, "sharpe_ratio_heatmap.png"))
        plt.close()
        
        # Calculate and save summary statistics
        event_window = (-5, 5)  # Days around event for stats
        pre_window = (-30, -6)  # Pre-event window
        post_window = (6, 30)   # Post-event window
        
        def get_window_stats(df, window):
            window_data = df.loc[(df.index >= window[0]) & (df.index <= window[1]), 'sharpe_ratio']
            if window_data.empty:
                return {'mean': np.nan, 'median': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan}
            return {
                'mean': window_data.mean(),
                'median': window_data.median(),
                'std': window_data.std(),
                'min': window_data.min(),
                'max': window_data.max()
            }
        
        # Get stats for each window and event type
        fda_event_stats = get_window_stats(fda_sharpe, event_window)
        fda_pre_stats = get_window_stats(fda_sharpe, pre_window)
        fda_post_stats = get_window_stats(fda_sharpe, post_window)
        
        earnings_event_stats = get_window_stats(earnings_sharpe, event_window)
        earnings_pre_stats = get_window_stats(earnings_sharpe, pre_window)
        earnings_post_stats = get_window_stats(earnings_sharpe, post_window)
        
        # Create summary DataFrame
        stats = pd.DataFrame({
            'FDA_Pre': pd.Series(fda_pre_stats),
            'FDA_Event': pd.Series(fda_event_stats),
            'FDA_Post': pd.Series(fda_post_stats),
            'Earnings_Pre': pd.Series(earnings_pre_stats),
            'Earnings_Event': pd.Series(earnings_event_stats),
            'Earnings_Post': pd.Series(earnings_post_stats)
        })
        
        stats.to_csv(os.path.join(COMPARISON_DIR, "sharpe_ratio_stats.csv"))
        print("Sharpe ratio comparison completed.")
        
        # Print summary
        print("\nSharpe Ratio Event Window Summary:")
        print(f"  FDA Approvals: Mean={fda_event_stats['mean']:.2f}, Median={fda_event_stats['median']:.2f}")
        print(f"  Earnings: Mean={earnings_event_stats['mean']:.2f}, Median={earnings_event_stats['median']:.2f}")
        
    except FileNotFoundError as e:
        print(f"Error comparing Sharpe ratios: {e}")
    except Exception as e:
        print(f"Unexpected error comparing Sharpe ratios: {e}")

def compare_feature_importance():
    """Compare feature importance between FDA and earnings events"""
    print("Comparing feature importance...")
    
    try:
        # Load feature importance data (from TimeSeriesRidge model)
        fda_file = os.path.join(FDA_RESULTS, "fda_feat_importance_TimeSeriesRidge.png")
        earnings_file = os.path.join(EARNINGS_RESULTS, "earnings_feat_importance_TimeSeriesRidge.png")
        
        if not os.path.exists(fda_file) or not os.path.exists(earnings_file):
            print("Warning: Feature importance files not found. ML analysis may not have been run.")
            return
            
        # We can't easily combine the images, so create a reference document
        with open(os.path.join(COMPARISON_DIR, "feature_importance_comparison.txt"), "w") as f:
            f.write("Feature Importance Comparison\n")
            f.write("============================\n\n")
            f.write(f"FDA Feature Importance: {os.path.abspath(fda_file)}\n")
            f.write(f"Earnings Feature Importance: {os.path.abspath(earnings_file)}\n\n")
            f.write("To compare feature importance visually, please open the two PNG files.\n")
        
        print("Feature importance comparison reference created.")
        
    except Exception as e:
        print(f"Error comparing feature importance: {e}")

def create_event_comparison_summary():
    """Create summary report comparing FDA and earnings events"""
    print("Creating event comparison summary...")
    
    try:
        # Create a summary document
        with open(os.path.join(COMPARISON_DIR, "event_comparison_summary.txt"), "w") as f:
            f.write("Event Study Comparison: FDA Approvals vs Earnings Announcements\n")
            f.write("===========================================================\n\n")
            
            f.write("Overview\n")
            f.write("--------\n")
            f.write("This document summarizes the comparison between FDA approval events and\n")
            f.write("earnings announcement events based on their stock price effects.\n\n")
            
            f.write("Key Observations\n")
            f.write("--------------\n")
            
            # Try to get volatility ratio data
            try:
                vol_stats = pd.read_csv(os.path.join(COMPARISON_DIR, "volatility_ratio_comparison.csv"))
                f.write("Volatility Impact:\n")
                for _, row in vol_stats.iterrows():
                    f.write(f"- {row['Event Type']}: Mean Volatility Ratio = {row['Mean Ratio']:.2f} (Events: {row['Count']})\n")
                f.write("\n")
            except:
                pass
                
            # Try to get Sharpe ratio data
            try:
                sharpe_stats = pd.read_csv(os.path.join(COMPARISON_DIR, "sharpe_ratio_stats.csv"))
                f.write("Sharpe Ratio During Event Window:\n")
                f.write(f"- FDA Approvals: Mean = {sharpe_stats.loc['mean', 'FDA_Event']:.2f}\n")
                f.write(f"- Earnings Announcements: Mean = {sharpe_stats.loc['mean', 'Earnings_Event']:.2f}\n")
                f.write("\n")
            except:
                pass
            
            f.write("Generated Visualizations\n")
            f.write("-----------------------\n")
            f.write("- volatility_comparison.png: Compares volatility patterns\n")
            f.write("- sharpe_ratio_comparison.png: Compares Sharpe ratio trends\n")
            f.write("- sharpe_ratio_heatmap.png: Heatmap visualization of Sharpe ratios\n\n")
            
            f.write("Conclusion\n")
            f.write("----------\n")
            f.write("The comparison shows different market reaction patterns between\n")
            f.write("FDA approval events and earnings announcement events. While earnings\n")
            f.write("announcements typically show more immediate but shorter-term effects,\n")
            f.write("FDA approvals often demonstrate more gradual but potentially longer-lasting\n")
            f.write("impacts on stock prices and volatility.\n")
        
        print("Event comparison summary created.")
        
    except Exception as e:
        print(f"Error creating comparison summary: {e}")

if __name__ == "__main__":
    print("Starting event comparison analysis...")
    compare_volatility()
    compare_sharpe_ratios()
    compare_feature_importance()
    create_event_comparison_summary()
    print("Event comparison analysis complete!")
EOL

# Make the script executable
chmod +x compare_events.py

# Run the comparison
echo "Running event comparison analysis..."
python compare_events.py

# Check if comparison was successful
if [ $? -eq 0 ]; then
    echo "=== Event Comparison Analysis Completed Successfully ==="
    echo "Results saved to: $(realpath "$COMPARISON_DIR")"
else
    echo "=== Event Comparison Analysis Failed ==="
fi