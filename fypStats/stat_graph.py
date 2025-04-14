import re
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

def parse_stat_file(filename, num_requests=None):
    """Parse vLLM stats file and extract TTFT, ITL, and TPOT metrics."""
    with open(filename, 'r') as f:
        content = f.read()
    
    # Extract TTFT values
    ttft_match = re.search(r'ttft:\[(.*?)\]', content)
    ttft_values = []
    if ttft_match:
        ttft_str = ttft_match.group(1)
        ttft_values = [float(val.strip()) for val in ttft_str.split(',')]
    
    # Auto-detect num_requests if not provided
    if num_requests is None:
        num_requests = len(ttft_values)
    
    # Extract ITL values
    itl_values = [None] * num_requests
    itl_matches = re.findall(r'itl_(\d+):(.*?)(?:\n|$)', content)
    for idx_str, val_str in itl_matches:
        request_num = int(idx_str)
        if request_num < num_requests:
            itl_values[request_num] = float(val_str.strip())
    
    # Extract TPOT values (first value of each tpot list)
    tpot_values = []
    tpot_matches = re.findall(r'tpot:\[(.*?)\]', content)
    for match in tpot_matches:
        values = [float(x.strip()) for x in match.split(',')]
        if values:
            tpot_values.append(values[0])
    
    # Limit to the specified number of requests
    ttft_values = ttft_values[:num_requests]
    tpot_values = tpot_values[:num_requests]
    
    return ttft_values, itl_values, tpot_values

def plot_metrics(ttft_values, itl_values, tpot_values, filename):
    """Create plots for TTFT, ITL, and TPOT metrics."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    
    # Plot TTFT
    axes[0].plot(range(1, len(ttft_values) + 1), ttft_values, 'o-', color='blue')
    axes[0].set_title('Time to First Token (TTFT)')
    axes[0].set_xlabel('Request Number')
    axes[0].set_ylabel('Time (seconds)')
    axes[0].grid(True)
    
    # Plot ITL
    x_vals = []
    y_vals = []
    for i, val in enumerate(itl_values):
        if val is not None:
            x_vals.append(i + 1)
            y_vals.append(val)
    
    axes[1].plot(x_vals, y_vals, 'o-', color='green')
    axes[1].set_title('Inter-Token Latency (ITL)')
    axes[1].set_xlabel('Request Number')
    axes[1].set_ylabel('Time (seconds)')
    axes[1].grid(True)
    
    # Plot TPOT
    axes[2].plot(range(1, len(tpot_values) + 1), tpot_values, 'o-', color='red')
    axes[2].set_title('Throughput Over Time (TPOT)')
    axes[2].set_xlabel('Request Number')
    axes[2].set_ylabel('Time (seconds)')
    axes[2].grid(True)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.splitext(filename)[0] + "_plot.png"
    plt.savefig(output_file)
    print(f"Plot saved as: {output_file}")
    
    return fig

def print_statistics(ttft_values, itl_values, tpot_values):
    """Print summary statistics for each metric."""
    print("\nSummary Statistics:")
    print("-" * 50)
    
    # TTFT stats
    print(f"TTFT: min={min(ttft_values):.6f}s, max={max(ttft_values):.6f}s, " 
          f"avg={np.mean(ttft_values):.6f}s")
    
    # ITL stats
    valid_itl = [v for v in itl_values if v is not None]
    if valid_itl:
        print(f"ITL:  min={min(valid_itl):.6f}s, max={max(valid_itl):.6f}s, "
              f"avg={np.mean(valid_itl):.6f}s")
    else:
        print("ITL:  No valid data")
    
    # TPOT stats
    if tpot_values:
        print(f"TPOT: min={min(tpot_values):.6f}s, max={max(tpot_values):.6f}s, "
              f"avg={np.mean(tpot_values):.6f}s")
    else:
        print("TPOT: No valid data")

def mai(filename,num_requests):

    # Parse the file
    ttft_values, itl_values, tpot_values = parse_stat_file(
        filename, num_requests
    )
    
    num_requests = len(ttft_values)
    print(f"Processing {num_requests} requests from file: {filename}")
    
    # Print statistics
    print_statistics(ttft_values, itl_values, tpot_values)
    
    # Create and display the plots
    fig = plot_metrics(ttft_values, itl_values, tpot_values, filename)
    plt.show()

if __name__ == "__main__":
    filename = "/home/hkngae/vllm/fypStats/local/stat_20.txt"
    num_requests = 20
    mai(filename, num_requests)