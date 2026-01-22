import re
import matplotlib.pyplot as plt
import os
import sys

def analyze_logs(log_path, output_plot):
    if not os.path.exists(log_path):
        print(f"Error: Log file not found at {log_path}")
        return

    steps = []
    losses = []
    lrs = []

    print(f"Reading log file: {log_path}")
    with open(log_path, 'r') as f:
        for line in f:
            # Example format: 2026-01-20 02:08:46,101 - INFO - Step 0 Loss: 1.1496 LR: 0.000000
            # We use regex to be robust
            match = re.search(r'Step (\d+) Loss: ([\d.]+) LR: ([\d.]+)', line)
            if match:
                try:
                    step = int(match.group(1))
                    loss = float(match.group(2))
                    lr = float(match.group(3))
                    
                    steps.append(step)
                    losses.append(loss)
                    lrs.append(lr)
                except ValueError:
                    continue

    if not steps:
        print("No valid data found in log file matching the expected format.")
        return

    print(f"Found {len(steps)} data points.")

    # Plotting
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:red'
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(steps, losses, color=color, alpha=0.6, label='Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Learning Rate', color=color) 
    ax2.plot(steps, lrs, color=color, linestyle='--', label='Learning Rate')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('DDPM Training Analysis: Loss and Learning Rate')
    fig.tight_layout()  
    
    print(f"Saving plot to {output_plot}")
    plt.savefig(output_plot)
    print("Done.")

if __name__ == "__main__":
    # Default paths based on user workspace
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(current_dir, 'logs', 'train_1.log')
    output_file = os.path.join(current_dir, 'training_analysis.png')
    
    # Allow command line arguments to override
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]

    analyze_logs(log_file, output_file)
