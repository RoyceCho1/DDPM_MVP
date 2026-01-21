import re
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np

def parse_log(log_path):
    """
    Parses the training log file to extract steps, losses, and learning rates.
    """
    steps = []
    losses = []
    lrs = []

    # Regex pattern to match the log line format
    # Example: 2026-01-21 01:32:14,275 - INFO - Step 798700 Loss: 0.0226 LR: 0.000000
    pattern = re.compile(r"Step\s+(\d+)\s+Loss:\s+([\d.]+)\s+LR:\s+([\d.]+)")

    if not os.path.exists(log_path):
        print(f"Error: Log file not found at {log_path}")
        return steps, losses, lrs

    with open(log_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                step = int(match.group(1))
                loss = float(match.group(2))
                lr = float(match.group(3))

                steps.append(step)
                losses.append(loss)
                lrs.append(lr)

    return steps, losses, lrs

def smooth_data(data, window_size=50):
    """
    Applies a moving average smoothing to the data.
    """
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_loss(steps, losses, output_path, window_size=100):
    """
    Plots the training loss and saves the figure.
    """
    plt.figure(figsize=(12, 6))
    
    # Ensure inputs are numpy arrays
    steps = np.array(steps)
    losses = np.array(losses)

    # Plot raw data with high transparency
    plt.plot(steps, losses, alpha=0.3, label='Raw Loss', color='gray', linewidth=0.5)

    # Plot smoothed data
    smoothed_losses = smooth_data(losses, window_size)
    # Adjust steps for valid convolution mode
    smoothed_steps = steps[len(steps) - len(smoothed_losses):]
    
    plt.plot(smoothed_steps, smoothed_losses, label=f'Smoothed Loss (MA={window_size})', color='blue', linewidth=2)

    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss over Steps')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse and plot training logs.")
    parser.add_argument("--log_path", type=str, default="train.log", help="Path to the log file")
    parser.add_argument("--output_path", type=str, default="loss_curve.png", help="Path to save the plot")
    parser.add_argument("--window_size", type=int, default=100, help="Window size for smoothing")
    parser.add_argument("--test", action="store_true", help="Run with dummy data for verification")

    args = parser.parse_args()

    if args.test:
        # Generate dummy data for testing
        print("Running in test mode...")
        steps = list(range(0, 1000, 10))
        losses = [0.1 * np.exp(-s/1000) + 0.01 * np.random.randn() for s in steps]
        lrs = [0.001] * len(steps)
        plot_loss(steps, losses, "test_plot.png", window_size=5)
        print("Test complete. Check test_plot.png")
    else:
        print(f"Reading log from {args.log_path}...")
        steps, losses, lrs = parse_log(args.log_path)
        
        if steps:
            print(f"Found {len(steps)} data points.")
            plot_loss(steps, losses, args.output_path, args.window_size)
        else:
            print("No data found in log file.")
