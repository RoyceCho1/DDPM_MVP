import argparse
import os
import re
import matplotlib
matplotlib.use('Agg') # Server-side plotting backend
import matplotlib.pyplot as plt
import numpy as np

def analyze_log(log_path):
    """
    학습 로그 파일을 읽어서 loss와 lr 변화를 시각화하고, 기본적인 통계 정보를 출력
    """
    print(f"Parsing log file: {log_path}")
    
    # 데이터를 저장할 리스트 초기화
    steps = []
    losses = []
    lrs = []
    
    # 1. 정규 표현식 패턴 정의
    # 로그 파일의 각 줄에서 데이터를 추출하기 위한 패턴
    # 예 : "Step 100 Loss: 0.1234 LR: 0.000200"
    pattern = re.compile(r"Step (\d+) Loss: ([\d.]+) LR: ([\d.]+)")
    
    if not os.path.exists(log_path):
        print("Log file not found!")
        return

    # 2. 로그 파일 읽기
    with open(log_path, 'r') as f:
        for line in f:
            match = pattern.search(line)    # 패턴과 일치하는 부분이 있는지 검사
            if match:
                # group(1): Step, group(2): Loss, group(3): LR
                steps.append(int(match.group(1)))
                losses.append(float(match.group(2)))
                lrs.append(float(match.group(3)))
                
    if not steps:
        print("No valid log entries found.")
        return

    steps = np.array(steps, dtype=np.float32)
    losses = np.array(losses, dtype=np.float32)
    lrs = np.array(lrs, dtype=np.float32)

    
    # 3. 통계 분석
    min_loss = np.min(losses)               # 최소 Loss 값
    min_step = steps[np.argmin(losses)]     # 최소 Loss가 발생한 지점(Step)
    final_loss = losses[-1]                 # 마지막 Loss 값
    
    print("\nTraining Statistics:")
    print(f"   - Total Steps Logged: {len(steps)}")
    print(f"   - Final Step: {steps[-1]}")
    print(f"   - Best Loss: {min_loss:.6f} (at step {min_step})")
    print(f"   - Final Loss: {final_loss:.6f}")
    
    # 4. Plotting
    save_path = log_path.replace(".log", "_analysis.png")
    plt.figure(figsize=(10, 6))
    
    # Loss Curve
    plt.subplot(2, 1, 1)
    # Raw Loss: 실제 변동을 보여주는 흐린 선
    plt.plot(steps, losses, alpha=0.3, color='blue', label='Raw Loss')
    
    # Smooth Curve: Moving average를 계산하여 추세를 보여주는 선
    # Window 크기: 5 또는 데이터의 5% 중 큰 값
    window = max(5, len(losses) // 20)
    if len(losses) > window:
        smooth_loss = np.convolve(losses, np.ones(window)/window, mode='valid')
        smooth_steps = steps[window-1:]
        plt.plot(smooth_steps, smooth_loss, color='red', label=f'Smoothed (MA={window})')
        
    plt.title(f"Training Loss Check ({os.path.basename(log_path)})")
    plt.xlabel("Step")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # LR Curve
    plt.subplot(2, 1, 2)
    plt.plot(steps, lrs, color='green', label='Learning Rate')
    plt.xlabel("Step")
    plt.ylabel("LR")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\nSaved analysis plot to: {save_path}")
    print("   (Check this image to see stability and convergence)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, default="results/ddpm_cifar10_v1/train.log", help="Path to train.log")
    args = parser.parse_args()
    
    analyze_log(args.log_path)
