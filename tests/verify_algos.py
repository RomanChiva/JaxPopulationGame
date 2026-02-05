import subprocess
import sys

def verify_algos():
    algos = ['ppo', 'online_q', 'online_sarsa', 'average_batch_q', 'wolf']
    
    for algo in algos:
        print(f"Testing algorithm: {algo}...")
        # Set minimal steps for fast verification
        # For PPO: NUM_UPDATES=2, NUM_STEPS=10 -> 20 steps
        # For Others: TOTAL_TIMESTEPS=50
        
        cmd = [
            "python", "main.py",
            f"ALGO={algo}",
            "USE_WANDB=false",
            "USE_WANDB=false",
            "TOTAL_EPISODES=4",
            "EPISODES_PER_UPDATE=2",
            "EPISODE_LENGTH=5",
            #"BATCH_SIZE=4", # Kept for PPO internal batching
            "NUM_SEEDS=1",
            "NUM_AGENTS=4" 
        ]
        
        # PPO and Batch Q use Batch Mode
        if algo in ["ppo", "average_batch_q"]:
             cmd.append("TRAINING_MODE=batch")
        else:
             cmd.append("TRAINING_MODE=online")

        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✅ {algo} passed.")
        except subprocess.CalledProcessError as e:
            print(f"❌ {algo} failed!")
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)
            sys.exit(1)

    print("All algorithms verified successfully.")

if __name__ == "__main__":
    verify_algos()
