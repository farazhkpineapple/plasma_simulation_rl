from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from env.plasma_env import PlasmaEnv
import os

# Create environment
env = PlasmaEnv()

# Optional: check if env follows Gym API
check_env(env)

# Logging
log_dir = "logs/"
os.makedirs(log_dir, exist_ok=True)

# Create PPO agent
model = PPO(
    "MlpPolicy",           # neural net policy
    env,
    verbose=1,
    tensorboard_log=log_dir
)

# Train the agent — 2 million timesteps for excellent plasma confinement
model.learn(total_timesteps=2_000_000)

# Save model
model.save("ppo_plasma")

print("✅ Training complete with 2M timesteps. Model saved as ppo_plasma.zip")
