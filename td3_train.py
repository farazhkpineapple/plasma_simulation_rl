from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
from env.plasma_env import PlasmaEnv
import numpy as np
import os

print("Creating simplified plasma environment for TD3 training...")

# Create a simpler environment for training (fewer particles = less chaos)
train_env = PlasmaEnv(n_particles=25)  # Same as SAC for comparison
eval_env = PlasmaEnv(n_particles=25)

print(f"Training environment: {train_env.n_particles} particles")
print(f"Observation space: {train_env.observation_space}")
print(f"Action space: {train_env.action_space}")

# Action noise for exploration (TD3 uses deterministic policy + noise)
n_actions = train_env.action_space.shape[-1]
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions), 
    sigma=0.1 * np.ones(n_actions)  # 10% noise for exploration
)

# TD3 with optimized hyperparameters for plasma physics
model = TD3(
    "MlpPolicy", 
    train_env,
    learning_rate=1e-3,           # Slightly higher learning rate for TD3
    buffer_size=100000,           # Larger buffer for more stable learning
    batch_size=256,               # Good batch size for continuous control
    tau=0.005,                    # Soft update coefficient
    gamma=0.99,                   # Discount factor
    train_freq=(1, "step"),       # Train after every step
    gradient_steps=1,             # One gradient step per env step
    action_noise=action_noise,    # Exploration noise
    target_policy_noise=0.2,      # Target smoothing noise
    target_noise_clip=0.5,        # Clip target noise
    policy_delay=2,               # Update policy every 2 critic updates (key TD3 feature)
    verbose=1,
    tensorboard_log="./td3_plasma_tensorboard/"
)

print("Starting TD3 training...")
print("This will train for 1,000,000 steps (should take 2-2.5 hours)")

# Create callback for evaluation during training
os.makedirs("./models", exist_ok=True)
eval_callback = EvalCallback(
    eval_env, 
    best_model_save_path="./models/",
    log_path="./logs/", 
    eval_freq=5000,
    deterministic=True, 
    render=False
)

# Train the model
model.learn(
    total_timesteps=1000000,  # 1 million steps for excellent performance
    callback=eval_callback,
    progress_bar=True
)

print("Training complete!")
print("Saving final model...")
model.save("./models/td3_plasma_final")

print("Model saved! You can now test it with td3_test.py")
