from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from env.plasma_env import PlasmaEnv
import os

print("Creating simplified plasma environment for SAC training...")

# Create a simpler environment for training (fewer particles = less chaos)
train_env = PlasmaEnv(n_particles=25)  # Reduced from 50 to 25 for easier learning
eval_env = PlasmaEnv(n_particles=25)

print(f"Training environment: {train_env.n_particles} particles")
print(f"Observation space: {train_env.observation_space}")
print(f"Action space: {train_env.action_space}")

# SAC with tuned hyperparameters for plasma physics
model = SAC(
    "MlpPolicy", 
    train_env,
    learning_rate=3e-4,        # Standard learning rate
    buffer_size=50000,         # Smaller buffer for faster updates
    batch_size=256,            # Good batch size for continuous control
    tau=0.005,                 # Soft update coefficient
    gamma=0.99,                # Discount factor
    train_freq=1,              # Train after every step
    gradient_steps=1,          # One gradient step per env step
    ent_coef='auto',           # Automatic entropy tuning for exploration
    target_update_interval=1,  # Update target networks frequently
    verbose=1,
    tensorboard_log="./sac_plasma_tensorboard/"
)

print("Starting SAC training...")
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
model.save("./models/sac_plasma_final")

print("Model saved! You can now test it with test_rl_controlled.py")
