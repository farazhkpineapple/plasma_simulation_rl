from stable_baselines3 import SAC
from env.plasma_env import PlasmaEnv
import time
import matplotlib.pyplot as plt
import os

# Enable interactive plotting
plt.ion()

print("Loading trained SAC model...")
try:
    model = SAC.load("./models/sac_plasma_final")
    print("✓ Model loaded successfully!")
except:
    print("❌ No trained model found. Please run train_sac_agent.py first")
    exit()

print("Creating plasma environment for RL control...")
env = PlasmaEnv(n_particles=25, rl_mode=True)  # Enable RL mode for proper title
obs, info = env.reset()

print(f"Environment created with {env.n_particles} plasma particles")
print(f"Observation shape: {obs.shape}")
print(f"Initial center of mass: {obs[:3]}")

print("Starting RL-controlled simulation...")
for step in range(100):  # 100 total environment steps
    # Use trained model to predict actions (instead of random)
    action, _states = model.predict(obs, deterministic=True)
    
    # Take 1 step at a time
    obs, reward, done, truncated, _ = env.step(action)
    env.render()
    
    # Print info every 5 steps
    if step % 5 == 0:
        center_of_mass = obs[:3]
        avg_velocity = obs[3:6]
        print(f"Step {step}: Center of mass: [{center_of_mass[0]:.2f}, {center_of_mass[1]:.2f}, {center_of_mass[2]:.2f}], "
              f"Avg velocity: [{avg_velocity[0]:.2f}, {avg_velocity[1]:.2f}, {avg_velocity[2]:.2f}], "
              f"Action: {action}, Reward: {reward:.2f}")
    
    time.sleep(0.05)  # Twice as fast as random test

print("RL-controlled simulation complete!")
input("Press Enter to close...")
env.close()
