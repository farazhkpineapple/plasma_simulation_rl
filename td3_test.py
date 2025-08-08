from stable_baselines3 import TD3
from env.plasma_env import PlasmaEnv
import time
import matplotlib.pyplot as plt
import numpy as np
import os

def evaluate_confinement(model, env, n_episodes=5, n_steps=100):
    """Evaluate model confinement rate over multiple episodes"""
    confinement_rates = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        total_particles = env.n_particles
        
        for step in range(n_steps):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            
            if done or truncated:
                break
        
        # Calculate final confinement rate by checking particle positions
        particles_lost = 0
        
        for i in range(total_particles):
            xy_dist = np.sqrt(env.positions[i][0]**2 + env.positions[i][1]**2)
            torus_distance = np.sqrt((xy_dist - env.major_radius)**2 + env.positions[i][2]**2)
            
            if torus_distance > env.minor_radius:
                particles_lost += 1
        
        confined_particles = total_particles - particles_lost
        confinement_rate = (confined_particles / total_particles) * 100
        confinement_rates.append(confinement_rate)
    
    avg_confinement = np.mean(confinement_rates)
    std_confinement = np.std(confinement_rates)
    
    return avg_confinement, std_confinement

# Enable interactive plotting
plt.ion()

print("Loading trained TD3 model...")
try:
    # Try curriculum model first (better performance)
    model = TD3.load("./models/td3_curriculum/Stage4_25particles/final_model")
    print("✓ TD3 Curriculum Model loaded successfully!")
except:
    try:
        # Fallback to individual model
        model = TD3.load("./models/td3_plasma_final")
        print("✓ TD3 Individual Model loaded successfully!")
    except:
        print("❌ No trained TD3 model found.")
        print("Please run either:")
        print("  - td3_train_curriculum.py (recommended)")
        print("  - td3_train.py")
        exit()

print("Creating plasma environment for TD3 control...")
env = PlasmaEnv(n_particles=25, rl_mode=True)  # Enable RL mode for proper title
obs, info = env.reset()

print(f"Environment created with {env.n_particles} plasma particles")
print(f"Observation shape: {obs.shape}")
print(f"Initial center of mass: {obs[:3]}")

# First, evaluate confinement performance
print("\n🔬 Evaluating TD3 confinement performance...")
avg_confinement, std_confinement = evaluate_confinement(model, env)
print(f"📊 TD3 Average Confinement: {avg_confinement:.1f}% ± {std_confinement:.1f}%")
print(f"📈 Performance Rating: {'🏆 Excellent' if avg_confinement > 85 else '👍 Good' if avg_confinement > 70 else '📈 Needs Improvement'}")

# Reset environment for fresh visualization starting from step 0
print("\nResetting environment for visualization...")
obs, info = env.reset()
print(f"Fresh start - Initial center of mass: {obs[:3]}")

print("\nStarting TD3-controlled simulation...")
for step in range(100):  # 100 total environment steps
    # Use trained TD3 model to predict actions (deterministic for testing)
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
    
    time.sleep(0.05)  # Same timing as SAC test

print("TD3-controlled simulation complete!")
input("Press Enter to close...")
env.close()
