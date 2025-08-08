from env.plasma_env import PlasmaEnv
import time
import matplotlib.pyplot as plt

# Enable interactive plotting
plt.ion()

print("Creating new multi-particle plasma environment...")
env = PlasmaEnv()
obs, info = env.reset()

print(f"Environment created with {env.n_particles} plasma particles")
print(f"Observation shape: {obs.shape}")
print(f"Initial center of mass: {obs[:3]}")

print("Starting simulation with random actions...")
for step in range(50):
    action = env.action_space.sample()  # random magnetic field controls
    # Take 2 steps at a time
    obs, reward, done, truncated, _ = env.step(action)
    obs, reward, done, truncated, _ = env.step(action)
    env.render()
    
    # Print info every 5 steps
    if step % 5 == 0:
        center_of_mass = obs[:3]
        avg_velocity = obs[3:6]
        print(f"Step {step}: Center of mass: [{center_of_mass[0]:.2f}, {center_of_mass[1]:.2f}, {center_of_mass[2]:.2f}], "
              f"Avg velocity: [{avg_velocity[0]:.2f}, {avg_velocity[1]:.2f}, {avg_velocity[2]:.2f}], "
              f"Action: {action}, Reward: {reward:.2f}")
    
    time.sleep(0.1)  # Slow down for better visualization

print("Simulation complete!")
input("Press Enter to close...")
env.close()
