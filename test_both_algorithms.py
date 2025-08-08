from stable_baselines3 import SAC, TD3
from env.plasma_env import PlasmaEnv
import time
import matplotlib.pyplot as plt
import threading
import numpy as np

# Enable interactive plotting
plt.ion()

def run_algorithm(algorithm_name, model_path, subplot_position, fig):
    """Run an algorithm in a subplot"""
    print(f"Loading {algorithm_name} model...")
    
    # Load the appropriate model
    try:
        if algorithm_name == "SAC":
            model = SAC.load(model_path)
        else:  # TD3
            model = TD3.load(model_path)
        print(f"✓ {algorithm_name} model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load {algorithm_name} model: {e}")
        return
    
    # Create environment
    env = PlasmaEnv(n_particles=25, rl_mode=True, plot_in_subplot=True, 
                   subplot_ax=subplot_position, figure=fig)
    obs, info = env.reset()
    
    print(f"{algorithm_name} environment created with {env.n_particles} particles")
    
    # Run simulation
    for step in range(100):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        
        # Update the subplot title with algorithm name and step info
        center_of_mass = obs[:3]
        particles_lost = env.n_particles - len(env.positions)
        confinement_rate = (1 - particles_lost / env.n_particles) * 100
        
        subplot_position.set_title(f'{algorithm_name} Control - Step {step}\n'
                                 f'Particles: {env.n_particles - particles_lost}/{env.n_particles} '
                                 f'({confinement_rate:.1f}% confined)\n'
                                 f'Reward: {reward:.1f}', fontsize=10)
        
        env.render()
        
        # Print progress every 20 steps
        if step % 20 == 0:
            avg_velocity = obs[3:6]
            print(f"{algorithm_name} Step {step}: CoM=[{center_of_mass[0]:.2f}, {center_of_mass[1]:.2f}, {center_of_mass[2]:.2f}], "
                  f"Action={action}, Reward={reward:.1f}")
        
        time.sleep(0.08)  # Slightly slower for dual visualization
    
    print(f"{algorithm_name} simulation complete!")
    env.close()

def main():
    print("🚀 DUAL ALGORITHM PLASMA CONTROL COMPARISON")
    print("Running SAC and TD3 side by side...")
    
    # Create figure with 2 subplots
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle('AI Plasma Control Comparison: SAC vs TD3', fontsize=16, fontweight='bold')
    
    # Create subplots
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Model paths
    sac_model_path = "./models/curriculum/Stage4_25particles/final_model"
    td3_model_path = "./models/td3_curriculum/Stage4_25particles/final_model"
    
    # Create and start threads for both algorithms
    sac_thread = threading.Thread(target=run_algorithm, 
                                 args=("SAC", sac_model_path, ax1, fig))
    td3_thread = threading.Thread(target=run_algorithm, 
                                 args=("TD3", td3_model_path, ax2, fig))
    
    # Start both threads
    sac_thread.start()
    time.sleep(1)  # Small delay to stagger startup
    td3_thread.start()
    
    # Wait for both to complete
    sac_thread.join()
    td3_thread.join()
    
    print("\n🏆 Both algorithms completed!")
    print("Compare the two control strategies:")
    print("- SAC (left): Soft Actor-Critic approach")
    print("- TD3 (right): Twin Delayed DDPG approach")
    
    input("Press Enter to close...")
    plt.close('all')

if __name__ == "__main__":
    main()
