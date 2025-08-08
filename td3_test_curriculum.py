from stable_baselines3 import TD3
from env.plasma_env import PlasmaEnv
import numpy as np
import os

def test_td3_curriculum_stage(stage_name, n_particles, n_episodes=10, n_steps=100):
    """Test a specific TD3 curriculum stage"""
    print(f"\n🧪 Testing TD3 {stage_name} ({n_particles} particles)")
    
    # Create environment
    env = PlasmaEnv(n_particles=n_particles, rl_mode=True)
    
    # Load model
    model_path = f"./models/td3_curriculum/{stage_name}/best_model"
    if not os.path.exists(model_path + ".zip"):
        model_path = f"./models/td3_curriculum/{stage_name}/final_model"
    
    if not os.path.exists(model_path + ".zip"):
        print(f"❌ No TD3 model found for {stage_name}")
        return None, None
    
    print(f"📁 Loading TD3 model: {model_path}")
    model = TD3.load(model_path)
    
    # Test episodes
    confinement_rates = []
    episode_rewards = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0
        total_particles = env.n_particles
        
        for step in range(n_steps):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            
            if done or truncated:
                break
        
        # Calculate confinement rate
        particles_lost = getattr(env, 'particles_lost', 0)
        confined_particles = total_particles - particles_lost
        confinement_rate = (confined_particles / total_particles) * 100
        
        confinement_rates.append(confinement_rate)
        episode_rewards.append(total_reward)
        
        print(f"Episode {episode + 1}: {confinement_rate:.1f}% confined, reward: {total_reward:.1f}")
    
    # Calculate statistics
    avg_confinement = np.mean(confinement_rates)
    std_confinement = np.std(confinement_rates)
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    print(f"\n📊 TD3 Results for {stage_name}:")
    print(f"   Confinement: {avg_confinement:.1f}% ± {std_confinement:.1f}%")
    print(f"   Reward: {avg_reward:.1f} ± {std_reward:.1f}")
    
    return avg_confinement, avg_reward

def test_random_baseline(n_particles, n_episodes=10, n_steps=100):
    """Test random policy baseline"""
    print(f"\n🎲 Testing Random Baseline ({n_particles} particles)")
    
    env = PlasmaEnv(n_particles=n_particles, rl_mode=False)
    
    confinement_rates = []
    episode_rewards = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0
        total_particles = env.n_particles
        
        for step in range(n_steps):
            action = env.action_space.sample()  # Random action
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            
            if done or truncated:
                break
        
        # Calculate confinement rate
        particles_lost = getattr(env, 'particles_lost', 0)
        confined_particles = total_particles - particles_lost
        confinement_rate = (confined_particles / total_particles) * 100
        
        confinement_rates.append(confinement_rate)
        episode_rewards.append(total_reward)
    
    avg_confinement = np.mean(confinement_rates)
    std_confinement = np.std(confinement_rates)
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    print(f"📊 Random Results ({n_particles} particles):")
    print(f"   Confinement: {avg_confinement:.1f}% ± {std_confinement:.1f}%")
    print(f"   Reward: {avg_reward:.1f} ± {std_reward:.1f}")
    
    return avg_confinement, avg_reward

def main():
    """Test all TD3 curriculum stages and compare with baselines"""
    print("🧪 TD3 CURRICULUM TESTING")
    print("Testing all trained TD3 models vs random baselines\n")
    
    # Define stages to test
    stages = [
        {"name": "Stage1_10particles", "particles": 10},
        {"name": "Stage2_15particles", "particles": 15},
        {"name": "Stage3_20particles", "particles": 20},
        {"name": "Stage4_25particles", "particles": 25}
    ]
    
    print("=" * 60)
    print("TD3 CURRICULUM RESULTS")
    print("=" * 60)
    
    td3_results = {}
    random_results = {}
    
    for stage in stages:
        # Test TD3 model
        td3_conf, td3_reward = test_td3_curriculum_stage(
            stage["name"], 
            stage["particles"]
        )
        
        # Test random baseline
        random_conf, random_reward = test_random_baseline(stage["particles"])
        
        if td3_conf is not None:
            td3_results[stage["particles"]] = {
                "confinement": td3_conf, 
                "reward": td3_reward
            }
        
        random_results[stage["particles"]] = {
            "confinement": random_conf, 
            "reward": random_reward
        }
        
        print("-" * 60)
    
    # Final comparison
    print("\n🏆 === FINAL TD3 CURRICULUM COMPARISON ===")
    print(f"{'Particles':<10} {'TD3 Conf%':<12} {'Random Conf%':<14} {'Improvement':<12}")
    print("-" * 60)
    
    for particles in [10, 15, 20, 25]:
        if particles in td3_results:
            td3_conf = td3_results[particles]["confinement"]
            random_conf = random_results[particles]["confinement"]
            improvement = td3_conf - random_conf
            
            print(f"{particles:<10} {td3_conf:<12.1f} {random_conf:<14.1f} {improvement:<12.1f}")
        else:
            random_conf = random_results[particles]["confinement"]
            print(f"{particles:<10} {'N/A':<12} {random_conf:<14.1f} {'N/A':<12}")
    
    # Check if curriculum learning was successful
    if 25 in td3_results:
        final_conf = td3_results[25]["confinement"]
        if final_conf >= 95.0:
            print(f"\n🎉 TD3 CURRICULUM SUCCESS! Final model achieves {final_conf:.1f}% confinement!")
        else:
            print(f"\n⚠️  TD3 Curriculum target not fully met. Final: {final_conf:.1f}% (target: 95%)")
    else:
        print("\n❌ TD3 Final model not found. Training may be incomplete.")
    
    print("\n💡 Note: Test individual models with specific particle counts using:")
    print("   python td3_test.py  # For standard 25-particle model")

if __name__ == "__main__":
    main()
