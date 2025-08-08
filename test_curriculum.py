from stable_baselines3 import SAC
from env.plasma_env import PlasmaEnv
import time
import matplotlib.pyplot as plt
import os

# Enable interactive plotting
plt.ion()

def test_curriculum_stage(stage_name, n_particles):
    """Test a specific curriculum stage"""
    print(f"\n🧪 Testing {stage_name} ({n_particles} particles)")
    
    model_path = f"./models/curriculum/{stage_name}/final_model"
    
    print(f"Loading model: {model_path}")
    try:
        model = SAC.load(model_path)
        print("✓ Model loaded successfully!")
    except:
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"Creating plasma environment ({n_particles} particles)...")
    env = PlasmaEnv(n_particles=n_particles, rl_mode=True)
    obs, info = env.reset()
    
    print(f"Environment: {env.n_particles} particles")
    print(f"Observation shape: {obs.shape}")
    print(f"Initial center of mass: {obs[:3]}")
    
    print("Starting RL-controlled simulation...")
    
    particles_lost_history = []
    confinement_history = []
    
    for step in range(100):  # 100 total environment steps
        # Use trained model to predict actions
        action, _states = model.predict(obs, deterministic=True)
        
        # Take 1 step
        obs, reward, done, truncated, _ = env.step(action)
        env.render()
        
        # Track confinement
        particles_lost = getattr(env, 'particles_lost', 0)
        confinement_rate = ((n_particles - particles_lost) / n_particles) * 100
        particles_lost_history.append(particles_lost)
        confinement_history.append(confinement_rate)
        
        # Print info every 10 steps
        if step % 10 == 0:
            center_of_mass = obs[:3]
            avg_velocity = obs[3:6]
            print(f"Step {step:2d}: CoM=[{center_of_mass[0]:5.2f}, {center_of_mass[1]:5.2f}, {center_of_mass[2]:5.2f}], "
                  f"Lost={particles_lost:2d}, Confinement={confinement_rate:5.1f}%, Reward={reward:8.1f}")
        
        time.sleep(0.05)
        
        if done or truncated:
            print(f"Episode ended early at step {step}")
            break
    
    # Final results
    final_lost = particles_lost_history[-1] if particles_lost_history else 0
    final_confinement = confinement_history[-1] if confinement_history else 0
    avg_confinement = sum(confinement_history) / len(confinement_history) if confinement_history else 0
    
    print(f"\n📊 RESULTS for {stage_name}:")
    print(f"   Final particles lost: {final_lost}/{n_particles}")
    print(f"   Final confinement: {final_confinement:.1f}%")
    print(f"   Average confinement: {avg_confinement:.1f}%")
    
    target_achieved = "✅ TARGET ACHIEVED!" if final_confinement >= 95.0 else "❌ Target not met"
    print(f"   {target_achieved} (Target: 95%)")
    
    input("Press Enter to continue...")
    env.close()
    
    return final_confinement

def main():
    """Test all curriculum stages"""
    print("🧪 Testing Curriculum Learning Results")
    print("Testing each stage for 95% confinement over 100 steps\n")
    
    stages = [
        ("Stage1_10particles", 10),
        ("Stage2_15particles", 15), 
        ("Stage3_20particles", 20),
        ("Stage4_25particles", 25)
    ]
    
    results = {}
    
    for stage_name, n_particles in stages:
        confinement = test_curriculum_stage(stage_name, n_particles)
        if confinement is not None:
            results[stage_name] = confinement
    
    # Summary
    print("\n🏆 === CURRICULUM TEST SUMMARY ===")
    for stage_name, confinement in results.items():
        particles = stage_name.split("_")[1]
        status = "✅" if confinement >= 95.0 else "❌"
        print(f"{status} {particles}: {confinement:.1f}% confinement")

if __name__ == "__main__":
    main()
