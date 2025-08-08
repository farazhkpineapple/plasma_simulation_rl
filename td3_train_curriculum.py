from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
from env.plasma_env import PlasmaEnv
import numpy as np
import os

def evaluate_confinement(model, env, n_episodes=10, n_steps=100):
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

def train_td3_curriculum_stage(stage_name, n_particles, model=None, target_confinement=95.0):
    """Train a single TD3 curriculum stage"""
    print(f"\n🎓 === TD3 CURRICULUM STAGE: {stage_name} ({n_particles} particles) ===")
    print(f"Target: {target_confinement}% confinement for 100 steps")
    
    # Create environments
    train_env = PlasmaEnv(n_particles=n_particles)
    eval_env = PlasmaEnv(n_particles=n_particles)
    
    # Action noise for exploration
    n_actions = train_env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), 
        sigma=0.1 * np.ones(n_actions)
    )
    
    # Create or transfer model
    if model is None:
        print("🆕 Creating new TD3 model...")
        model = TD3(
            "MlpPolicy", 
            train_env,
            learning_rate=1e-3,
            buffer_size=100000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=(1, "step"),
            gradient_steps=1,
            action_noise=action_noise,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
            policy_delay=2,
            verbose=1,
            tensorboard_log=f"./td3_curriculum_tensorboard/{stage_name}/"
        )
    else:
        print("🔄 Transferring existing TD3 model to new environment...")
        model.set_env(train_env)
    
    # Setup evaluation callback
    os.makedirs(f"./models/td3_curriculum/{stage_name}", exist_ok=True)
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=f"./models/td3_curriculum/{stage_name}/",
        log_path=f"./logs/td3_curriculum/{stage_name}/", 
        eval_freq=5000,
        deterministic=True, 
        render=False
    )
    
    # Training loop with confinement checking
    max_attempts = 3
    timesteps_per_attempt = 200000  # 200K steps per attempt
    
    for attempt in range(max_attempts):
        print(f"\n📚 TD3 Training attempt {attempt + 1}/{max_attempts}")
        print(f"Training for {timesteps_per_attempt:,} steps...")
        
        # Train the model
        model.learn(
            total_timesteps=timesteps_per_attempt,
            callback=eval_callback,
            progress_bar=True,
            reset_num_timesteps=False  # Continue from previous training
        )
        
        # Evaluate confinement
        print("🔬 Evaluating TD3 confinement performance...")
        avg_confinement, std_confinement = evaluate_confinement(model, eval_env)
        
        print(f"📊 TD3 Results: {avg_confinement:.1f}% ± {std_confinement:.1f}% confinement")
        
        # Check if target achieved
        if avg_confinement >= target_confinement:
            print(f"🎉 TD3 SUCCESS! Achieved {avg_confinement:.1f}% confinement (target: {target_confinement}%)")
            break
        else:
            print(f"❌ TD3 Target not met. Need {target_confinement - avg_confinement:.1f}% more.")
            if attempt < max_attempts - 1:
                print("Continuing TD3 training...")
    
    # Save final model for this stage
    model_path = f"./models/td3_curriculum/{stage_name}/final_model"
    model.save(model_path)
    print(f"💾 Saved TD3 model: {model_path}")
    
    return model, avg_confinement

def main():
    """Run complete TD3 curriculum learning"""
    print("🚀 Starting TD3 Curriculum Learning for Plasma Confinement")
    print("Goal: Progressive TD3 training from 10 → 15 → 20 → 25 particles")
    print("Target: 95% confinement for 100 environmental steps\n")
    
    # Define curriculum stages
    curriculum = [
        {"name": "Stage1_10particles", "particles": 10, "target": 95.0},
        {"name": "Stage2_15particles", "particles": 15, "target": 95.0},
        {"name": "Stage3_20particles", "particles": 20, "target": 95.0},
        {"name": "Stage4_25particles", "particles": 25, "target": 95.0}
    ]
    
    model = None
    results = {}
    
    for stage in curriculum:
        model, final_confinement = train_td3_curriculum_stage(
            stage["name"], 
            stage["particles"], 
            model, 
            stage["target"]
        )
        results[stage["name"]] = final_confinement
    
    # Print final results
    print("\n🏆 === TD3 CURRICULUM LEARNING COMPLETE ===")
    for stage_name, confinement in results.items():
        particles = stage_name.split("_")[1]
        print(f"TD3 {particles}: {confinement:.1f}% confinement")
    
    print(f"\n💾 Final TD3 model saved as: ./models/td3_curriculum/Stage4_25particles/final_model")
    print("🧪 Test with: python td3_test_curriculum.py")

if __name__ == "__main__":
    main()
