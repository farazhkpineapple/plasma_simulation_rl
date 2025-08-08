import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from env.plasma_env import PlasmaEnv
import os

# Configure the page
st.set_page_config(
    page_title="Plasma Simulation",
    page_icon="🔬",
    layout="wide"
)

# --- About Me (top section) ---
st.container()
st.header("Machine Learning for Random Plasma Drift Control")
st.subheader("Faraz Hakim, Harvard '27 – Physics & Statistics")

st.write(
    "Nuclear fusion, the process that powers the Sun, generates immense energy by fusing light nuclei at extremely high temperatures. "
    "In a tokamak reactor, plasma must be confined in a torus-shaped chamber using strong magnetic fields. "
    "One of the biggest challenges is keeping this plasma stable—if it touches the chamber walls, the reaction stops. "
    "In recent years, scientists have begun using machine learning to help predict and control plasma behavior in real time. "
    "For this project, I built a learning-oriented 3D simulation of random plasma drift (simplified compared to real plasma physics) "
    "and trained a reinforcement learning agent to keep the particles contained within the tokamak boundary.\n\n"

    "I initially tested PPO and SAC reinforcement learning models, but even after over a million training steps, confinement performance was limited. "
    "After trial and error, I switched to the TD3 algorithm, which is well-suited for chaotic, high-variance environments. "
    "Using a curriculum learning approach—starting with fewer particles and gradually increasing the count—helped the agent learn more effectively. "
    "While this simplified simulation cannot achieve perfect confinement, the results improved significantly: "
    "from **66.7% ± 1.9% confinement without RL** to **84.8% ± 3.9% with the trained TD3 agent**. "
    "Below, I first show the simulation without any RL control, followed by the TD3-trained RL model actively working to contain the plasma. "
    "In real-world applications, both the physical models and control algorithms would be far more complex and precise."
)
st.divider()

# Simple title
st.subheader("Random Plasma Drift without RL")

# Single button to run the non-RL simulation and show the live matplotlib graph
if st.button("Run Simulation"):
    # Placeholder for live plot
    plot_placeholder = st.empty()

    # Initialize environment (same as test_plasma_env.py)
    env = PlasmaEnv()
    obs, info = env.reset()

    # Show Non-RL confinement metric before visualization (like the TD3 flow)
    status_placeholder = st.empty()
    def _evaluate_non_rl_on_env(env, n_episodes=3, n_steps=100):
        rates = []
        for _ in range(n_episodes):
            obs, info = env.reset()
            total_particles = env.n_particles
            for _ in range(n_steps):
                action = env.action_space.sample()
                obs, reward, done, truncated, _ = env.step(action)
                if done or truncated:
                    break
            particles_lost = 0
            for i in range(total_particles):
                xy_dist = np.sqrt(env.positions[i][0]**2 + env.positions[i][1]**2)
                torus_distance = np.sqrt((xy_dist - env.major_radius)**2 + env.positions[i][2]**2)
                if torus_distance > env.minor_radius:
                    particles_lost += 1
            confined = total_particles - particles_lost
            rates.append((confined / total_particles) * 100)
        return float(np.mean(rates)), float(np.std(rates))

    with st.spinner("Evaluating non-RL confinement…"):
        avg_conf_nr, std_conf_nr = _evaluate_non_rl_on_env(env)
    status_placeholder.success(f"Non-RL Average Confinement: {avg_conf_nr:.1f}% ± {std_conf_nr:.1f}%")

    # Reset for fresh visualization
    obs, info = env.reset()

    # Run simulation: 50 iterations, taking 2 steps each (as in test_plasma_env.py)
    for step in range(50):
        action = env.action_space.sample()  # random magnetic field controls
        # Take 2 steps at a time
        obs, reward, done, truncated, _ = env.step(action)
        obs, reward, done, truncated, _ = env.step(action)

        # Render and display the matplotlib figure each iteration
        env.render()
        fig = env.fig  # use the env's figure directly for accurate updates
        # Make the figure smaller so it fits on screen
        fig.set_size_inches(6, 4)
        fig.tight_layout()
        plot_placeholder.pyplot(fig, use_container_width=True)

        # Small delay for animation effect
        time.sleep(0.1)

    env.close()
    st.success("Simulation completed!")
else:
    st.info("Click 'Run Simulation' to start the live plasma simulation.")

# --- RL (TD3) section ---
st.divider()
st.subheader("TD3 RL Simulation")

# Guard to avoid training on Streamlit Cloud (inference-only)
LOAD_ONLY = True

# Run button first (kept the same, only changed the model-loaded message)
if st.button("Run TD3 RL Simulation"):
    status_placeholder = st.empty()
    rl_plot_placeholder = st.empty()

    # Import TD3 lazily so the app still works without the package for non-RL mode
    try:
        from stable_baselines3 import TD3
    except Exception:
        st.error("TD3 (stable-baselines3) is not installed. Install with: pip install stable-baselines3")
        st.stop()

    # Try to load a trained model from the same paths and order as td3_test.py
    model = None
    candidate_paths = [
        "./models/td3_curriculum/Stage4_25particles/final_model",
        "./models/td3_plasma_final",
    ]
    for p in candidate_paths:
        try:
            model = TD3.load(p)
            status_placeholder.success("TD3 model loaded.")  # no file path shown
            break
        except Exception:
            continue

    if model is None:
        st.error("No trained TD3 model found. Train with td3_train_curriculum.py (recommended) or td3_train.py")
        st.stop()

    # Create RL-enabled environment exactly as td3_test.py
    env = PlasmaEnv(n_particles=25, rl_mode=True)
    obs, info = env.reset()

    # Evaluate confinement exactly like td3_test.py (5 episodes, 100 steps)
    def evaluate_confinement(model, env, n_episodes=5, n_steps=100):
        rates = []
        for _ in range(n_episodes):
            obs, info = env.reset()
            total_particles = env.n_particles
            for _ in range(n_steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, _ = env.step(action)
                if done or truncated:
                    break
            particles_lost = 0
            for i in range(total_particles):
                xy_dist = np.sqrt(env.positions[i][0]**2 + env.positions[i][1]**2)
                torus_distance = np.sqrt((xy_dist - env.major_radius)**2 + env.positions[i][2]**2)
                if torus_distance > env.minor_radius:
                    particles_lost += 1
            confined = total_particles - particles_lost
            rates.append((confined / total_particles) * 100)
        return float(np.mean(rates)), float(np.std(rates))

    status_placeholder.info("Evaluating TD3 confinement performance…")
    avg_conf, std_conf = evaluate_confinement(model, env)
    status_placeholder.success(f"TD3 Average Confinement: {avg_conf:.1f}% ± {std_conf:.1f}%")

    # Reset environment for visualization and run 100 steps like td3_test.py
    obs, info = env.reset()
    for step in range(50):  # 100 total steps, 2 at a time
        for _ in range(2):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
        # Hide magnetic field overlays (arrows and |B|) to match non-RL view
        try:
            env.current_action = None
        except Exception:
            pass
        env.render()
        fig = env.fig  # use the env's figure directly for accurate updates
        # Make legend/annotations smaller and not bolded, specifically for |B| and Magnetic Field Control
        for ax in fig.axes:
            # Adjust legend entries
            leg = ax.get_legend()
            if leg is not None:
                for text in leg.get_texts():
                    label_text = text.get_text()
                    if "|B|" in label_text or "magnetic field control" in label_text.lower():
                        text.set_fontsize(8)
                        text.set_fontweight("normal")
                # Also normalize legend title if present
                title = leg.get_title()
                if title is not None:
                    title.set_fontsize(8)
                    title.set_fontweight("normal")
            # Adjust any direct text annotations
            for txt in ax.texts:
                s = txt.get_text()
                if "|B|" in s or "magnetic field control" in s.lower():
                    txt.set_fontsize(8)
                    txt.set_fontweight("normal")
        fig.set_size_inches(6, 4)
        fig.tight_layout()
        rl_plot_placeholder.pyplot(fig, use_container_width=True)
        time.sleep(0.05)
        # td3_test.py does not break early on done/truncated during the 100-step visualization

    env.close()
    st.success("TD3-controlled simulation completed!")
