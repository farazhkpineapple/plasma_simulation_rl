import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PlasmaEnv(gym.Env):
    """
    3D plasma distribution inside a torus-shaped to        
        # Force the plot to update
        plt.draw()
        plt.pause(0.01)    Multiple plasma particles form a ring around the torus.
    Agent applies magnetic field controls to keep plasma confined.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, n_particles=50, rl_mode=False):
        super(PlasmaEnv, self).__init__()

        # Number of plasma particles to simulate the plasma ring
        self.n_particles = n_particles
        self.rl_mode = rl_mode  # Track if we're using RL control
        self.current_action = None  # Store current magnetic field action
        
        # --- State: average position and velocity of plasma ring ---
        # We'll track the center of mass and average velocity
        low = np.array([-5.0, -5.0, -5.0, -2.0, -2.0, -2.0], dtype=np.float32)
        high = np.array([5.0, 5.0, 5.0, 2.0, 2.0, 2.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # --- Action space: continuous magnetic field controls ---
        # Actions are [Bx, By, Bz] values between -1 and 1
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Torus dimensions
        self.major_radius = 3.0  # distance from origin to centerline of tube
        self.minor_radius = 1.0  # tube radius

        self.dt = 0.1
        self.max_steps = 500  # Increased from 200 for longer episodes but still reasonable
        self.step_count = 0

        # Plasma particle arrays
        self.positions = np.zeros((self.n_particles, 3))  # [n_particles, 3]
        self.velocities = np.zeros((self.n_particles, 3))  # [n_particles, 3]

        # Plotting
        self.fig = None
        self.ax = None

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize plasma particles in a ring around the torus
        # Distribute particles around the toroidal direction (phi)
        phi_angles = np.linspace(0, 2*np.pi, self.n_particles, endpoint=False)
        
        for i in range(self.n_particles):
            phi = phi_angles[i]
            # Add some randomness in the poloidal direction (theta) and radial position
            theta = np.random.uniform(-0.3, 0.3)  # Small poloidal variation
            r_variation = np.random.uniform(-0.2, 0.2)  # Small radial variation
            
            # Convert to Cartesian coordinates
            R = self.major_radius + r_variation
            x = R * np.cos(phi) + 0.3 * np.cos(theta) * np.cos(phi)
            y = R * np.sin(phi) + 0.3 * np.cos(theta) * np.sin(phi)
            z = 0.3 * np.sin(theta)
            
            self.positions[i] = [x, y, z]
            # Small initial velocities with some toroidal component
            self.velocities[i] = np.random.uniform(-0.1, 0.1, size=3)
            self.velocities[i][0] += -0.05 * np.sin(phi)  # toroidal velocity
            self.velocities[i][1] += 0.05 * np.cos(phi)   # toroidal velocity
        
        self.step_count = 0
        
        # Return center of mass and average velocity as observation
        center_of_mass = np.mean(self.positions, axis=0)
        avg_velocity = np.mean(self.velocities, axis=0)
        return np.concatenate((center_of_mass, avg_velocity)).astype(np.float32), {}

    def step(self, action):
        # Store the current action for visualization
        self.current_action = action.copy() if hasattr(action, 'copy') else action
        
        # Convert continuous action to magnetic field strengths
        max_field_strength = 0.2  # Increased from 0.1 for stronger control
        magnetic_field = max_field_strength * action  # action is [Bx, By, Bz] in [-1, 1]

        # Update each particle
        for i in range(self.n_particles):
            # Apply magnetic field effect (Lorentz force: F = q(v × B))
            # Simplified: magnetic field creates force perpendicular to velocity
            if np.linalg.norm(magnetic_field) > 0:
                cross_product = np.cross(self.velocities[i], magnetic_field)
                self.velocities[i] += 0.2 * cross_product  # Increased from 0.1 for stronger effect
            
            # Apply some drift and diffusion
            self.velocities[i] += np.random.normal(0, 0.005, size=3)  # thermal motion
            
            # Update position
            self.positions[i] += self.velocities[i] * self.dt
            
            # Add small random thermal motion to positions
            self.positions[i] += np.random.normal(0, 0.01, size=3)

        self.step_count += 1

        # Calculate observation (center of mass and average velocity)
        center_of_mass = np.mean(self.positions, axis=0)
        avg_velocity = np.mean(self.velocities, axis=0)
        obs = np.concatenate((center_of_mass, avg_velocity)).astype(np.float32)

        # Calculate reward based on plasma confinement
        # Reward based on how well particles stay within the torus
        distances_from_torus = []
        particles_lost = 0
        
        for i in range(self.n_particles):
            xy_dist = np.sqrt(self.positions[i][0]**2 + self.positions[i][1]**2)
            torus_distance = np.sqrt((xy_dist - self.major_radius)**2 + self.positions[i][2]**2)
            distances_from_torus.append(torus_distance)
            
            if torus_distance > self.minor_radius:
                particles_lost += 1

        # Reward is based on average confinement and penalize lost particles
        avg_distance = np.mean(distances_from_torus)
        
        # More encouraging reward structure for better learning
        confinement_ratio = (self.n_particles - particles_lost) / self.n_particles
        
        # Base reward for confinement (positive for good confinement)
        confinement_reward = 100.0 * confinement_ratio  # 0-100 points
        
        # Penalty for average distance from ideal position
        distance_penalty = -10.0 * avg_distance
        
        # Bonus for maintaining tight formation
        plasma_spread = np.std(distances_from_torus)
        formation_bonus = -5.0 * plasma_spread
        
        # Huge bonus for high confinement
        if confinement_ratio > 0.9:  # 90%+ confinement
            confinement_reward += 200.0  # Big bonus!
        elif confinement_ratio > 0.8:  # 80%+ confinement
            confinement_reward += 100.0  # Good bonus
            
        reward = confinement_reward + distance_penalty + formation_bonus

        # Episode ends if too many particles are lost or max steps reached
        done = particles_lost > self.n_particles * 0.3 or self.step_count >= self.max_steps  # 30% loss threshold
        truncated = self.step_count >= self.max_steps
        
        return obs, reward, done, truncated, {}

    def render(self, mode="human"):
        if self.fig is None or self.ax is None:
            self.fig = plt.figure(figsize=(10, 8))
            self.ax = self.fig.add_subplot(111, projection='3d')
            # Show the window immediately
            plt.show(block=False)
            plt.draw()

        self.ax.clear()
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.set_zlim(-2, 2)
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.set_zlabel('Z (meters)')
        
        title = 'RL-Controlled Plasma Confinement in Tokamak' if self.rl_mode else 'Simulated Random Plasma Drift in Tokamak (Without RL)'
        self.ax.set_title(title)
        
        # Plot the tokamak torus structure
        self._plot_torus_shell(self.ax)
        
        # Plot all plasma particles
        # Color particles based on their distance from ideal torus position
        colors = []
        sizes = []
        for i in range(self.n_particles):
            xy_dist = np.sqrt(self.positions[i][0]**2 + self.positions[i][1]**2)
            torus_distance = np.sqrt((xy_dist - self.major_radius)**2 + self.positions[i][2]**2)
            
            if torus_distance > self.minor_radius:
                colors.append('red')      # Lost particles
                sizes.append(30)
            elif torus_distance > 0.7 * self.minor_radius:
                colors.append('orange')   # Particles near the boundary
                sizes.append(40)
            else:
                colors.append('orange')   # Well-confined particles
                sizes.append(50)
        
        # Plot plasma particles
        self.ax.scatter(self.positions[:, 0], self.positions[:, 1], self.positions[:, 2], 
                       c=colors, s=sizes, alpha=0.8, label='Plasma')
        
        # Add center of mass indicator
        center_of_mass = np.mean(self.positions, axis=0)
        self.ax.scatter(center_of_mass[0], center_of_mass[1], center_of_mass[2], 
                       c='blue', s=100, marker='o', alpha=0.9, label='Center of Mass')
        
        self.ax.legend()
        
        # Add some statistics as text
        particles_confined = sum(1 for c in colors if c != 'red')
        confinement_percentage = (particles_confined / self.n_particles) * 100
        self.ax.text2D(0.02, 0.95, f'Confined: {particles_confined}/{self.n_particles}', 
                      transform=self.ax.transAxes)
        self.ax.text2D(0.02, 0.90, f'Confinement: {confinement_percentage:.1f}%', 
                      transform=self.ax.transAxes)
        self.ax.text2D(0.02, 0.85, f'Step: {self.step_count}', 
                      transform=self.ax.transAxes)
        
        # Add magnetic field visualization for RL mode
        if self.rl_mode and self.current_action is not None:
            # Disabled arrows and |B| overlay per Streamlit app request
            pass
        
        # Force the plot to update
        plt.draw()
        plt.pause(0.01)

    def _plot_torus_shell(self, ax, num_points=30):
        if ax is None:
            ax = self.ax
            
        R = self.major_radius
        r = self.minor_radius
        u = np.linspace(0, 2 * np.pi, num_points)
        v = np.linspace(0, 2 * np.pi, num_points)
        u, v = np.meshgrid(u, v)
        x = (R + r * np.cos(v)) * np.cos(u)
        y = (R + r * np.cos(v)) * np.sin(u)
        z = r * np.sin(v)
        
        # Plot the torus with a more realistic tokamak appearance
        ax.plot_surface(x, y, z, alpha=0.15, color='lightblue', linewidth=0)
        
        # Add magnetic field coil representations (simplified)
        # Toroidal coils
        coil_angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        for angle in coil_angles:
            coil_x = []
            coil_y = []
            coil_z = []
            phi = np.linspace(0, 2*np.pi, 20)
            for p in phi:
                x_coil = (R + (r + 0.3) * np.cos(p)) * np.cos(angle)
                y_coil = (R + (r + 0.3) * np.cos(p)) * np.sin(angle)
                z_coil = (r + 0.3) * np.sin(p)
                coil_x.append(x_coil)
                coil_y.append(y_coil)
                coil_z.append(z_coil)
            ax.plot(coil_x, coil_y, coil_z, color='gray', alpha=0.6, linewidth=1)

    def _plot_magnetic_field_arrows(self, ax):
        """Plot arrows showing the applied magnetic field from RL agent"""
        if ax is None:
            ax = self.ax
            
        if self.current_action is None:
            return
            
        # Get magnetic field components (scaled for visualization)
        Bx, By, Bz = self.current_action
        
        # Scale arrows based on field strength (make them visible but not overwhelming)
        scale = 4.0  # increased from 2.0 for better visibility in Streamlit
        arrow_length_x = Bx * scale
        arrow_length_y = By * scale  
        arrow_length_z = Bz * scale
        
        # Position arrows around the plasma center of mass
        center_of_mass = np.mean(self.positions, axis=0)
        
        # Only show arrows if the magnetic field is significant
        field_strength = np.sqrt(Bx**2 + By**2 + Bz**2)
        if field_strength > 0.01:  # lowered threshold from 0.1 so arrows show for small fields
            
            # Show individual field components as colored arrows
            arrow_positions = [
                center_of_mass + np.array([1.5, 0, 0]),  # Right side
                center_of_mass + np.array([-1.5, 0, 0]), # Left side  
                center_of_mass + np.array([0, 1.5, 0]),  # Top
            ]
            
            colors = ['red', 'green', 'blue']
            components = [arrow_length_x, arrow_length_y, arrow_length_z]
            directions = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
            labels = ['Bx', 'By', 'Bz']
            
            for i, (pos, color, component, direction, label) in enumerate(zip(arrow_positions, colors, components, directions, labels)):
                if abs(component) > 0.01:  # show small components too
                    end_pos = pos + direction * component
                    ax.quiver(pos[0], pos[1], pos[2],
                                 direction[0] * component, direction[1] * component, direction[2] * component,
                                 color=color, alpha=0.8, arrow_length_ratio=0.3, linewidth=2)
                    
                    # Add labels for the magnetic field components
                    ax.text(pos[0], pos[1], pos[2], f'{label}={component:.2f}', 
                               color=color, fontsize=7, weight='normal')
        
        # Add overall field strength indicator (smaller, not bold)
        ax.text2D(0.02, 0.80, f'|B| field: {field_strength:.2f}', 
                      transform=ax.transAxes, color='purple', fontsize=8, weight='normal')
