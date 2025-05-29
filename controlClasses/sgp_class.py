import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R
import pandas as pd

class StewartPlatform:
    def __init__(self, base_radius=30.0, platform_radius=15.0, 
                 base_angles=None, platform_angles=None, 
                 min_length=20.0, max_length=30.0):
        """
        Initialize Stewart Platform geometry
        
        Parameters:
        - base_radius: radius of base platform
        - platform_radius: radius of top platform
        - base_angles: angles (in degrees) for base attachment points (default: 60° spacing)
        - platform_angles: angles for platform attachment points (default: 60° spacing offset by 30°)
        - min_length: minimum leg length
        - max_length: maximum leg length
        """
        self.base_radius = base_radius
        self.platform_radius = platform_radius
        self.min_length = min_length
        self.max_length = max_length
        
        # Default angles for attachment points (6 points at 60° intervals)
        if base_angles is None:
            base_angles = np.arange(0, 360, 60)
        if platform_angles is None:
            platform_angles = np.arange(30, 390, 60)
            
        self.base_angles = np.radians(base_angles)
        self.platform_angles = np.radians(platform_angles)
        
        # Calculate base and platform attachment points in their local frames
        self.basex = np.array([7.5,-7.5,-25.4,-17.9,17.9,25.4])
        self.basey = np.array([25,25,-6,-19,-19,-7])
        self.basez = np.zeros_like(self.basex)

        self.platform_x = np.array([11.7,-11.7,-14.2,-2.5,2.5,14.2])
        self.platform_y = np.array([9.7,9.7,5.3,-15,-15,5.3])
        self.platform_z = np.zeros_like(self.platform_x)

        self.base_points = self._calculate_base_points(self.basex,self.basey,self.basez)
        self.platform_points = self._calculate_base_points(self.platform_x,self.platform_y,self.platform_z)
        
    def _calculate_circle_points(self, radius, angles):
        """Calculate points on a circle given radius and angles"""
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        z = np.zeros_like(x)
        return np.vstack((x, y, z)).T
    
    def _calculate_base_points(self, x,y,z):
        """Just to stack the values"""
        return np.vstack((x, y, z)).T
    
    
    
    def calculate_leg_vectors(self, position, orientation):
        """
        Calculate leg vectors for given platform position and orientation
        
        Parameters:
        - position: [x, y, z] position of platform center
        - orientation: [roll, pitch, yaw] in degrees or rotation matrix
        
        Returns:
        - leg_vectors: 6x3 array of leg vectors (platform_point - base_point)
        - leg_lengths: array of 6 leg lengths
        """
        position = np.array(position)
        
        # Handle different orientation representations
        if isinstance(orientation, (list, np.ndarray)) and len(orientation) == 3:
            # Assume [roll, pitch, yaw] in degrees
            rot = R.from_euler('xyz', orientation, degrees=True)
            rotation_matrix = rot.as_matrix()
        elif isinstance(orientation, np.ndarray) and orientation.shape == (3, 3):
            # Already a rotation matrix
            rotation_matrix = orientation
        else:
            raise ValueError("Orientation must be [roll,pitch,yaw] or 3x3 rotation matrix")
        
        # Transform platform points to global frame
        platform_global = (rotation_matrix @ self.platform_points.T).T + position
        
        # Calculate leg vectors (platform point - base point)
        leg_vectors = platform_global - self.base_points
        
        # Calculate leg lengths (norm of each vector)
        leg_lengths = np.linalg.norm(leg_vectors, axis=1)
        
        return leg_vectors, leg_lengths
    
    def check_leg_lengths(self, leg_lengths):
        """Check if leg lengths are within valid range"""
        return np.all((leg_lengths >= self.min_length) & (leg_lengths <= self.max_length))
    
    def plot_platform(self, position=None, orientation=None, ax=None):
        """Plot the Stewart platform in 3D"""
        if position is None:
            position = [0, 0, 8]  # Default position
        if orientation is None:
            orientation = [0, 0, 0]  # Default orientation
            
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        
        # Get leg vectors and lengths
        leg_vectors, leg_lengths = self.calculate_leg_vectors(position, orientation)
        platform_global = self.base_points + leg_vectors
        
        # Plot base platform
        base_points = np.vstack((self.base_points, self.base_points[0]))  # Close the circle
        ax.plot(base_points[:, 0], base_points[:, 1], base_points[:, 2], 'b-o', label='Base')
        
        # Plot top platform
        top_points = np.vstack((platform_global, platform_global[0]))  # Close the circle
        ax.plot(top_points[:, 0], top_points[:, 1], top_points[:, 2], 'r-o', label='Platform')
        
        # Plot legs
        for i in range(6):
            ax.plot([self.base_points[i, 0], platform_global[i, 0]],
                    [self.base_points[i, 1], platform_global[i, 1]],
                    [self.base_points[i, 2], platform_global[i, 2]], 'g-', linewidth=2)
        
        # Set plot limits based on leg lengths
        max_dim = np.max(leg_lengths) * 1.2
        ax.set_xlim([-max_dim, max_dim])
        ax.set_ylim([-max_dim, max_dim])
        ax.set_zlim([0, max_dim * 1.5])
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Stewart Platform')
        ax.legend()
        
        return ax
    
    def follow_trajectory(self, trajectory):
        """
        Follow a trajectory of positions/orientations
        
        Parameters:
        - trajectory: list of dicts with 'position' and 'orientation' keys
        
        Returns:
        - all_leg_lengths: list of leg lengths for each trajectory point
        - valid_flags: list indicating if each position is achievable
        """
        all_leg_lengths = []
        valid_flags = []
        
        for point in trajectory:
            _, leg_lengths = self.calculate_leg_vectors(point['position'], point['orientation'])
            all_leg_lengths.append(leg_lengths)
            valid_flags.append(self.check_leg_lengths(leg_lengths))
        
        return np.array(all_leg_lengths), np.array(valid_flags)
    
    def animate_trajectory(self, trajectory, interval=100):
        """Animate the platform following a trajectory"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        def update(frame):
            ax.clear()
            point = trajectory[frame]
            self.plot_platform(point['position'], point['orientation'], ax=ax)
            ax.set_title(f'Stewart Platform - Position {frame+1}/{len(trajectory)}')
        
        ani = FuncAnimation(fig, update, frames=len(trajectory), interval=interval)
        plt.close()  # Prevents duplicate display in notebooks
        return ani
    

# Create platform with default parameters
sp = StewartPlatform()

# Or with custom parameters
sp_custom = StewartPlatform(
    base_radius=30.0,
    platform_radius=15.0,
    min_length=20.0,
    max_length=30.0
)

# Define position [x, y, z] and orientation [roll, pitch, yaw] in degrees
position = [0.5, 0.5, 20]
orientation = [10, 5, 0]  # 10° roll, 5° pitch, 0° yaw

# Calculate leg vectors and lengths
leg_vectors, leg_lengths = sp.calculate_leg_vectors(position, orientation)
print("Leg lengths:", leg_lengths)
print("Are lengths valid?", sp.check_leg_lengths(leg_lengths))

# Plot the platform in 3D
# ax = sp.plot_platform(position, orientation)
# plt.show()


# Define a trajectory (list of positions/orientations)
trajectory = [
    {'position': [0, 0, 20], 'orientation': [0, 0, 0]},
    {'position': [1, 0, 19], 'orientation': [5, 0, 0]},
    {'position': [1, 1, 19], 'orientation': [5, 5, 0]},
    {'position': [0, 1, 18], 'orientation': [0, 5, 5]},
    {'position': [0, 0, 17], 'orientation': [0, 0, 5]},
    {'position': [0, 0, 17], 'orientation': [0, 0, 0]},
    {'position': [1, 0, 18], 'orientation': [5, 0, 0]},
    {'position': [1, 1, 19], 'orientation': [5, 5, 0]},
    {'position': [0, 1, 20], 'orientation': [0, 5, 5]},
    {'position': [0, 0, 21], 'orientation': [0, 0, 5]},
    {'position': [0, 0, 22], 'orientation': [0, 0, 0]},
    {'position': [1, 0, 23], 'orientation': [5, 0, 0]},
    {'position': [1, 1, 23], 'orientation': [5, 5, 0]},
    {'position': [0, 1, 22], 'orientation': [0, 5, 0]},
    {'position': [0, 0, 21], 'orientation': [0, 0, 5]},
]

# Obtain the cvs file
df = pd.read_csv('.\\data\\dusthon_boat.csv')
trajectory = [
    {'position': [0, 0, row['Y']+20], 'orientation': [row['Roll'],0,  row['Pitch']]}
    for _, row in df.iterrows()
]

# Calculate leg lengths for each trajectory point
all_leg_lengths, valid_flags = sp.follow_trajectory(trajectory)
print("All leg lengths:\n", all_leg_lengths)
print("Valid positions:", valid_flags)

actuators = (all_leg_lengths-22)*0.5 + 2.5

# Create CSV file with trajectory data
trajectory_df = pd.DataFrame(actuators, columns=[f'Leg_{i+1}' for i in range(6)])
trajectory_df.to_csv('.\\Data\\boat_trajectory.csv', index=False)



# Create animation (this will display in notebooks or can be saved)
# ani = sp.animate_trajectory(trajectory, interval=100)

# To display in a notebook:
# from IPython.display import HTML
# HTML(ani.to_jshtml())

# # To save as GIF (requires pillow)
# ani.save('.\\stewart_trajectory.gif', writer='pillow', fps=10)