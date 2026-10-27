"""
Environment Model - Logical Zones and Sensor Placement

ETHICAL NOTE:
- Drone does NOT have direct knowledge of camera positions
- Risk is inferred probabilistically from spatial reasoning
- Simulates realistic operational awareness
"""

import numpy as np

class EnvironmentZone:
    """Defines logical zones in the environment"""
    
    TREE_ZONE = {
        'center': [20, 30, 0],
        'radius': 10,
        'occlusion_factor': 0.6  # Trees provide 60% occlusion
    }
    
    BUILDING_ZONE = {
        'center': [0, 0, 0],
        'bounds': {
            'x': [-30, 30],
            'y': [-30, 30],
            'z': [0, 50]
        }
    }
    
    OPEN_ZONE = {
        'center': [-40, -40, 0],
        'radius': 25
    }

class CameraModel:
    """
    Simulated surveillance camera model
    
    ETHICAL DESIGN:
    - Represents realistic sensor capabilities
    - Used for risk assessment, not evasion
    - Probabilistic detection model
    """
    
    def __init__(self):
        # Camera positioned on building
        self.position = np.array([10, 10, 15])
        self.fov_degrees = 90  # Field of view
        self.max_range = 100  # meters
        self.direction = np.array([1, 1, 0])  # Looking northeast
        self.direction = self.direction / np.linalg.norm(self.direction)
    
    def compute_detection_probability(self, drone_position, altitude):
        """
        Compute probability of detection based on geometry
        
        Factors:
        - Distance from camera
        - Within FOV cone
        - Altitude (higher = more visible)
        - Occlusion from environment
        """
        drone_pos = np.array(drone_position)
        
        # Distance factor
        distance = np.linalg.norm(drone_pos - self.position)
        if distance > self.max_range:
            return 0.0
        
        distance_factor = 1.0 - (distance / self.max_range)
        
        # FOV factor
        to_drone = drone_pos - self.position
        to_drone_norm = to_drone / (np.linalg.norm(to_drone) + 1e-6)
        
        angle = np.arccos(np.clip(np.dot(self.direction, to_drone_norm), -1, 1))
        fov_rad = np.radians(self.fov_degrees / 2)
        
        if angle > fov_rad:
            fov_factor = 0.0
        else:
            fov_factor = 1.0 - (angle / fov_rad)
        
        # Altitude factor (higher altitude = more exposed)
        altitude_factor = min(altitude / 50.0, 1.0)
        
        # Occlusion check (simplified)
        occlusion_factor = self._check_occlusion(drone_pos)
        
        # Combined probability
        detection_prob = (
            0.4 * distance_factor +
            0.3 * fov_factor +
            0.2 * altitude_factor +
            0.1 * (1 - occlusion_factor)
        )
        
        return np.clip(detection_prob, 0, 1)
    
    def _check_occlusion(self, position):
        """Check if position is occluded by trees"""
        tree_center = np.array(EnvironmentZone.TREE_ZONE['center'])
        tree_radius = EnvironmentZone.TREE_ZONE['radius']
        
        dist_to_tree = np.linalg.norm(position[:2] - tree_center[:2])
        
        if dist_to_tree < tree_radius:
            return EnvironmentZone.TREE_ZONE['occlusion_factor']
        
        return 0.0
    
    def get_fov_cone_points(self, num_points=50):
        """Generate points for FOV visualization"""
        angles = np.linspace(-self.fov_degrees/2, self.fov_degrees/2, num_points)
        angles_rad = np.radians(angles)
        
        # Base direction angle
        base_angle = np.arctan2(self.direction[1], self.direction[0])
        
        points = []
        for angle_offset in angles_rad:
            angle = base_angle + angle_offset
            for r in [0, self.max_range * 0.5, self.max_range]:
                x = self.position[0] + r * np.cos(angle)
                y = self.position[1] + r * np.sin(angle)
                points.append([x, y])
        
        return points

class EnvironmentModel:
    """Complete environment representation"""
    
    def __init__(self):
        self.zones = EnvironmentZone()
        self.camera = CameraModel()
    
    def get_zone_proximity(self, position):
        """Calculate proximity to different zones"""
        pos = np.array(position)
        
        # Distance to tree zone
        tree_dist = np.linalg.norm(
            pos[:2] - np.array(self.zones.TREE_ZONE['center'][:2])
        )
        tree_proximity = max(0, 1 - tree_dist / self.zones.TREE_ZONE['radius'])
        
        # Distance to building
        building_center = np.array(self.zones.BUILDING_ZONE['center'])
        building_dist = np.linalg.norm(pos - building_center)
        building_proximity = max(0, 1 - building_dist / 50)
        
        return {
            'tree': tree_proximity,
            'building': building_proximity
        }
