"""
Spatial Risk Heatmap Generator

Creates 2D risk visualization for dashboard

ETHICAL DESIGN:
- Transparent risk visualization
- Supports spatial awareness
- Educational tool for understanding exposure
"""

import numpy as np
from environment import EnvironmentModel

class HeatmapEngine:
    """Generate spatial risk heatmaps"""
    
    def __init__(self, grid_size=50, map_range=100):
        self.grid_size = grid_size
        self.map_range = map_range
        self.environment = EnvironmentModel()
        
        # Pre-compute grid
        self.x_grid = np.linspace(-map_range, map_range, grid_size)
        self.y_grid = np.linspace(-map_range, map_range, grid_size)
        
        # Cache heatmap for performance
        self.cached_heatmap = None
        self.cache_valid = False
    
    def generate_heatmap(self, altitude=20):
        """
        Generate 2D risk heatmap at given altitude
        
        Returns:
            dict with x, y, z (risk values) for plotting
        """
        if self.cache_valid and hasattr(self, 'cached_altitude'):
            if self.cached_altitude == altitude:
                return self.cached_heatmap
        
        # Create meshgrid
        X, Y = np.meshgrid(self.x_grid, self.y_grid)
        Z = np.zeros_like(X)
        
        # Compute risk for each grid point
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                position = [X[i, j], Y[i, j], -altitude]  # NED coords
                
                # Compute detection probability
                risk = self.environment.camera.compute_detection_probability(
                    position,
                    altitude
                )
                
                Z[i, j] = risk
        
        # Cache result
        self.cached_heatmap = {
            'x': X,
            'y': Y,
            'z': Z
        }
        self.cached_altitude = altitude
        self.cache_valid = True
        
        return self.cached_heatmap
    
    def get_risk_at_position(self, x, y, altitude):
        """Get risk value at specific position"""
        position = [x, y, -altitude]
        risk = self.environment.camera.compute_detection_probability(
            position,
            altitude
        )
        return float(risk)
    
    def get_safe_zones(self, altitude=20, risk_threshold=0.3):
        """
        Identify safe zones (low risk areas)
        
        Returns:
            list of (x, y) coordinates for safe zones
        """
        heatmap = self.generate_heatmap(altitude)
        
        safe_zones = []
        X, Y, Z = heatmap['x'], heatmap['y'], heatmap['z']
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if Z[i, j] < risk_threshold:
                    safe_zones.append((X[i, j], Y[i, j]))
        
        return safe_zones
    
    def get_high_risk_zones(self, altitude=20, risk_threshold=0.7):
        """Identify high-risk zones"""
        heatmap = self.generate_heatmap(altitude)
        
        high_risk_zones = []
        X, Y, Z = heatmap['x'], heatmap['y'], heatmap['z']
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if Z[i, j] > risk_threshold:
                    high_risk_zones.append((X[i, j], Y[i, j]))
        
        return high_risk_zones
    
    def get_camera_fov_polygon(self):
        """Get camera FOV as polygon for visualization"""
        camera = self.environment.camera
        
        # FOV cone vertices
        fov_rad = np.radians(camera.fov_degrees / 2)
        base_angle = np.arctan2(camera.direction[1], camera.direction[0])
        
        vertices = [camera.position[:2].tolist()]  # Start at camera
        
        # Add arc points
        for angle_offset in np.linspace(-fov_rad, fov_rad, 20):
            angle = base_angle + angle_offset
            r = camera.max_range * 0.6  # Visual range
            x = camera.position[0] + r * np.cos(angle)
            y = camera.position[1] + r * np.sin(angle)
            vertices.append([x, y])
        
        vertices.append(camera.position[:2].tolist())  # Close polygon
        
        return vertices
    
    def get_zone_boundaries(self):
        """Get environment zone boundaries for visualization"""
        zones = []
        
        # Tree zone (circle)
        tree = self.environment.zones.TREE_ZONE
        tree_circle = self._generate_circle(
            tree['center'][:2],
            tree['radius'],
            num_points=30
        )
        zones.append({
            'name': 'Tree Zone',
            'type': 'circle',
            'points': tree_circle,
            'color': 'green'
        })
        
        # Building zone (rectangle)
        building = self.environment.zones.BUILDING_ZONE
        building_rect = [
            [building['bounds']['x'][0], building['bounds']['y'][0]],
            [building['bounds']['x'][1], building['bounds']['y'][0]],
            [building['bounds']['x'][1], building['bounds']['y'][1]],
            [building['bounds']['x'][0], building['bounds']['y'][1]],
            [building['bounds']['x'][0], building['bounds']['y'][0]]
        ]
        zones.append({
            'name': 'Building Zone',
            'type': 'rectangle',
            'points': building_rect,
            'color': 'gray'
        })
        
        return zones
    
    def _generate_circle(self, center, radius, num_points=30):
        """Generate circle points"""
        angles = np.linspace(0, 2*np.pi, num_points)
        points = []
        for angle in angles:
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            points.append([x, y])
        return points
