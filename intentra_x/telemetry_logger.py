"""
Telemetry Logger

Records flight data for analysis and replay

ETHICAL DESIGN:
- Full data transparency
- Audit trail for decisions
- Supports post-mission analysis
"""

import json
import csv
from datetime import datetime
import os

class TelemetryLogger:
    """Log telemetry and decision data"""
    
    def __init__(self, log_dir='intentra_x/logs'):
        self.log_dir = log_dir
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = None
        self.csv_file = None
        self.csv_writer = None
        
        self._initialize_logging()
    
    def _initialize_logging(self):
        """Set up logging infrastructure"""
        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)
        
        # JSON log file
        json_path = os.path.join(self.log_dir, f'session_{self.session_id}.json')
        self.log_file = open(json_path, 'w')
        self.log_file.write('[\n')
        self.first_entry = True
        
        # CSV log file
        csv_path = os.path.join(self.log_dir, f'session_{self.session_id}.csv')
        self.csv_file = open(csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # CSV header
        self.csv_writer.writerow([
            'timestamp', 'pos_x', 'pos_y', 'pos_z',
            'vel_x', 'vel_y', 'vel_z',
            'altitude', 'speed',
            'intent', 'intent_confidence',
            'uncertainty', 'risk_score',
            'state', 'counterfactual_risk'
        ])
    
    def log(self, data):
        """Log telemetry data"""
        # JSON log
        if not self.first_entry:
            self.log_file.write(',\n')
        else:
            self.first_entry = False
        
        json.dump(data, self.log_file, indent=2)
        self.log_file.flush()
        
        # CSV log
        if self.csv_writer:
            row = [
                data.get('timestamp', 0),
                data['position'][0], data['position'][1], data['position'][2],
                data['velocity'][0], data['velocity'][1], data['velocity'][2],
                data.get('altitude', 0),
                data.get('speed', 0),
                data.get('intent', ''),
                data.get('intent_confidence', 0),
                data.get('uncertainty', 0),
                data.get('risk_score', 0),
                data.get('state', ''),
                data.get('counterfactual_risk', 0)
            ]
            self.csv_writer.writerow(row)
            self.csv_file.flush()
    
    def close(self):
        """Close log files"""
        if self.log_file:
            self.log_file.write('\n]')
            self.log_file.close()
        
        if self.csv_file:
            self.csv_file.close()
    
    def __del__(self):
        """Cleanup on deletion"""
        self.close()

class DataReplay:
    """Replay logged data for analysis"""
    
    def __init__(self, log_file):
        self.log_file = log_file
        self.data = None
    
    def load(self):
        """Load log file"""
        if self.log_file.endswith('.json'):
            with open(self.log_file, 'r') as f:
                self.data = json.load(f)
        elif self.log_file.endswith('.csv'):
            self.data = []
            with open(self.log_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.data.append(row)
        
        return self.data
    
    def get_trajectory(self):
        """Extract trajectory from log"""
        if not self.data:
            return []
        
        trajectory = []
        for entry in self.data:
            if isinstance(entry, dict):
                if 'position' in entry:
                    trajectory.append(entry['position'])
                elif 'pos_x' in entry:
                    trajectory.append([
                        float(entry['pos_x']),
                        float(entry['pos_y']),
                        float(entry['pos_z'])
                    ])
        
        return trajectory
    
    def get_risk_timeline(self):
        """Extract risk over time"""
        if not self.data:
            return []
        
        timeline = []
        for entry in self.data:
            if isinstance(entry, dict):
                timestamp = entry.get('timestamp', 0)
                risk = entry.get('risk_score', 0)
                timeline.append((timestamp, float(risk)))
        
        return timeline
