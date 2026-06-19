import threading
import time
import numpy as np
import math

class SimulatedRadarStream(threading.Thread):
    def __init__(self, cli_port, data_port, cfg_path, callback, log_callback=None):
        super().__init__()
        self.cli_port = cli_port
        self.data_port = data_port
        self.cfg_path = cfg_path
        self.callback = callback
        self.log_callback = log_callback
        self.running = False
        self.daemon = True
        self.frame_count = 0

    def log(self, msg):
        if self.log_callback:
            self.log_callback(msg)
        else:
            print(msg)

    def run(self):
        self.running = True
        self.log("[SimStream] 시뮬레이션 데이터 생성 시작 (Simulation Started)")
        
        while self.running:
            time.sleep(0.1) # 10 FPS
            self.frame_count += 1
            
            # Simulate a person walking in a circle
            angle = self.frame_count * 0.1
            cx = np.cos(angle) * 1.5
            cy = np.sin(angle) * 1.5
            cz = 0
            
            # Generate points for a cylinder (Body)
            points = []
            height = 1.7
            num_points = 50
            
            for i in range(num_points):
                h = (i / num_points) * height
                # Cylinder radius 0.2
                theta = np.random.rand() * 2 * np.pi
                r = np.random.rand() * 0.2
                
                px = cx + r * np.cos(theta)
                py = cy + r * np.sin(theta)
                pz = cz + h
                
                # Add doppler (just random or relative to movement)
                # Tangential velocity direction: (-sin, cos)
                vx = -np.sin(angle)
                vy = np.cos(angle)
                doppler = vx * np.cos(theta) + vy * np.sin(theta) # Rough approx
                
                points.append([px, py, pz, doppler])
                
            # Simulate a "Fall" periodically (every 100 frames)
            if (self.frame_count % 200) > 180:
                 # Fast drop
                 drop_progress = (self.frame_count % 200 - 180) / 20.0
                 # Z goes from 1.7 (standing) to 0.2 (lying) effectively
                 # Just flatten z
                 points = []
                 for i in range(num_points):
                    px = cx + (np.random.rand()-0.5)*1.5 # Spread out on ground
                    py = cy + (np.random.rand()-0.5)*0.5
                    pz = np.random.rand() * 0.3 # Low Z
                    points.append([px, py, pz, 0])

            # Convert to numpy
            points_np = np.array(points, dtype=np.float32)
            
            # Create a mock objects that mimics parsed structure
            class MockParsed:
                pass
            parsed = MockParsed()
            parsed.points = points_np
            # parsed.skeleton could be mocked here if we wanted explicit skeleton from "sensor"
            # but we rely on processing.py to generate it from points
            
            self.callback(parsed)

    def stop(self):
        self.running = False
