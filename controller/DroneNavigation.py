import cv2
import torch
import numpy as np
import socket
import time
import json
import matplotlib.pyplot as plt
from collections import deque
from datetime import datetime
import random 
from predictions.PredictionProcessor import PredictionProcessor


class DroneNavigation:
    def __init__(self, tello_ip, cmd_port, logs_dir="./logs"):
        self.tello_ip = tello_ip
        self.cmd_port = cmd_port
        self.tello_address = (tello_ip, cmd_port)
        self.cmd_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.cmd_socket.bind(('', cmd_port))

        # Initialize YOLO and MiDaS models
        yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

        # Set devices
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        midas_device = torch.device("mps") if torch.backends.mps.is_available() else device

        # Define target labels
        target_labels = ["vase", "apple", "sports ball"]

        # Initialize Prediction Processor
        self.prediction_processor = PredictionProcessor(
            yolo_model=yolo_model,
            midas_model=midas_model,
            midas_transforms=midas_transforms,
            target_labels=target_labels,
            device=device,
            midas_device=midas_device,
        )

        # Drone state and logging setup
        self.position = [0, 0, 0]  # [x, y, heading]
        self.landmark_map = {}  # Store landmarks
        self.drone_path = []  # Track drone path
        self.action_log = []  # Track actions
        self.landmark_log = {}  # Log landmarks

        # Logs directory
        self.logs_dir = logs_dir
        self.path_log_file = f"{logs_dir}/drone_path_log.json"
        self.actions_log_file = f"{logs_dir}/actions_log.json"
        self.landmarks_log_file = f"{logs_dir}/landmarks_log.json"

        # Drone commands
        self.action_commands = {
            "forward": "forward 50",
            "left": "left 50",
            "right": "right 50",
            "rotate": "cw 45",
        }
    # Drone control
    def send_command(self, command, retries=3):
        for attempt in range(retries):
            try:
                self.cmd_socket.sendto(command.encode('utf-8'), self.tello_address)
                response, _ = self.cmd_socket.recvfrom(1024)
                response = response.decode('utf-8', errors='ignore').strip()
                print(f"Command '{command}' Response: {response}")
                if response.lower() in ['ok', 'ok\r\n']:
                    return response
            except socket.timeout:
                print(f"Timeout on command '{command}', retrying... ({attempt + 1}/{retries})")
        return "error"
    
    def execute_action(self, action):
        """
        Execute a drone action based on the provided action name.
        """
        command = self.action_commands.get(action)
        if command:
            response = self.send_command(command)
            if response == "error":
                print(f"Action '{action}' failed.")
        else:
            print(f"Invalid action: {action}")

    # --- State management ---
    def parse_state_data(self, state_str):
        """
        Parses the Tello state data string into a dictionary.
        """
        state_dict = {}
        for item in state_str.split(';'):
            if item:
                key, value = item.split(':')
                state_dict[key.strip()] = float(value.strip()) if '.' in value else int(value.strip())
        return state_dict

    def receive_state(self):
        """
        Continuously receive state data from the Tello and save it to a log file.
        """
        with open(self.state_log_file, "a") as file:
            while True:
                try:
                    state_data, _ = self.state_socket.recvfrom(1024)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    state_str = state_data.decode('utf-8')
                    self.state_log_file = f"{logs_dir}/state_log.txt"
                    self.state_data = self.parse_state_data(state_str)

                    # Log the raw and parsed state data
                    log_entry = f"{timestamp} - Tello State:\n{state_str}\nParsed: {self.state_data}\n\n"
                    file.write(log_entry)

                    # Example: Use specific data in your control logic
                    current_yaw = self.state_data.get('yaw', 0)
                    current_height = self.state_data.get('h', 0)

                    print(f"Yaw: {current_yaw}, Height: {current_height}")
                    time.sleep(0.2)

                except Exception as e:
                    print(f"Error in receiving state: {e}")
                    break

    def update_position(self, action, time_delta=1.0, state_data=None):
        x, y, heading = self.position
        vgx = state_data.get('vgx', 0) * 0.01 if state_data else 0  # cm/s to m/s
        vgy = state_data.get('vgy', 0) * 0.01 if state_data else 0
        yaw = state_data.get('yaw', heading) if state_data else heading

        # Update position based on velocity and yaw
        x += vgx * time_delta * np.cos(np.radians(yaw))
        y += vgy * time_delta * np.sin(np.radians(yaw))
        heading = yaw

        # Log drone path
        self.drone_path.append([x, y])
        self.position = [x, y, heading]

    # Landmark mapping
    def update_landmark_map(self, frame):
        """
        Use PredictionProcessor to update landmarks based on the current frame.
        """
        predictions = self.prediction_processor.process_frame(frame)
        detected_objects = predictions["detected_objects"]
        depth_map = predictions["depth_map"]

        for obj in detected_objects:
            label = obj["label"]
            depth = obj["depth"]
            x1, y1, x2, y2 = obj["bounding_box"]

            relative_x = self.position[0] + depth * np.cos(np.radians(self.position[2]))
            relative_y = self.position[1] + depth * np.sin(np.radians(self.position[2]))

            # Save or update the landmark map
            if label not in self.landmark_map:
                self.landmark_map[label] = []
            self.landmark_map[label].append((relative_x, relative_y, depth))

        self.prune_landmark_map()

    def prune_landmark_map(self, max_size=100):
        if len(self.landmark_map) > max_size:
            oldest_key = next(iter(self.landmark_map))
            del self.landmark_map[oldest_key]

    # Logging
    def log_action(self, action):
        self.action_log.append(action)

    def log_landmarks(self):
        for label, positions in self.landmark_map.items():
            if label not in self.landmark_log:
                self.landmark_log[label] = []
            self.landmark_log[label].extend(positions)

    # Save logs to JSON
    def save_logs(self):
        try:
            with open(self.path_log_file, 'w') as f:
                json.dump(self.drone_path, f, indent=4)
            print(f"Drone path log saved to {self.path_log_file}")

            with open(self.actions_log_file, 'w') as f:
                json.dump(self.action_log, f, indent=4)
            print(f"Actions log saved to {self.actions_log_file}")

            with open(self.landmarks_log_file, 'w') as f:
                json.dump(self.landmark_log, f, indent=4)
            print(f"Landmark log saved to {self.landmarks_log_file}")

        except Exception as e:
            print(f"Error saving logs: {e}")

    # Plotting
    def plot_drone_path(self):
        path = np.array(self.drone_path)
        plt.figure(figsize=(10, 8))

        # Plot drone path
        plt.plot(path[:, 0], path[:, 1], label="Drone Path", marker="o", linestyle="-")

        # Plot landmarks
        for label, positions in self.landmark_log.items():
            positions = np.array(positions)
            plt.scatter(positions[:, 0], positions[:, 1], label=f"Landmark: {label}")

        # Annotations
        plt.title("Drone Path and Landmarks")
        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")
        plt.legend()
        plt.grid()
        plt.show()

'''
# Example Usage
if __name__ == "__main__":
    drone = DroneNavigation(tello_ip="192.168.10.1", cmd_port=8889)

    try:
        # Start drone commands
        drone.send_command("command")
        drone.send_command("takeoff")

        # Simulate actions and state updates
        actions = ["forward", "left", "right", "rotate"]
        for i in range(5):
            action = random.choice(actions)
            drone.log_action(action)
            drone.update_position(action=0, state_data={"vgx": 10, "vgy": 0, "yaw": 90})
            time.sleep(1)

        # Log landmarks
        drone.log_landmarks()

        # Save and plot logs
        drone.save_logs()
        drone.plot_drone_path()

    except KeyboardInterrupt:
        print("Landing the drone...")
        drone.send_command("land")
    finally:
        drone.cmd_socket.close()
'''