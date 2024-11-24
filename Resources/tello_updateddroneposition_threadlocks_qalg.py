import cv2
import torch
import numpy as np
import socket
import time
import threading
import queue
from datetime import datetime
import random
from collections import deque 

import torch
if torch.backends.mps.is_available():
    print("Metal (MPS) backend is available!")
else:
    print("Metal (MPS) backend is not available.")

midas_device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize YOLOv5 and MiDaS
yolo_model = torch.hub.load('/Users/josephsketl/yolov5', 'yolov5s', source='local', pretrained=True)
yolo_model.conf = 0.4
midas = torch.hub.load("/Users/josephsketl/MiDaS/", "MiDaS_small", source="local")
midas_transforms = torch.hub.load("/Users/josephsketl/MiDaS/", "transforms", source="local").small_transform

yolo_model.to(device)
midas.to(midas_device)

# Tello IP and Ports
TELLO_IP = '192.168.10.1'
CMD_PORT = 8889
VIDEO_PORT = 11111
STATE_PORT = 8890
TELLO_ADDRESS = (TELLO_IP, CMD_PORT)

# Initialize sockets
cmd_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
cmd_socket.bind(('', CMD_PORT))

# Set up UDP socket for receiving state information
state_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
state_socket.bind(('', STATE_PORT))



frame_queue = queue.Queue(maxsize=10)  # Limit queue size for memory efficiency

#--------------------------------------------------------------Receive State--------------------------------------------------------------#
def parse_state_data(state_str):
    """
    Parse the Tello state data string into a dictionary.

    Args:
        state_str (str): State data as a semicolon-separated string.

    Returns:
        dict: Parsed state data with key-value pairs.
    """
    state_dict = {}
    for item in state_str.split(';'):
        if item:
            key, value = item.split(':')
            state_dict[key.strip()] = float(value.strip()) if '.' in value else int(value.strip())
    return state_dict

def receive_state():
    """
    Continuously receive state data from the Tello and save it to a log file.
    """
    with open("tello_state_log.txt", "a") as file:
        while True:
            try:
                state_data, _ = state_socket.recvfrom(1024)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                state_str = state_data.decode('utf-8')
                state_dict = parse_state_data(state_str)

                # Log the raw and parsed state data
                log_entry = f"{timestamp} - Tello State:\n{state_str}\nParsed: {state_dict}\n\n"
                file.write(log_entry)

                # Example: Use specific data in your control logic
                current_yaw = state_dict.get('yaw', 0)
                current_height = state_dict.get('h', 0)

                print(f"Yaw: {current_yaw}, Height: {current_height}")

                time.sleep(0.2)
            except Exception as e:
                print(f"Error in receiving state: {e}")
                break
#--------------------------------------------------------------Send Command--------------------------------------------------------------#

# Function to send commands to the Tello
def send_command(command):
    cmd_socket.sendto(command.encode('utf-8'), TELLO_ADDRESS)
    response, _ = cmd_socket.recvfrom(1024)
    return response.decode('utf-8', errors='ignore')

# Function to capture video frames
def capture_video():
    video_url = f'udp://{TELLO_IP}:{VIDEO_PORT}'
    cap = cv2.VideoCapture(video_url)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    while True:
        ret, frame = cap.read()
        if ret:
           # Modify frame_queue.put() in capture_video
            if frame_queue.full():
                frame_queue.get()  # Drop the oldest frame
            frame_queue.put(frame)
        else:
            print("Failed to capture frame.")
            break

    cap.release()

def display_video():
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()

            # Retrieve predictions
            detected_objects = prediction_results.get("detected_objects", [])
            depth_map = prediction_results.get("depth_map", None)

            # Annotate the frame and display results
            annotated_frame = annotate_frame(frame, detected_objects)

            if depth_map is not None and depth_map.size > 0:
                depth_map_visual = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                combined_frame = np.hstack((annotated_frame, cv2.cvtColor(depth_map_visual, cv2.COLOR_GRAY2BGR)))
            else:
                combined_frame = annotated_frame

            cv2.imshow("Tello Video Stream with YOLO and MiDaS", combined_frame)

            # Break if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                send_command("land")
                break





def midas_depth_at_center(bounding_box, depth_map):
    """
    Get the depth value at the center of a bounding box.

    Args:
        bounding_box (tuple): The bounding box (x1, y1, x2, y2).
        depth_map (np.array): The depth map generated by MiDaS.

    Returns:
        float: The depth value at the center of the bounding box.
    """
    x1, y1, x2, y2 = bounding_box
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    if 0 <= cy < depth_map.shape[0] and 0 <= cx < depth_map.shape[1]:
        return depth_map[cy, cx]
    else:
        return float('nan')  # Return NaN if the center is out of bounds

prediction_results = {
    "detected_objects": [],
    "depth_map": None
}
results_lock = threading.Lock()
predictions_started = threading.Event()

def process_predictions():
    predictions_started.set()  # Signal that prediction processing has started
    depth_cache = {}  # Cache depth values for bounding boxes

    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()

            # Convert frame to RGB for YOLO
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run YOLOv5 detection
            yolo_results = yolo_model(rgb_frame)
            detections = yolo_results.xyxy[0]

            # Process MiDaS for depth estimation
            input_batch = midas_transforms(rgb_frame).to(midas_device)
            with torch.no_grad():
                depth_map = midas(input_batch)
                depth_map = torch.nn.functional.interpolate(
                    depth_map.unsqueeze(1),
                    size=rgb_frame.shape[:2],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze().cpu().numpy()

            # Cache depth values and process YOLO detections
            detected_objects = []
            for *box, confidence, class_id in detections:
                x1, y1, x2, y2 = map(int, box)
                bounding_box = (x1, y1, x2, y2)
                label = yolo_model.names[int(class_id)]

                # Check if depth is already cached
                if bounding_box not in depth_cache:
                    depth_cache[bounding_box] = midas_depth_at_center(bounding_box, depth_map)

                object_depth = depth_cache[bounding_box]
                detected_objects.append({
                    "label": label,
                    "confidence": confidence.item(),
                    "depth": object_depth,
                    "bounding_box": bounding_box
                })

            # Update global prediction results
            prediction_results["detected_objects"] = detected_objects
            prediction_results["depth_map"] = depth_map


def annotate_frame(frame, detected_objects):
    annotated_frame = frame.copy()

    for obj in detected_objects:
        x1, y1, x2, y2 = obj["bounding_box"]
        label = obj["label"]
        confidence = obj["confidence"]
        object_depth = obj["depth"]

        # Create distance text
        distance_text = f"{label} ({confidence:.2f}): {object_depth:.2f} depth units"

        # Draw bounding box and text
        color = (0, 255, 0)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated_frame, distance_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

    return annotated_frame


#--------------------------------------------------------------Detect Color--------------------------------------------------------------#
def detect_colored_objects(frame, detected_objects):
    """
    Detect red, green, and blue objects and match them with YOLO-detected objects.

    Args:
        frame (np.array): The input video frame in BGR format.
        detected_objects (list): List of detected objects with bounding boxes and depth.

    Returns:
        dict: A dictionary containing detected objects categorized by color.
              e.g., {"red": [...], "green": [...], "blue": [...]}
    """
    # Convert frame to HSV for color detection
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define HSV ranges for red, green, and blue
    color_ranges = {
        "red": [(np.array([0, 120, 70]), np.array([10, 255, 255]))],
        "green": [(np.array([36, 100, 100]), np.array([86, 255, 255]))],
        "blue": [(np.array([94, 80, 2]), np.array([126, 255, 255]))],
    }

    # Store detected objects for each color
    colored_objects = {color: [] for color in color_ranges.keys()}

    # Generate masks for each color and check detected objects
    for color, ranges in color_ranges.items():
        # Combine multiple ranges for colors (if needed)
        mask = np.zeros_like(hsv_frame[:, :, 0], dtype=np.uint8)
        for lower, upper in ranges:
            mask |= cv2.inRange(hsv_frame, lower, upper)

        for obj in detected_objects:
            bounding_box = obj["bounding_box"]
            label = obj["label"]
            confidence = obj["confidence"]
            depth = obj.get("depth", float('nan'))  # Default to NaN if depth is not available

            # Calculate bounding box center
            x1, y1, x2, y2 = bounding_box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Check if the center of the YOLO box is within the mask
            if 0 <= cy < mask.shape[0] and 0 <= cx < mask.shape[1] and mask[cy, cx] > 0:
                colored_objects[color].append({
                    "label": label,
                    "confidence": confidence,
                    "bounding_box": bounding_box,
                    "depth": depth
                })

    return colored_objects

#--------------------------------------------------------------Positioning--------------------------------------------------------------#

import json

def save_landmark_map(filename='landmark_map.json'):
    with open(filename, 'w') as file:
        json.dump(landmark_map, file)
    print(f"Landmark map saved to {filename}")

landmark_map = {}  # Store detected objects as {label: [(x, y, depth), ...]}

def load_landmark_map(filename='landmark_map.json'):
    global landmark_map
    try:
        with open(filename, 'r') as file:
            landmark_map = json.load(file)
        print(f"Landmark map loaded from {filename}")
    except FileNotFoundError:
        print("No saved landmark map found. Starting fresh.")
        landmark_map = {}

def prune_landmark_map(max_size=100):
    if len(landmark_map) > max_size:
        oldest_key = next(iter(landmark_map))  # Example: Remove the oldest entry
        del landmark_map[oldest_key]


def update_landmark_map(detections, depth_map, drone_position):
    """
    Update the map of detected objects with relative positions.
    
    Args:
        detections (list): YOLO detections from the current frame.
        depth_map (np.array): MiDaS depth map.
        drone_position (tuple): Estimated (x, y) position of the drone.
    """
    for *box, confidence, class_id in detections:
        x1, y1, x2, y2 = map(int, box)
        label = yolo_model.names[int(class_id)]

        # Calculate the center of the bounding box
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Estimate depth
        depth = midas_depth_at_center((x1, y1, x2, y2), depth_map)
        if np.isnan(depth):
            continue  # Skip invalid depth values

        # Calculate relative position based on depth and drone position
        relative_x = drone_position[0] + depth * np.cos(np.radians(drone_position[2]))
        relative_y = drone_position[1] + depth * np.sin(np.radians(drone_position[2]))

        # Save or update the landmark map
        if label not in landmark_map:
            landmark_map[label] = []
        landmark_map[label].append((relative_x, relative_y, depth))
        prune_landmark_map()

def update_drone_position(position, state_data, time_delta):
    x, y, heading = position
    vgx = state_data.get('vgx', 0) * 0.01  # Convert cm/s to m/s
    vgy = state_data.get('vgy', 0) * 0.01
    yaw = state_data.get('yaw', heading)

    # Update position based on velocity and yaw
    x += vgx * time_delta * np.cos(np.radians(yaw))
    y += vgy * time_delta * np.sin(np.radians(yaw))
    heading = yaw

    return [x, y, heading]

#--------------------------------------------------------------Logging Info--------------------------------------------------------------#
import matplotlib.pyplot as plt

# List to log drone positions and actions for plotting
drone_path = []  # Stores (x, y, heading)
action_log = []  # Stores actions
landmark_log = {}  # Stores landmarks as {label: [(x, y, depth), ...]}

def log_drone_path(drone_position, action):
    """
    Log the drone's position and action for later visualization.

    Args:
        drone_position (list): Current [x, y, heading] position.
        action (int): Action index (0 = forward, 1 = left, 2 = right, 3 = rotate).
    """
    drone_path.append(drone_position[:2])  # Log only x, y for path
    action_log.append(actions[action])  # Log the action name

def log_landmarks():
    """
    Update the global landmark_log with the current landmark map.
    """
    for label, positions in landmark_map.items():
        if label not in landmark_log:
            landmark_log[label] = []
        landmark_log[label].extend(positions)  # Append new positions to log

def plot_drone_path():
    """
    Plot the drone's path and landmarks using matplotlib.
    """
    # Convert drone path to numpy array for easier manipulation
    path = np.array(drone_path)
    plt.figure(figsize=(10, 8))

    # Plot drone path
    plt.plot(path[:, 0], path[:, 1], label="Drone Path", marker="o", linestyle="-")

    # Plot landmarks
    for label, positions in landmark_log.items():
        positions = np.array(positions)
        plt.scatter(positions[:, 0], positions[:, 1], label=f"Landmark: {label}")

    # Add annotations and labels
    plt.title("Drone Path and Landmarks")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.legend()
    plt.grid()
    plt.show()

DRONE_PATH_LOG_FILE = "/Users/josephsketl/Docs/notes/School/MSML642/Project/logs/drone_path_log.json"
ACTIONS_LOG_FILE = "/Users/josephsketl/Docs/notes/School/MSML642/Project/logs/actions_log.json"
LANDMARKS_LOG_FILE = "/Users/josephsketl/Docs/notes/School/MSML642/Project/logs/landmarks_log.json"

def save_logs():
    """
    Save the drone path, actions, and landmark logs to JSON files.
    """
    try:
        # Save drone path log
        with open(DRONE_PATH_LOG_FILE, 'w') as f:
            json.dump(drone_path, f, indent=4)
        print(f"Drone path log saved to {DRONE_PATH_LOG_FILE}")

        # Save action log
        with open(ACTIONS_LOG_FILE, 'w') as f:
            json.dump(action_log, f, indent=4)
        print(f"Actions log saved to {ACTIONS_LOG_FILE}")

        # Save landmark log
        with open(LANDMARKS_LOG_FILE, 'w') as f:
            json.dump(landmark_log, f, indent=4)
        print(f"Landmark log saved to {LANDMARKS_LOG_FILE}")

    except Exception as e:
        print(f"Error saving logs: {e}")
#--------------------------------------------------------------Q-Learning--------------------------------------------------------------#

actions = ['forward', 'left', 'right', 'cw']
Q = np.zeros((100, len(actions)))

alpha = 0.1
gamma = 0.9
epsilon = 0.1


target_color = None  # Set this to "red", "blue", or "green" via some command
depth_threshold = 700  # Stop rewarding once the depth is below this value
proximity_penalty = -5  # Penalty for moving toward incorrect objects
lost_target_penalty = -10  # Penalty for losing the target

# Parameters for handling variations
missing_frames_tolerance = 5  # Number of frames to allow without detecting the target color
depth_window_size = 5         # Number of frames for averaging depth

# Queues to track missing frames and depth
missing_frames_counter = 0
depth_history = deque(maxlen=depth_window_size)

# Track cumulative rewards for dynamic exploration
cumulative_reward_window = deque(maxlen=10)  # Store rewards from the last 10 steps

# Load the Q-table
'''
def load_q_table(filename='q_table.npy'):
    try:
        global Q
        Q = np.load(filename)
        print("Q-table loaded from", filename)
    except FileNotFoundError:
        print("No saved Q-table found. Starting with a fresh Q-table.")
        Q = np.zeros((100, len(actions)))  # Initialize with zeros if no file found
# Save the Q-table
def save_q_table(filename='q_table.npy'):
    np.save(filename, Q)
    print("Q-table saved to", filename)
'''
import os
import numpy as np

def load_q_table(folder='.', filename='q_table.npy'):
    """
    Load the Q-table from a specified folder.

    Args:
        folder (str): The folder path where the Q-table is stored.
        filename (str): The name of the Q-table file.

    Returns:
        None
    """
    global Q
    file_path = os.path.join(folder, filename)
    try:
        Q = np.load(file_path)
        print(f"Q-table loaded from {file_path}")
    except FileNotFoundError:
        print(f"No saved Q-table found at {file_path}. Starting with a fresh Q-table.")
        Q = np.zeros((100, len(actions)))  # Initialize with zeros if no file found


def save_q_table(folder='.', filename='q_table.npy'):
    """
    Save the Q-table to a specified folder.

    Args:
        folder (str): The folder path where the Q-table will be saved.
        filename (str): The name of the Q-table file.

    Returns:
        None
    """
    os.makedirs(folder, exist_ok=True)  # Ensure the folder exists
    file_path = os.path.join(folder, filename)
    np.save(file_path, Q)
    print(f"Q-table saved to {file_path}")


def take_action(state):
    if random.uniform(0, 1) < epsilon:
        action = random.choice(range(len(actions)))
    else:
        action = np.argmax(Q[state])

    # Execute the action and print it
    if action == 0:  # forward
        send_command("forward 15")
    elif action == 1:  # left
        send_command("left 15")
    elif action == 2:  # right
        send_command("right 15")
    elif action == 3:  # cw (rotate)
        send_command("cw 45")

    return action

# Obstacle Avoidance:
def avoid_obstacle_action(closest_object, drone_position):
    """
    Select an action to avoid the closest obstacle.

    Args:
        closest_object (dict): Metadata of the closest detected object.
        drone_position (list): Current [x, y, heading] position of the drone.

    Returns:
        int: Action index to avoid the obstacle.
    """
    x1, y1, x2, y2 = closest_object["bounding_box"]
    obj_center_x = (x1 + x2) // 2
    obj_center_y = (y1 + y2) // 2

    # Calculate avoidance direction
    if obj_center_x < drone_position[0]:  # Obstacle on the left
        print("Obstacle on the left. Moving right.")
        return 2  # Right action
    elif obj_center_x > drone_position[0]:  # Obstacle on the right
        print("Obstacle on the right. Moving left.")
        return 1  # Left action
    else:
        print("Obstacle directly ahead. Rotating.")
        return 3  # Rotate clockwise
    
# Q-learning update function
def update_q(state, action, reward, next_state):
    best_next_action = np.argmax(Q[next_state])
    Q[state, action] += alpha * (reward + gamma * Q[next_state, best_next_action] - Q[state, action])

# Flag to indicate if the Q-learning loop has started
loop_started = threading.Event()

def q_learning_loop():
    """
    Q-learning loop for autonomous flight with fallback exploration and obstacle avoidance.
    """
    global epsilon
    try:
        send_command('takeoff')  # Take off command
        send_command('up 50')
        time.sleep(1)
        state = 0
        drone_position = [0, 0, 0]  # x, y, heading (initialize at origin)
        missing_frames_counter = 0  # Initialize here
        while True:
            if frame_queue.empty():
                continue

            # Get the current frame
            frame = frame_queue.get()

            # Access shared predictions
            with results_lock:
                detected_objects = prediction_results["detected_objects"]
                depth_map = prediction_results["depth_map"]

            if not detected_objects or depth_map is None:
                print("Waiting for predictions...")
                time.sleep(0.05)
                continue

            if detected_objects and depth_map is not None:
                update_landmark_map(detected_objects, depth_map, drone_position)
                log_landmarks()
                
            # Detect colored objects
            colored_objects = detect_colored_objects(frame, detected_objects)

            # Determine reward and actions
            reward = 0
            found_landmark = False

            if target_color in colored_objects:
                target_objects = colored_objects[target_color]
                for obj in target_objects:
                    if np.isnan(obj["depth"]):
                        continue

                # Choose the closest target
                closest_object = min(target_objects, key=lambda o: o["depth"])
                target_depth = closest_object["depth"]
                depth_history.append(target_depth)

                smoothed_depth = sum(depth_history) / len(depth_history)
                print(f"Smoothed Target Depth: {smoothed_depth}")

                reward += 10 if smoothed_depth > depth_threshold else 20
                found_landmark = True

            # Obstacle avoidance logic
            obstacle_detected = False
            for obj in detected_objects:
                if obj["depth"] < 700:  # If an object is closer than the threshold
                    print(f"Obstacle detected: {obj['label']} at depth {obj['depth']}. Avoiding...")
                    action = avoid_obstacle_action(obj, drone_position)
                    obstacle_detected = True
                    break

            # Fallback exploration or take action
            if not found_landmark and not obstacle_detected:
                if missing_frames_counter > missing_frames_tolerance:
                    print("No landmarks detected. Switching to fallback exploration.")
                    action = random.choice(range(len(actions)))  # Random action for exploration
                else:
                    action = take_action(state)

            # Print the chosen action
            print(f"Action taken: {actions[action]}")
            log_drone_path(drone_position, action)

            # Execute the action
            if action == 0:  # forward
                send_command("forward 15")
            elif action == 1:  # left
                send_command("left 15")
            elif action == 2:  # right
                send_command("right 15")
            elif action == 3:  # cw (rotate)
                send_command("cw 45")

            # Update drone position and Q-table
            next_state = (state + 1) % 100
            update_q(state, action, reward, next_state)
            state = next_state

    except Exception as e:
        print(f"An error occurred in the Q-learning loop: {e}")
    finally:
        send_command("land")
        save_q_table(folder='/Users/josephsketl/Docs/notes/School/MSML642/Project/logs', filename='my_q_table.npy')
        save_logs()  # Save logs at the end
        print("Main Q-learning loop completed.")

if __name__ == "__main__":
    try:
        load_q_table(folder='/Users/josephsketl/Docs/notes/School/MSML642/Project/logs', filename='my_q_table.npy')  # Load the Q-table from file
        load_landmark_map()
        send_command('command')
        time.sleep(1)
        send_command('streamon')
        time.sleep(1)

        # Start threads
        threading.Thread(target=capture_video, daemon=True).start()
        threading.Thread(target=process_predictions, daemon=True).start()

        # Wait for predictions to start
        print("Waiting for predictions to initialize...")
        predictions_started.wait()  # Wait until predictions are initialized
        print("Predictions initialized. Starting display and Q-learning loop.")

        # Run display in the main thread
        display_video()
        q_learning_loop()
    except KeyboardInterrupt:
        print("Keyboard Interrupt: Landing the drone...")
        save_logs()  # Save logs at the end
    finally:
        send_command('streamoff')
        save_q_table(folder='/Users/josephsketl/Docs/notes/School/MSML642/Project/logs', filename='my_q_table.npy')
        cmd_socket.close()
