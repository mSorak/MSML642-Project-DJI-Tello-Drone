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
import json

# Check to see if Metal can be used (Apple GPU)

if torch.backends.mps.is_available():
    print("Metal (MPS) backend is available!")
else:
    print("Metal (MPS) backend is not available.")

# Set MiDAS device to mps, YOLO to cpu

midas_device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # If using NVIDIA GPU

# Initialize YOLOv5 and MiDaS
yolo_model = torch.hub.load('/Users/josephsketl/yolov5', 'yolov5s', source='local', pretrained=True)  # https://github.com/ultralytics/yolov5.git
yolo_model.conf = 0.4
midas = torch.hub.load("/Users/josephsketl/MiDaS/", "MiDaS_small", source="local") # https://github.com/isl-org/MiDaS.git
midas_transforms = torch.hub.load("/Users/josephsketl/MiDaS/", "transforms", source="local").small_transform

yolo_model.to(device)
midas.to(midas_device)

# Tello IP and Ports
TELLO_IP = '192.168.10.1'
CMD_PORT = 8889
VIDEO_PORT = 11111
TELLO_ADDRESS = (TELLO_IP, CMD_PORT)

# Initialize sockets
cmd_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
cmd_socket.bind(('', CMD_PORT))

frame_queue = queue.Queue(maxsize=10)  # Limit queue size for memory efficiency

# Function to send commands to the Tello
def send_command(command, retries=3):
    """
    Send a command to the Tello drone and wait for a response.
    
    Args:
        command (str): The command string to send.
        retries (int): Number of retries if the command fails.
        
    Returns:
        str: Response from the drone.
    """
    for attempt in range(retries):
        try:
            cmd_socket.sendto(command.encode('utf-8'), TELLO_ADDRESS)
            response, _ = cmd_socket.recvfrom(1024) 
            response = response.decode('utf-8', errors='ignore')
            print(f"Command '{command}' Response: {response}")
            if response.lower() == 'ok' or response.lower() == 'ok\r\n':
                return response
        except socket.timeout:
            print(f"Timeout on command '{command}', retrying... ({attempt + 1}/{retries})")
    print(f"Failed to send command '{command}' after {retries} retries.")
    return "error"


#------------------------------------------------------------------Video--------------------------------------------------------------#

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
            if frame_queue.full():
                frame_queue.get()  # Drop the oldest frame
            frame_queue.put(frame)
        else:
            print("Failed to capture frame.")
            break

    cap.release()

frame_count = 0

def display_video():
    global frame_count
    while True:
        if not frame_queue.empty():
            frame_count += 1
            if frame_count % 5 != 0:  # Skip frames to reduce processing
                frame_queue.get()  # Discard the frame
                continue

            frame = frame_queue.get()

            # Retrieve latest predictions
            detected_objects = prediction_results["detected_objects"]

            # Annotate the frame with YOLO detections and depth estimation
            annotated_frame = annotate_frame(frame, detected_objects)

            # Display the annotated YOLO camera feed
            cv2.imshow("Tello Video Stream with YOLO and Depth Estimation", annotated_frame)

            # Handle 'q' key press to quit and land the drone
            if cv2.waitKey(1) & 0xFF == ord('q'):
                send_command('land')
                break

def annotate_frame(frame, detected_objects):
    """
    Annotates the frame with bounding boxes, labels, and depth information.

    Args:
        frame (numpy.ndarray): The original video frame.
        detected_objects (list): List of detected objects with their metadata.

    Returns:
        numpy.ndarray: Annotated frame.
    """
    # Make a copy of the frame to draw annotations on
    annotated_frame = frame.copy()

    for obj in detected_objects:
        x1, y1, x2, y2 = obj["bounding_box"]
        label = obj["label"]
        confidence = obj["confidence"]
        object_depth = obj["depth"]

        # Create the distance text
        distance_text = f"{label} ({confidence:.2f}): {object_depth:.2f} depth units"

        # Define color for bounding box
        color = (0, 255, 0)

        # Draw bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

        # Add label and depth information above the bounding box
        cv2.putText(
            annotated_frame, distance_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

    return annotated_frame

#-------------------------------------------------------------------Prediction--------------------------------------------------------------#

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
# List of target labels to be used by Tello in Q-Learning
TARGET_LABELS = ["vase", "apple", "sports ball"]

def process_predictions():
    depth_cache = {}

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

            # Filter detections for target labels
            detected_objects = []
            for *box, confidence, class_id in detections:
                label = yolo_model.names[int(class_id)]
                if label in TARGET_LABELS:
                    x1, y1, x2, y2 = map(int, box)
                    object_depth = midas_depth_at_center((x1, y1, x2, y2), depth_map)
                    detected_objects.append({
                        "label": label,
                        "confidence": confidence.item(),
                        "depth": object_depth,
                        "bounding_box": (x1, y1, x2, y2)
                    })
                    #print(f"Detected: {label}, Confidence: {confidence:.2f}, Depth: {object_depth:.2f} units")

            prediction_results["detected_objects"] = detected_objects
            prediction_results["depth_map"] = depth_map
                          


#--------------------------------------------------------------Detect Color--------------------------------------------------------------#
def detect_colored_objects(frame, yolo_detections):
    """
    Detect red, green, and blue objects and match them with YOLO-detected objects.
    
    Args:
        frame (np.array): The input video frame in BGR format.
        yolo_detections (list): List of YOLO-detected objects, each with a bounding box.
        
    Returns:
        dict: A dictionary containing detected objects categorized by color.
              e.g., {"red": [...], "green": [...], "blue": [...]}
    
    Currently not using this function due to issues with camera sensing colors.
    """
    # Convert frame to HSV for color detection
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define HSV ranges for red, green, and blue
    color_ranges = {
        "red": [
            (np.array([0, 120, 70]), np.array([10, 255, 255])),  # Lower red
            (np.array([170, 120, 70]), np.array([180, 255, 255]))  # Upper red
        ],
        "green": [
            (np.array([36, 100, 100]), np.array([86, 255, 255]))
        ],
        "blue": [
            (np.array([94, 80, 20]), np.array([126, 255, 255]))
        ]
    }

    # Store detected objects for each color
    colored_objects = {color: [] for color in color_ranges.keys()}

    # Generate masks for each color and check YOLO detections
    for color, ranges in color_ranges.items():
        mask = np.zeros_like(hsv_frame[:, :, 0], dtype=np.uint8)
        for lower, upper in ranges:
            mask |= cv2.inRange(hsv_frame, lower, upper)

        for *box, confidence, class_id in yolo_detections:
            x1, y1, x2, y2 = map(int, box)
            
            # Calculate YOLO bounding box center
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Check if the center of the YOLO box is within the mask
            if 0 <= cy < mask.shape[0] and 0 <= cx < mask.shape[1] and mask[cy, cx] > 0:
                label = yolo_model.names[int(class_id)]
                colored_objects[color].append({
                    "label": label,
                    "confidence": confidence.item(),
                    "bounding_box": (x1, y1, x2, y2)
                })

    return colored_objects


#--------------------------------------------------------------Positioning--------------------------------------------------------------#


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
    except json.JSONDecodeError:
        print(f"Error: {filename} contains invalid JSON. Starting with an empty landmark map.")
        landmark_map = {}
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
        relative_x = float(drone_position[0] + depth * np.cos(np.radians(drone_position[2])))
        relative_y = float(drone_position[1] + depth * np.sin(np.radians(drone_position[2])))
        depth = float(depth)  # Convert depth to Python float

        # Save or update the landmark map
        if label not in landmark_map:
            landmark_map[label] = []
        landmark_map[label].append((relative_x, relative_y, depth))
        prune_landmark_map()


def update_drone_position(position, action):
    """
    Update the drone's position based on the action taken.

    Args:
        position (list): Current [x, y, heading] of the drone.
        action (int): Action index (0 = forward, 1 = left, 2 = right, 3 = cw).

    Returns:
        list: Updated [x, y, heading] position.
    """
    x, y, heading = position
    distance = 0.20  # 20 cm in meters (conversion factor)

    if action == 0:  # Forward
        x += distance * np.cos(np.radians(heading))
        y += distance * np.sin(np.radians(heading))
    elif action == 1:  # Left 
        x += distance * np.cos(np.radians(heading + 90))
        y += distance * np.sin(np.radians(heading + 90))
    elif action == 2:  # Right 
        x += distance * np.cos(np.radians(heading - 90))
        y += distance * np.sin(np.radians(heading - 90))
    elif action == 3:  # Rotate clockwise
        heading = (heading - 45) % 360

    return [x, y, heading]

#--------------------------------------------------------------Q-Learning--------------------------------------------------------------#

# Actions for Tello
actions = ['forward', 'left', 'right', 'cw']
Q = np.zeros((100, len(actions)))

alpha = 0.1
gamma = 0.9
epsilon = 0.1


target_color = "blue"  # Set this to "red", "blue", or "green" via some command
depth_threshold = 700  # Stop rewarding once the depth is below this value
proximity_penalty = -5  # Penalty for moving toward incorrect objects
lost_target_penalty = -1  # Penalty for losing the target

# Parameters for handling variations
missing_frames_tolerance = 1000  # Number of frames to allow without detecting the target color
depth_window_size = 5         # Number of frames for averaging depth

# Queues to track missing frames and depth
missing_frames_counter = 0
depth_history = deque(maxlen=depth_window_size)

# Track cumulative rewards for dynamic exploration (in case Landmark mapping fails)
cumulative_reward_window = deque(maxlen=10)  # Store rewards from the last 10 steps

# Load the Q-table
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


def take_action(state):
    """
    Chooses and executes an action based on the current state.

    Args:
        state (int): Current state index.

    Returns:
        int: The index of the chosen action.
    """
    if random.uniform(0, 1) < epsilon:
        action = random.choice(range(len(actions)))
        print(f"Exploring: Random action chosen: {actions[action]}")
    else:
        action = np.argmax(Q[state])
        print(f"Exploiting: Best action chosen: {actions[action]}")

    # Execute the chosen action (cm)
    response = None
    if action == 0:  # Forward
        response = send_command("forward 50")
    elif action == 1:  # Left
        response = send_command("left 50")
    elif action == 2:  # Right
        response = send_command("right 50")
    elif action == 3:  # Rotate clockwise
        response = send_command("cw 45")

    if response == "ok":
        print(f"Action '{actions[action]}' executed successfully.")
    else:
        print(f"Action '{actions[action]}' failed: {response}")

    # Add a 5-second delay before the next action
    time.sleep(5)
    
    return action


# Q-learning update function
def update_q(state, action, reward, next_state):
    best_next_action = np.argmax(Q[next_state])
    Q[state, action] += alpha * (reward + gamma * Q[next_state, best_next_action] - Q[state, action])

# Flag to indicate if the Q-learning loop has started
loop_started = threading.Event()

def q_learning_loop():
    global epsilon # initialize epsilon within the loop

    try:
        #send_command('takeoff')  # Take off command
        #send_command('up 50') # Move up 50cm to get a view of the area
        time.sleep(1)
        state = 0
        drone_position = [0, 0, 0]  # x, y, heading (initialize at origin)
        missing_frames_counter = 0  # Initialize missing frames

        while True:
            if frame_queue.empty():
                continue

            # Get the current frame
            frame = frame_queue.get()
            detected_objects = prediction_results["detected_objects"]
            # Run YOLO to detect objects
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

            # Update landmarks
            update_landmark_map(detections, depth_map, drone_position)


            if state % 10 == 0:
                save_landmark_map()

            # Detect colored objects
            #colored_objects = detect_colored_objects(frame, detections)

            # Action-based reward
            base_reward = -1  # Penalize small steps to encourage actions
            # Determine reward
            reward = base_reward
            found_landmark = False

            if detected_objects:
                # Find the closest object based on depth
                closest_object = min(detected_objects, key=lambda obj: obj["depth"])
                label = closest_object["label"]
                depth = closest_object["depth"]

                print(f"Target Detected: {label} at Depth: {depth:.2f}")

                # Encourage reducing depth (moving closer to the object)
                if depth < 700:
                    reward += (depth) * 0.1  # Higher reward for moving closer
                    print(f"Encouraging movement closer to {label}, depth: {depth:.2f}")
                else:
                    reward -= 5  # Penalize for moving too far from the object

                # Stop condition: Very close to the object
                if depth > 800:
                    reward += 20  # High reward for reaching the target
                    print(f"Target {label} reached! Depth: {depth:.2f}. Landing...")
                    send_command('land')
                    break

                found_target = True
            else:
                print("No target objects detected.")
                reward -= 5  # Penalize if no objects are found

            if not found_landmark and missing_frames_counter > missing_frames_tolerance:
                reward += -5  # Penalize excessive exploration without results
                print("No landmarks detected. Penalizing for excessive exploration.")

            # Take action and include action-specific rewards
            action = take_action(state)
            if action == 0:  # Forward
                reward += 1  # Reward for moving forward 
            elif action in [1, 2]:  # Left or Right
                reward += 0.5  # Smaller reward for exploring side directions
            elif action == 3:  # Rotate
                reward += 1.5  # Reward for rotations (will encourage finding target)

            # Ensure the action is executed
            print(f"Action chosen: {actions[action]}")
            # Update cumulative rewards
            cumulative_reward_window.append(reward)

            # Use cumulative rewards to infer progress
            if len(cumulative_reward_window) == cumulative_reward_window.maxlen:
                avg_reward = sum(cumulative_reward_window) / len(cumulative_reward_window)
                print(f"Average Reward Trend: {avg_reward:.2f}")

                if avg_reward < 0:
                    print("Low average reward detected. Encouraging exploration.")
                    action = random.choice(range(len(actions)))  # Force exploration

            # Update drone position based on action
            drone_position = update_drone_position(drone_position, action)

            # Transition to the next state
            next_state = (state + 1) % 100

            # Update Q-table
            update_q(state, action, reward, next_state)
            state = next_state
            epsilon = max(0.01, epsilon * 0.995)  # Decay epsilon dynamically

    except Exception as e:
        print(f"An error occurred in the Q-learning loop: {e}")
    finally:
        save_q_table()
        send_command('land')
        print("Main Q-learning loop completed.")


if __name__ == "__main__":
    try:
        # Initialize Tello
        send_command('command')  # Enter SDK mode
        time.sleep(2)
        send_command('streamon')  # Start video streaming
        send_command('battery?')
        time.sleep(2)
        
        # Load resources
        #load_q_table()
        #load_landmark_map()

        # Start threads
        threading.Thread(target=capture_video, daemon=True).start()
        threading.Thread(target=process_predictions, daemon=True).start()
        time.sleep(5)

        # Start the display thread
        #threading.Thread(target=display_video, daemon=True).start()

        # Ensure threads are ready
        time.sleep(5)

        # Takeoff
        #print("Taking off...")
        #send_command('takeoff')

        # Run display and Q-learning loop
        display_video()
        q_learning_loop()

    except KeyboardInterrupt:
        print("Keyboard Interrupt: Landing the drone...")
        send_command('land')  # Safely land
    finally:
        send_command('streamoff')  # Stop video stream
        save_q_table()
        cmd_socket.close()
