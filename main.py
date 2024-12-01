

import cv2
import torch
import numpy as np
import time
import threading
import queue
from datetime import datetime
import random
from collections import deque
import socket 
from controller.DroneNavigation import DroneNavigation
from tello_models.Q import QLearningAgent
from tello_models.Rewards import RewardCalculator
from predictions.PredictionProcessor import PredictionProcessor
from video.Video import VideoProcessor

cv2.namedWindow("Tello Video Stream", cv2.WINDOW_NORMAL)

# Initialize devices
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps") 

midas_device = torch.device("mps") if torch.backends.mps.is_available() else device

# Initialize models
yolo_model = torch.hub.load('/Users/josephsketl/yolov5', 'yolov5s', source='local', pretrained=True)
yolo_model.conf = 0.2  # Display objects with confidence > 0.4
midas_model = torch.hub.load("/Users/josephsketl/MiDaS/", "MiDaS_small", source="local")
midas_transforms = torch.hub.load("/Users/josephsketl/MiDaS/", "transforms", source="local").small_transform

# Tello IP and Ports
TELLO_IP = '192.168.10.1'
CMD_PORT = 8889
VIDEO_PORT = 11111
STATE_PORT = 8890
TELLO_ADDRESS = (TELLO_IP, CMD_PORT)


# Set up UDP socket for receiving state information
state_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
state_socket.bind(('', STATE_PORT))

frame_queue = queue.Queue(maxsize=20)  # Limit queue size for memory efficiency

# Set target labels
target_labels = ["vase", "apple", "sports ball", 'person']

# Initialize modules
prediction_processor = PredictionProcessor(
    yolo_model=yolo_model,
    midas_model=midas_model,
    midas_transforms=midas_transforms,
    target_labels=target_labels,
    device=device,
    midas_device=midas_device,
)

#video_processor = VideoProcessor(tello_ip="192.168.10.1", video_port=11111, prediction_processor=prediction_processor)
q_agent = QLearningAgent(actions=["forward", "left", "right", "cw"])
reward_calculator = RewardCalculator(target_color="blue")
drone = DroneNavigation(tello_ip="192.168.10.1", cmd_port=8889)

# Start threads
def start_video_threads():
    capture_thread = threading.Thread(target=video_processor.capture_video, daemon=True)
    capture_thread.start()
    return capture_thread

# Start threads
def start_prediction_thread(prediction_processor, frame_queue):
    prediction_thread = threading.Thread(
        target=prediction_processor.process_predictions,
        args=(frame_queue,),
        daemon=True  # Ensures thread exits when the main program ends
    )
    prediction_thread.start()
    return prediction_thread


# Main loop
def main_loop():
    try:
        time.sleep(2)

        state = 0
        missing_frames_counter = 0

        while True:
            # Get the latest prediction results
            prediction_results = prediction_processor.get_results()
            detected_objects = prediction_results["detected_objects"]

            if detected_objects:
                missing_frames_counter = 0  # Reset counter if objects are detected
            else:
                missing_frames_counter += 1

            # Calculate reward
            reward = reward_calculator.calculate_reward(
                detected_objects,
                missing_frames_counter,
                missing_frames_tolerance=1000
            )

            # Choose and execute action
            action_index = q_agent.choose_action(state)
            action_name = q_agent.actions[action_index]
            #drone.execute_action(action_name)

            # Update Q-table
            next_state = (state + 1) % 100
            q_agent.update_q_table(state, action_index, reward, next_state)

            time.sleep(3)
            # Transition to the next state
            state = next_state

    except KeyboardInterrupt:
        print("Landing the drone...")
        drone.send_command("land")
    finally:
        q_agent.save_q_table()
        video_processor.stop()
        drone.send_command("land")
        drone.cmd_socket.close()


if __name__ == "__main__":
    drone.send_command("command")
    drone.send_command("streamon")

    # Initialize VideoProcessor
    video_processor = VideoProcessor(
        tello_ip="192.168.10.1",
        video_port=11111,
        prediction_processor=prediction_processor,
    )

    # Start threads
    capture_thread = threading.Thread(target=video_processor.capture_video, daemon=True)
    process_thread = threading.Thread(target=video_processor.process_frames, daemon=True)
    capture_thread.start()
    process_thread.start()
    main_loop_thread = threading.Thread(target=main_loop, daemon=True)
    main_loop_thread.start()

    # Run the PyGame-based display in the main thread
    try:
        video_processor.display_video_with_pygame()

    except KeyboardInterrupt:
        video_processor.stop()
        capture_thread.join()
        process_thread.join()

