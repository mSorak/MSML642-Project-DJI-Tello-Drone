import numpy as np
import cv2
import socket
import time
import threading
import queue
import random
from datetime import datetime
import queue

# Tello IP and Ports
TELLO_IP = '192.168.10.1'
CMD_PORT = 8889
STATE_PORT = 8890
VIDEO_PORT = 11111
TELLO_ADDRESS = (TELLO_IP, CMD_PORT)

# Set up UDP socket for sending commands
cmd_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
cmd_socket.bind(('', CMD_PORT))

# Set up UDP socket for receiving state information
state_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
state_socket.bind(('', STATE_PORT))

# Queue for frames to be displayed
frame_queue = queue.Queue()

# Function to send commands to Tello and log them
def send_command(command):
    cmd_socket.sendto(command.encode('utf-8'), TELLO_ADDRESS)
    response, _ = cmd_socket.recvfrom(1024)
    return response.decode('utf-8', errors='ignore')

# Function to log Tello's state data
def receive_state():
    with open("tello_state_log.txt", "a") as file:
        while True:
            state_data, _ = state_socket.recvfrom(1024)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            state_str = state_data.decode('utf-8').replace(';', ';\n')
            log_entry = f"{timestamp} - Tello State:\n{state_str}\n"
            file.write(log_entry)
            time.sleep(0.2)
            
# Function to capture video frames and store them in a queue
def capture_video():
    video_url = f'udp://{TELLO_IP}:{VIDEO_PORT}'
    cap = cv2.VideoCapture(video_url)

    output_filename = 'tello_stream.mp4'
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 24
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    while True:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            frame_queue.put(frame)  # Put frame in queue for display
        else:
            print("Failed to retrieve frame.")
            break

    cap.release()
    out.release()

def display_video():
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            if frame is not None:
                cv2.imshow("Tello Video Stream", frame)
            else:
                print("Warning: Received None frame from video stream.")
            
            # Check for 'q' key to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                send_command('land')  # Land when exiting display
                break
    cv2.destroyAllWindows()

# Save the Q-table
def save_q_table(filename='q_table.npy'):
    np.save(filename, Q)
    print("Q-table saved to", filename)


# Function to handle user input for commands in real-time
def command_input():
    while True:
        user_command = input("Enter Tello command (or 'quit' to exit): ")
        if user_command.lower() == 'quit':
            send_command('streamoff')
            break
        else:
            send_command(user_command)


frame_queue = queue.Queue()
actions = ['forward', 'left', 'right', 'cw']  # 4 discrete actions
Q = np.zeros((100, len(actions)))  # Example Q-table for 100 states, 4 actions

# Parameters for Q-learning
alpha = 0.1       # Learning rate
gamma = 0.9       # Discount factor
epsilon = 0.1     # Epsilon-greedy parameter

# Load the Q-table
def load_q_table(filename='q_table.npy'):
    try:
        global Q
        Q = np.load(filename)
        print("Q-table loaded from", filename)
    except FileNotFoundError:
        print("No saved Q-table found. Starting with a fresh Q-table.")
        Q = np.zeros((100, len(actions)))  # Initialize with zeros if no file found


# Detect the red object in the frame and return its position and size
def detect_red_object(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv_frame, lower_red, upper_red)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(c)
        area = w * h  # Calculate bounding box area
        return (x + w // 2, y + h // 2), area  # Return position and area
    return None, 0  # No red object detected

def take_action(state):
    if random.uniform(0, 1) < epsilon:
        action = random.choice(range(len(actions)))
    else:
        action = np.argmax(Q[state])

    # Execute the action and print it
    if action == 0:  # forward
        send_command("forward 12.5")
        print("Action chosen: forward 12.5 cm")
    elif action == 1:  # left
        send_command("left 12.5")
        print("Action chosen: left 12.5 cm")
    elif action == 2:  # right
        send_command("right 12.5")
        print("Action chosen: right 12.5 cm")
    elif action == 3:  # cw (rotate)
        send_command("cw 45")
        print("Action chosen: rotate clockwise 45 degrees")

    return action

# Q-learning update function
def update_q(state, action, reward, next_state):
    best_next_action = np.argmax(Q[next_state])
    Q[state, action] += alpha * (reward + gamma * Q[next_state, best_next_action] - Q[state, action])

# Flag to indicate if the Q-learning loop has started
loop_started = threading.Event()

# Q-learning loop without OpenCV display
def q_learning_loop(proximity_threshold=5000):
    try:
        send_command('takeoff')  # Take off command
        time.sleep(1)
        state = 0
        while True:
  
            if frame_queue.empty():
                continue
            
            frame = frame_queue.get()
            red_object_pos, red_object_area = detect_red_object(frame)

            if red_object_pos is not None:
                reward = 10
                if red_object_area > proximity_threshold:
                    print("Red object reached. Landing.")
                    send_command('land')
                    break
            else:
                reward = -1  # No red object in view

            action = take_action(state)
            next_state = (state + 1) % 100 
            update_q(state, action, reward, next_state)
            state = next_state

    except Exception as e:
        print(f"An error occurred in the Q-learning loop: {e}")
    finally:
        save_q_table()
        print("Main Q-learning loop completed.")


if __name__ == "__main__":
    load_q_table()
    send_command('command')  # Enter SDK mode
    time.sleep(1)
    send_command('streamon')  # Turn on video stream
    time.sleep(1)

    # Start threads for state logging and video capture
    threading.Thread(target=receive_state, daemon=True).start()
    threading.Thread(target=capture_video, daemon=True).start()
    threading.Thread(target=q_learning_loop, daemon=True).start()  # Run q_learning in a separate thread

    # Display video in main thread
    display_video()

    # Final cleanup
    save_q_table()
    send_command('streamoff')  # Turn off video stream
    cmd_socket.close()
    state_socket.close()