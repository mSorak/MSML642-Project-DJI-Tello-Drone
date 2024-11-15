import socket
import cv2
import time
import threading
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
def send_command(command: str):
    cmd_socket.sendto(command.encode('utf-8'), TELLO_ADDRESS)
    response, _ = cmd_socket.recvfrom(1024)

    # Log the command and the response
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp} - Command sent: {command}\n"
    log_entry += f"{timestamp} - Tello response: {response.decode('utf-8', errors='ignore')}\n"
    print(log_entry.strip())
    
    # Append log entry to the log file
    with open("tello_state_log.txt", "a") as file:
        file.write(log_entry)

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

# Main thread function to display frames
def display_video():
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            cv2.imshow("Tello Video Stream", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()

# Function to handle user input for commands in real-time
def command_input():
    while True:
        user_command = input("Enter Tello command (or 'quit' to exit): ")
        if user_command.lower() == 'quit':
            send_command('streamoff')
            break
        else:
            send_command(user_command)

# Main script execution
if __name__ == "__main__":
    # Initialize Tello and start video stream
    send_command('command')  # Enter SDK mode
    time.sleep(1)
    send_command('streamon')  # Turn on video stream
    time.sleep(1)

    # Start state logging in a separate thread
    state_thread = threading.Thread(target=receive_state, daemon=True)
    state_thread.start()

    # Start video capture in a separate thread
    video_thread = threading.Thread(target=capture_video, daemon=True)
    video_thread.start()

    # Start command input in a separate thread
    command_thread = threading.Thread(target=command_input, daemon=True)
    command_thread.start()

    # Display video in the main thread
    display_video()

    # Clean up after quitting
    cmd_socket.close()
    state_socket.close()
