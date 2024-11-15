from controller import Robot, Keyboard
import math

def clamp(value, low, high):
    return max(min(value, high), low)

# Initialize the robot
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Get and enable devices
camera = robot.getDevice("camera")
camera.enable(timestep)

front_left_led = robot.getDevice("front left led")
front_right_led = robot.getDevice("front right led")

imu = robot.getDevice("inertial unit")
imu.enable(timestep)

gps = robot.getDevice("gps")
gps.enable(timestep)

gyro = robot.getDevice("gyro")
gyro.enable(timestep)

keyboard = Keyboard()
keyboard.enable(timestep)

camera_roll_motor = robot.getDevice("camera roll")
camera_pitch_motor = robot.getDevice("camera pitch")

# Get propeller motors and set them to velocity mode
motors = {
    "front_left": robot.getDevice("front left propeller"),
    "front_right": robot.getDevice("front right propeller"),
    "rear_left": robot.getDevice("rear left propeller"),
    "rear_right": robot.getDevice("rear right propeller"),
}
for motor in motors.values():
    motor.setPosition(float('inf'))
    motor.setVelocity(0.0)

# Constants (empirically adjusted)
k_vertical_thrust = 69.5      # Thrust to lift the drone
k_vertical_offset = 10.0        # Offset for stabilization
k_vertical_p = 1.0             # Reduced vertical PID constant
k_vertical_damping = 0.3       # Damping for smoother altitude control
k_roll_p = 30.0                # Roll PID constant
k_pitch_p = 20.0               # Pitch PID constant
k_yaw_damping = 1.0            # Damping constant for yaw

# Variables
target_altitude = 1.0          # Target altitude
movement_speed = 2.0           # Forward/backward speed control

# Main control loop
while robot.step(timestep) != -1:
    time = robot.getTime()  # Time in seconds

    # Retrieve orientation and altitude data
    roll = imu.getRollPitchYaw()[0]
    pitch = imu.getRollPitchYaw()[1]
    altitude = gps.getValues()[2]
    roll_velocity = gyro.getValues()[0]
    pitch_velocity = gyro.getValues()[1]
    yaw_velocity = gyro.getValues()[2]

    # Blink the front LEDs alternately with a 1-second rate
    led_state = int(time) % 2
    front_left_led.set(led_state)
    front_right_led.set(1 - led_state)

    # Stabilize the camera
    camera_roll_motor.setPosition(-0.115 * roll_velocity)
    camera_pitch_motor.setPosition(-0.1 * pitch_velocity)

    # Initialize disturbances
    roll_disturbance = 0.0
    pitch_disturbance = 0.0
    yaw_disturbance = 0.0

    # Process keyboard input for disturbances and speed/altitude adjustments
    key = keyboard.getKey()
    while key > 0:
        if key == Keyboard.UP:
            pitch_disturbance = -movement_speed  # Control forward speed
        elif key == Keyboard.DOWN:
            pitch_disturbance = movement_speed   # Control backward speed
        elif key == Keyboard.RIGHT:
            yaw_disturbance = -1.3
        elif key == Keyboard.LEFT:
            yaw_disturbance = 1.3
        elif key == (Keyboard.SHIFT + Keyboard.RIGHT):
            roll_disturbance = -1.0
        elif key == (Keyboard.SHIFT + Keyboard.LEFT):
            roll_disturbance = 1.0
        elif key == (Keyboard.SHIFT + Keyboard.UP):
            target_altitude += 0.1  # Increase target altitude
            print(f"Target altitude increased to: {target_altitude:.2f} m")
        elif key == (Keyboard.SHIFT + Keyboard.DOWN):
            target_altitude -= 0.1  # Decrease target altitude
            print(f"Target altitude decreased to: {target_altitude:.2f} m")
        elif key == 32:  # Increase movement speed with space bar
            movement_speed += 0.1
            print(f"Movement speed increased to: {movement_speed:.2f}")
        elif key == 8:  # Decrease movement speed with delete key
            movement_speed = max(0, movement_speed - 0.1)
            print(f"Movement speed decreased to: {movement_speed:.2f}")
        key = keyboard.getKey()

    # Altitude stabilization with target altitude, adding damping for smoother control
    altitude_error = target_altitude - altitude
    vertical_input = k_vertical_p * altitude_error - k_vertical_damping * gps.getValues()[2]

    # Compute the roll, pitch, yaw inputs
    roll_input = k_roll_p * clamp(roll, -1.0, 1.0) + roll_velocity + roll_disturbance
    pitch_input = k_pitch_p * clamp(pitch, -1.0, 1.0) + pitch_velocity + pitch_disturbance
    yaw_input = yaw_disturbance - k_yaw_damping * yaw_velocity  # Damping for yaw

    # Actuate the motors with computed inputs
    front_left_motor_input = k_vertical_thrust + vertical_input - roll_input + pitch_input - yaw_input
    front_right_motor_input = k_vertical_thrust + vertical_input + roll_input + pitch_input + yaw_input
    rear_left_motor_input = k_vertical_thrust + vertical_input - roll_input - pitch_input + yaw_input
    rear_right_motor_input = k_vertical_thrust + vertical_input + roll_input - pitch_input - yaw_input

    motors["front_left"].setVelocity(front_left_motor_input)
    motors["front_right"].setVelocity(-front_right_motor_input)
    motors["rear_left"].setVelocity(-rear_left_motor_input)
    motors["rear_right"].setVelocity(rear_right_motor_input)

    # Display current altitude and speed in the console (output every 0.5 seconds)
    if int(time * 2) % 2 == 0:  # Limits the print frequency to improve readability
        print(f"Current altitude: {altitude:.2f} m, Target altitude: {target_altitude:.2f} m")
        print(f"Current movement speed: {movement_speed:.2f}")
