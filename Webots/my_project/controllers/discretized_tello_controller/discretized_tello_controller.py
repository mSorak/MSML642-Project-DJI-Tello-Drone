from controller import Robot, Keyboard
import numpy as np

def clamp(value, value_min, value_max):
    return min(max(value, value_min), value_max)

class Mavic(Robot):
    # Constants for stabilization and control
    K_VERTICAL_THRUST = 68.5    # Thrust to lift the drone
    K_VERTICAL_OFFSET = 0.6     # Vertical offset for altitude hold
    K_VERTICAL_P = 3.0          # Altitude PID constant
    K_ROLL_P = 50.0             # Roll PID constant
    K_PITCH_P = 30.0            # Pitch PID constant
    K_YAW_DAMPING = 0.3         # Damping for yaw stability

    target_precision = 0.02     # Tight precision threshold for reaching a target

    def __init__(self):
        super().__init__()
        self.time_step = int(self.getBasicTimeStep())

        # Initialize devices
        self.camera = self.getDevice("camera")
        self.camera.enable(self.time_step)
        self.imu = self.getDevice("inertial unit")
        self.imu.enable(self.time_step)
        self.gps = self.getDevice("gps")
        self.gps.enable(self.time_step)
        self.gyro = self.getDevice("gyro")
        self.gyro.enable(self.time_step)
        self.keyboard = Keyboard()
        self.keyboard.enable(self.time_step)

        # Initialize motors
        self.front_left_motor = self.getDevice("front left propeller")
        self.front_right_motor = self.getDevice("front right propeller")
        self.rear_left_motor = self.getDevice("rear left propeller")
        self.rear_right_motor = self.getDevice("rear right propeller")
        self.motors = [self.front_left_motor, self.front_right_motor,
                       self.rear_left_motor, self.rear_right_motor]
        for motor in self.motors:
            motor.setPosition(float('inf'))
            motor.setVelocity(1.0)

        # Initialize position and target tracking
        self.current_position = [0, 0, 0]   # x, y, altitude
        self.target_position = [0, 0, 0]
        self.target_altitude = 1.0          # Default target altitude
        self.target_reached = True          # Indicates if the target has been reached

    def update_position(self):
        """Update the current position based on GPS."""
        x_pos, y_pos, altitude = self.gps.getValues()
        self.current_position = [x_pos, y_pos, altitude]

    def set_new_waypoint(self, direction):
        """Set a new waypoint 1 meter in the specified direction."""
        x, y, z = self.current_position
        if direction == "UP":  # Move forward in y-axis
            self.target_position = [x, y + 1.0, z]
        elif direction == "DOWN":  # Move backward in y-axis
            self.target_position = [x, y - 1.0, z]
        elif direction == "RIGHT":  # Move right in x-axis
            self.target_position = [x + 1.0, y, z]
        elif direction == "LEFT":  # Move left in x-axis
            self.target_position = [x - 1.0, y, z]
        self.target_reached = False  # New target set, so not yet reached

    def move_to_target(self):
        """Calculate disturbances to move towards the target position."""
        x, y, z = self.current_position
        target_x, target_y, _ = self.target_position

        # Calculate distance to target
        distance_to_target = np.sqrt((target_x - x) ** 2 + (target_y - y) ** 2)

        if distance_to_target < self.target_precision:
            self.target_reached = True
            return 0.0, 0.0  # No pitch or yaw disturbance needed

        # Calculate the yaw disturbance based on target angle
        target_angle = np.arctan2(target_y - y, target_x - x)
        current_yaw = self.imu.getRollPitchYaw()[2]
        yaw_error = target_angle - current_yaw

        # Normalize yaw error to range [-pi, pi]
        yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi
        yaw_disturbance = clamp(yaw_error * 0.5, -0.3, 0.3)  # Yaw control with damping

        # Calculate forward pitch disturbance based on distance to target
        pitch_disturbance = clamp(1.0 * distance_to_target, -1.0, 1.0)  # Forward pitch control

        return yaw_disturbance, -pitch_disturbance

    def control_altitude(self, altitude):
        """Control the altitude to reach the target altitude."""
        altitude_error = self.target_altitude - altitude
        clamped_difference_altitude = clamp(altitude_error + self.K_VERTICAL_OFFSET, -1.0, 1.0)
        return self.K_VERTICAL_P * pow(clamped_difference_altitude, 3.0)

    def run(self):
        while self.step(self.time_step) != -1:
            self.update_position()
            x, y, altitude = self.current_position

            # Process keyboard input for setting waypoints
            key = self.keyboard.getKey()
            if key == Keyboard.UP:
                self.set_new_waypoint("UP")
            elif key == Keyboard.DOWN:
                self.set_new_waypoint("DOWN")
            elif key == Keyboard.RIGHT:
                self.set_new_waypoint("RIGHT")
            elif key == Keyboard.LEFT:
                self.set_new_waypoint("LEFT")

            # Calculate control inputs
            roll, pitch, yaw = self.imu.getRollPitchYaw()
            roll_velocity, pitch_velocity, yaw_velocity = self.gyro.getValues()
            yaw_disturbance, pitch_disturbance = (0.0, 0.0)

            if not self.target_reached:
                # Move toward the target
                yaw_disturbance, pitch_disturbance = self.move_to_target()
            else:
                # Stabilize and hold position at the target
                yaw_disturbance = -self.K_YAW_DAMPING * yaw_velocity  # Yaw stabilization
                pitch_disturbance = 0.0  # No forward/backward movement
                roll_disturbance = 0.0  # Zero roll to hold position

            # Stabilize altitude
            vertical_input = self.control_altitude(altitude)
            roll_input = self.K_ROLL_P * clamp(roll, -1, 1) + roll_velocity
            pitch_input = self.K_PITCH_P * pitch_disturbance + pitch_velocity
            yaw_input = yaw_disturbance

            # Actuate the motors based on control inputs
            front_left_motor_input = self.K_VERTICAL_THRUST + vertical_input - yaw_input + pitch_input - roll_input
            front_right_motor_input = self.K_VERTICAL_THRUST + vertical_input + yaw_input + pitch_input + roll_input
            rear_left_motor_input = self.K_VERTICAL_THRUST + vertical_input + yaw_input - pitch_input - roll_input
            rear_right_motor_input = self.K_VERTICAL_THRUST + vertical_input - yaw_input - pitch_input + roll_input

            self.front_left_motor.setVelocity(front_left_motor_input)
            self.front_right_motor.setVelocity(-front_right_motor_input)
            self.rear_left_motor.setVelocity(-rear_left_motor_input)
            self.rear_right_motor.setVelocity(rear_right_motor_input)

            # Display current position and target status
            print(f"Current position: ({x:.2f}, {y:.2f}), Target position: ({self.target_position[0]:.2f}, {self.target_position[1]:.2f})")
            if self.target_reached:
                print("Target reached and holding position.")

# Initialize and run the robot
robot = Mavic()
robot.run()
