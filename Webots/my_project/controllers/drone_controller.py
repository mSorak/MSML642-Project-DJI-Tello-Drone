from controller import Robot, Keyboard

TIME_STEP = 64  # Time step for simulation

class WebotsDroneController:
    def __init__(self):
        # Initialize the Webots robot controller
        self.robot = Robot()
        self.keyboard = Keyboard()
        self.keyboard.enable(TIME_STEP)

        # Update motor names based on actual names in the Scene Tree
        self.front_left_motor = self.robot.getDevice("front left propeller")  # Update to correct name
        self.front_right_motor = self.robot.getDevice("front right propeller") # Update to correct name
        self.rear_left_motor = self.robot.getDevice("rear left propeller")   # Update to correct name
        self.rear_right_motor = self.robot.getDevice("rear right propeller")  # Update to correct name

        # Set motors to velocity control mode by setting position to infinity
        self.front_left_motor.setPosition(float('inf'))
        self.front_right_motor.setPosition(float('inf'))
        self.rear_left_motor.setPosition(float('inf'))
        self.rear_right_motor.setPosition(float('inf'))

        # Set initial velocity to 0 for all motors
        self.set_motor_velocity(0)

        # Initial drone state
        self.is_flying = False

    def set_motor_velocity(self, velocity):
        # Set the motor velocity for all propellers
        self.front_left_motor.setVelocity(velocity)
        self.front_right_motor.setVelocity(velocity)  # Use positive values to avoid errors
        self.rear_left_motor.setVelocity(velocity)
        self.rear_right_motor.setVelocity(velocity)

    def takeoff(self):
        if not self.is_flying:
            self.set_motor_velocity(6.28)  # Start propellers
            self.is_flying = True
            print("Drone taking off...")

    def land(self):
        if self.is_flying:
            self.set_motor_velocity(0)  # Stop propellers
            self.is_flying = False
            print("Drone landing...")

    def move_up(self, duration=1.0):
        if self.is_flying:
            self.set_motor_velocity(7)  # Increase speed
            self.robot.step(int(duration * 1000))  # Move for specified duration
            self.set_motor_velocity(6.28)  # Restore default flying speed

    def move_down(self, duration=1.0):
        if self.is_flying:
            self.set_motor_velocity(5.5)  # Decrease speed to descend
            self.robot.step(int(duration * 1000))
            self.set_motor_velocity(6.28)

    def rotate_cw(self, duration=1.0):
        if self.is_flying:
            # Set different speeds to rotate clockwise
            self.front_left_motor.setVelocity(6.5)
            self.front_right_motor.setVelocity(6.0)
            self.rear_left_motor.setVelocity(6.5)
            self.rear_right_motor.setVelocity(6.0)
            self.robot.step(int(duration * 1000))
            self.set_motor_velocity(6.28)  # Restore normal hover

    def rotate_ccw(self, duration=1.0):
        if self.is_flying:
            # Set different speeds to rotate counterclockwise
            self.front_left_motor.setVelocity(6.0)
            self.front_right_motor.setVelocity(6.5)
            self.rear_left_motor.setVelocity(6.0)
            self.rear_right_motor.setVelocity(6.5)
            self.robot.step(int(duration * 1000))
            self.set_motor_velocity(6.28)

    def control_loop(self):
        # Main control loop
        while self.robot.step(TIME_STEP) != -1:
            key = self.keyboard.getKey()
            if key == ord('T'):  # 'T' for takeoff
                self.takeoff()
            elif key == ord('L'):  # 'L' for landing
                self.land()
            elif key == ord('W'):  # 'W' for up
                self.move_up()
            elif key == ord('S'):  # 'S' for down
                self.move_down()
            elif key == ord('A'):  # 'A' for rotate counterclockwise
                self.rotate_ccw()
            elif key == ord('D'):  # 'D' for rotate clockwise
                self.rotate_cw()

# Initialize and run the Webots drone controller
if __name__ == "__main__":
    drone_controller = WebotsDroneController()
    drone_controller.control_loop()
