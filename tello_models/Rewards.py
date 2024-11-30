class RewardCalculator:
    def __init__(self, target_color="blue", depth_threshold=700, proximity_penalty=-5, lost_target_penalty=-1):
        """
        Initialize the reward calculator.

        Args:
            target_color (str): The target color for the drone.
            depth_threshold (float): Depth threshold for rewards.
            proximity_penalty (float): Penalty for moving toward incorrect objects.
            lost_target_penalty (float): Penalty for losing the target.
        """
        self.target_color = target_color
        self.depth_threshold = depth_threshold
        self.proximity_penalty = proximity_penalty
        self.lost_target_penalty = lost_target_penalty

    def calculate_reward(self, detected_objects, missing_frames_counter, missing_frames_tolerance):
        """
        Calculate the reward based on detected objects and drone state.

        Args:
            detected_objects (list): List of detected objects.
            missing_frames_counter (int): Number of frames without a detected target.
            missing_frames_tolerance (int): Allowed number of missing frames.

        Returns:
            float: Calculated reward.
        """
        base_reward = -1  # Penalize small steps to encourage significant actions
        reward = base_reward

        if detected_objects:
            # Find the closest object based on depth
            closest_object = min(detected_objects, key=lambda obj: obj["depth"])
            label = closest_object["label"]
            depth = closest_object["depth"]

            print(f"Target Detected: {label} at Depth: {depth:.2f}")

            # Reward for reducing depth
            if depth < self.depth_threshold:
                reward += (self.depth_threshold - depth) * 0.01
                print(f"Encouraging movement closer to {label}, depth: {depth:.2f}")
            else:
                reward += self.proximity_penalty

            # Additional reward for reaching very close to the target
            if depth > self.depth_threshold - 50:
                reward += 10
                print(f"Target {label} reached! Depth: {depth:.2f}")

        else:
            reward += self.lost_target_penalty
            #print("No target objects detected.")

            # Penalize excessive exploration
            if missing_frames_counter > missing_frames_tolerance:
                reward += -2
                #print("Excessive exploration detected. Penalizing.")

        return reward
