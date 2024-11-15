NOTE: You will need to make sure that WorldInfo -> defaultDamping is not set to 'NULL' and add damping of: linear 0.5, angular 0.5 to prevent the drone from oscillating and flying crazy.

The tello_controller.py's main controls:

- UP: Moves the drone fowrward
- DOWN: Moves backward
- LEFT: Rotates the drone
- RIGHT: Rotates the drone in opposite direction
- SHIFT + LEFT: Drone moves to the left
- SHIFT + RIGHT: Drone moves to the right
- SHIFT + UP: Increases altitude
- SHIFT + DOWN: Decreases altitude
- SPACEBAR: Increases speed (sort of works, highly recommend you don't use it because it is pretty unstable)
- BACKSPACE: Decreases speed (doesn't work as far as I can tell)


These controls essentially mimic the movements of Tello and if we can discretize these movements (e.g. UP moves the drone forward 1m.. etc), then we should have a working simulation model of Tello. We can then build upon this to incorporate our reinforcement learning algorithm. I basically repurposed the mavic2pro.c file with a few adjustments to create this controller.


Discretized_tello_controller.py:

Doesn't really work. Functions on x,y,z coordinates. Have not figured out a way to make the drone stop when it reaches a target position. Highly unstable and will most likely crash before you get it to function. I tried to draw inspiration from mavic2pro_patrol to set waypoints the drone can fly to, except instead of a list of waypoints it flies to and around, the keyboard commands will set a waypoint exactly 1m in any direction (thus, creating discretized actions within the space). Struggling to get it to stop, hover, or anything besides fly off in the specified direction. Trying to combine the tello_controller.py controls with a set position/waypoint method, but am not successful so far. Trying to also develop a command to cut power to the motors to stop it from moving any further in a direction.. unsuccessful so far.
