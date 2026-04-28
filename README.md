# Emotion-ForestRover
Code for an emotion modulated algorithm for robot recovery from entrapment areas in forest environments

This repository shows some of the code from my third year university project that formed my dissertation. The python script emotion_robot.py is the main script that ran on the rover, however the code specific to the hardware of the rover that was used in the project, and the controller that was used to start and stop the rover, and take over driving using the joystick during testing has been removed from the script, as this code was not written by me. This does mean, however that the algorithm itself to be seen more clearly, and that code could be added in to allow this algorithm to work on a different robot, so long as it has an IMU sensor and a camera.

The algorithm is made up of several parts, including:
- Interpreting sensor data
- Interpreting camera data
- Detecting abnormalities based on the interpreted data
- Behaviours for the rover to perform
- The emotion model, which modulates the behaviours

## To do:
Add back in the code producing the data log file
Upload the script that creates graphs from the data log
