# Code for low level behaviours and emotion model for robot recovery from entrapment areas in forest environments
#
# Code for the specific hardware used in the project has been removed

import time
import os
import numpy as np
import math
import cv2 as cv
from threading import Thread
from threading import RLock
from random import randint
from random import random

bearings = [0]
pitches = [0,0,0]
rolls = [0,0,0]
acc_xs = [0]
acc_ys = [0.1,0.1,0.1]
acc_zs = [0]
top_bottom_brightnesses = [26,26,26]
left_right_brightnesses = [0]
similarities = [40]
collision_rate = [0]

COLLISION_THRESHOLD = 10
GOAL_THRESHOLD = 0.98
SIMILARITY_THRESHOLD = 40 # Found
PITCH_VARY_THRESHOLD = 4 # Find this
ROLL_VARY_THRESHOLD = 4
MAX_TURN_SPEED = 0.3
MIN_BEHAVIOUR_LENGTH = 24
MAX_BEHAVIOUR_LENGTH = 48
DEFAULT_PITCH_THRESHOLD = 30
MAX_PITCH_THRESHOLD = 45
MIN_PITCH_THRESHOLD = 10
MAX_PROBE_ANGLE = 80
MIN_PROBE_ANGLE = 20
MIN_BACK_OUT_DISTANCE = 6
MAX_BACK_OUT_DISTANCE = 12
MIN_SPEED = 0.45
DEFAULT_SPEED = 0.9
WIDE_TURN_AMOUNT = 0.5
IMAGE_RATE = 6
DEFAULT_BACK_OUT_DISTANCE = 9

config_params = {}
config_params["anger"] = False
config_params["boredom"] = False
config_params["fear"] = False
config_params["happiness"] = False
config_params["goal_bearing"] = 0

parameters = {}
parameters["pitch_threshold"] = 30
parameters["roll_threshold"] = 20
parameters["speed"] = 0.9
parameters["turn_speed"] = 1
parameters["turn_amount"] = 1
parameters["direction"] = 0
parameters["behaviour_length"] = 36
parameters["default_behaviour_length"] = 36
parameters["bearing_turning_to"] = 0
parameters["bearing_turned_from"] = 0
parameters["collision"] = 0
parameters["turning"] = 0
parameters["probe_angle"] = 30
parameters["back_out_distance"] = 9
parameters["reversed_by"] = 0
parameters["roll_on_obstacle"] = 0
parameters["top_bottom_brightness_threshold"] = 25
parameters["previous_image"] = np.empty((120,160,3),dtype=np.float64)

parameters["data_index"] = 0
parameters["time_since_collision"] = 0
parameters["collision_count"] = 0
parameters["collision_rate"] = 0
parameters["iter_index"] = 0
parameters["trapped"] = False
parameters["stuck"] = False
parameters["current_behaviour"] = 0
parameters["previous_behaviour"] = 0
parameters["previous_trapped_behaviour"] = 0
parameters["current_trapped_behaviour"] = 0

parameters["time_trapped"] = 0
parameters["time_free"] = 0
parameters["time_doing_behaviour"] = 0
parameters["time_since_behaviour_started"] = 0

parameters["suggested_behaviour"] = -1
parameters["extra_probe_angle"] = 0
parameters["probe_direction"] = 0
parameters["how_long_dark"] = 0
parameters["reacted"] = IMAGE_RATE
parameters["side"] = 0
parameters["probe_direction_time"] = 0
parameters["previous_probe_length"] = 0
parameters["obstacle_type"] = 0
parameters["time_similar"] = 0

emotions = {}
emotions["anger"] = 0
emotions["boredom"] = 0
emotions["fear"] = 0
emotions["happiness"] = 0

turning = {}
turning["Kp"] = MAX_TURN_SPEED/180
turning["Ki"] = (MAX_TURN_SPEED/180) * 0.05 * (17/3)
turning["Kd"] = ((MAX_TURN_SPEED/180) / 0.05) * 0.1
turning["previous_error"] = 0
turning["integral"] = 0

process_loop = True
moving = False

data_index_lock = RLock()
similarities_lock = RLock()
vertical_lock = RLock()
horizontal_lock = RLock()

parent_dir = "" # Enter directory path here
# Read in config info
config_file = os.path.join(parent_dir, "config.txt")
config = open(config_file, 'r', buffering=1)
config_params["goal_bearing"] = int(config.readline())

# Parameters to turn certain emotions on and off for testing
config_params["anger"] = bool(int(config.readline()))
config_params["boredom"] = bool(int(config.readline()))
config_params["fear"] = bool(int(config.readline()))
config_params["happiness"] = bool(int(config.readline()))

data_dir = str(time.time())
path = os.path.join(parent_dir, data_dir)

interval = 0.05

# Insert an element into a list up to a specific size
def insertUpToN(item, ls, n):
    if len(ls) >= n:
        ls.pop(n-1)
    ls.insert(0, item)
    return ls

def softmax(arr):
    return np.exp(arr)/sum(np.exp(arr))

# Returns a specified section of an image
def splitImage(image, where):
    if where == "all":
        return image
    elif where == "top":
        return np.split(image, 2, axis=0)[0]
    elif where == "bottom":
        return np.split(image, 2, axis=0)[1]
    elif where == "left":
        return np.split(image, 2, axis=1)[0]
    elif where == "right":
        return np.split(image, 2, axis=1)[1]
    elif where == "topleft":
        return np.split(np.split(image, 2, axis=0)[0], 2, axis=0)[0]
    elif where == "topright":
        return np.split(np.split(image, 2, axis=0)[0], 2, axis=0)[1]
    elif where == "bottomleft":
        return np.split(np.split(image, 2, axis=0)[1], 2, axis=0)[0]
    elif where == "bottomright":
        return np.split(np.split(image, 2, axis=0)[1], 2, axis=0)[1]
    elif where == "topquarter":
        return np.split(image, 4, axis=0)[0]
    elif where == "bottomthreequarters":
        return np.concatenate((np.split(image, 4, axis=0)[1],np.split(image, 4, axis=0)[2],np.split(image, 4, axis=0)[3]))

# Returns the brightness of a specific area of an image
def getImageBrightness(image, where):
    img = splitImage(image[:,:,2], where)
    brightness = np.mean(img,dtype=np.float64)
    return brightness

# Given the index of an image and whether it is in colour or greyscale, read the image and return it
def getImageFromIndex(index, colour):
    filename = path + "/Image" + str(index) + ".jpg"
    if colour:
        img = cv.imread(filename, cv.IMREAD_COLOR)
    else:
        img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
    return img

# Normalizes the saturation and value fields of a HSV image converted from the input BGR image
def normalizeHSV(image):
    hsv_img = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    hues = hsv_img[:,:,0]
    saturations = np.zeros(hues.shape)
    values = np.zeros(hues.shape)
    cv.normalize(hsv_img[:,:,1].astype(np.float64), saturations, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    cv.normalize(hsv_img[:,:,2].astype(np.float64), values, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    return np.swapaxes(np.swapaxes(np.array([hues.astype(np.float64), saturations, values]),0,1),1,2)

# Returns how similar an image is to the previous image
def imageSimilarity(img):
    for i in range(0, 2):
        img = cv.pyrDown(img)
    previous_image = parameters["previous_image"]
    parameters["previous_image"] = img
    return math.sqrt(np.mean(np.square(previous_image - img, dtype=np.float64),dtype=np.float64))

# Get all the useful data that can be computed about an image
def getImageData(image_num):
    image_to_process = getImageFromIndex(image_num, True)
    img_norm1 = normalizeHSV(image_to_process)
    if parameters["data_index"] >= 1:
        similarity_value = imageSimilarity(img_norm1)
        with similarities_lock:
            similarities.insert(0, similarity_value)
    else:
        with similarities_lock:
            similarities.insert(0,17)

    top_bottom_brightness = getImageBrightness(img_norm1, "topquarter") - getImageBrightness(img_norm1, "bottomthreequarters")
    with vertical_lock:
        top_bottom_brightnesses.insert(0, top_bottom_brightness)
    left_right_brightness = getImageBrightness(img_norm1, "right") - getImageBrightness(img_norm1, "left")
    with horizontal_lock:
        left_right_brightnesses.insert(0, left_right_brightness)

# Loop in the image processing thread
# Repeatedly captures and processes images
def imageDataLoop():
    while process_loop:
        if moving: # Only capture images when rover is moving
            captureImage(path + '/Image' + str(parameters["data_index"]) + '.jpg')
            getImageData(parameters["data_index"])
            with data_index_lock:
                parameters["data_index"] += 1

# Returns for how many images in a row it has been dark in the top part of the image
# Can be used to set the probing angle, as it can be a sense of how large an obstacle is
def howLongDarkBeforeCollision(): 
    count = 0
    with vertical_lock:
        if len(top_bottom_brightnesses) > 1:
            current_brightness = top_bottom_brightnesses[0]
            while current_brightness < parameters["top_bottom_brightness_threshold"]:
                count += 1
                current_brightness = top_bottom_brightnesses[count]
    return count

# Return for how many images in a row the images have been similar
def howLongSimilar():
    count = 0
    with similarities_lock:
        if len(similarities) > 1:
            current_similarity = similarities[0]
            while current_similarity < SIMILARITY_THRESHOLD:
                count += 1
                current_similarity = similarities[count]
    return count

# Compute the current emotion state
def getEmotionState():
    parameters["collision_rate"] = np.sum(collision_rate)
    oldAnger = emotions["anger"]
    oldBoredom = emotions["boredom"]
    oldFear = emotions["fear"]
    oldHappiness = emotions["happiness"]

    # Anger
    emotions["anger"] += (parameters["time_trapped"]/(MAX_BEHAVIOUR_LENGTH*2)- parameters["time_free"]/(MAX_BEHAVIOUR_LENGTH/10) - oldFear/100)
    if emotions["anger"] > 100:
        emotions["anger"] = 100
    elif emotions["anger"] < 0:
        emotions["anger"] = 0
    
    # Boredom
    emotions["boredom"] += (parameters["time_doing_behaviour"]/(MAX_BEHAVIOUR_LENGTH*2) - 1 - (oldAnger + oldFear + oldHappiness)/300)
    if emotions["boredom"] > 100:
        emotions["boredom"] = 100
    elif emotions["boredom"] < 0:
        emotions["boredom"] = 0

    # Fear
    emotions["fear"] += ((parameters["collision_rate"] * np.mean(pitches[0:3])/45 - parameters["time_since_collision"]/(MAX_BEHAVIOUR_LENGTH/20)))
    if emotions["fear"] > 100:
        emotions["fear"] = 100
    elif emotions["fear"] < 0:
        emotions["fear"] = 0

    # Happiness
    emotions["happiness"] += (parameters["time_free"]/(MAX_BEHAVIOUR_LENGTH*2) + parameters["time_since_collision"]/(MAX_BEHAVIOUR_LENGTH) - parameters["time_trapped"]/(MAX_BEHAVIOUR_LENGTH/10) - (parameters["collision_rate"] * np.mean(pitches[0:3])/45) - (oldFear + oldAnger)/200)
    if emotions["happiness"] > 100:
        emotions["happiness"] = 100
    elif emotions["happiness"] < 0:
        emotions["happiness"] = 0

    return np.array([emotions["anger"], emotions["boredom"], emotions["fear"], emotions["happiness"]])

# Return if the rover is currently stopped
def stopped():
    with similarities_lock:
        similarity = similarities[0]
    return similarity < SIMILARITY_THRESHOLD and math.cos(np.ptp(bearings[0:3])*math.pi/180) > 0.9999 and -0.1 < acc_ys[0] < 0.1

# Abnormality detection
def detectAbnormality(parameters): #Return 0 if no collision, 1 if collided forwards, 2 if collided backwards, 3 if stuck but no collision, 4 if stuck on obstacle
    parameters["reacted"] += 1

    parameters['time_since_collision'] += 1
    if parameters["time_since_collision"] > MAX_BEHAVIOUR_LENGTH:
        parameters["time_since_collision"] = MAX_BEHAVIOUR_LENGTH

    parameters["time_similar"] = howLongSimilar()

    if parameters["collision"] != 0:
        parameters["reacted"] = 0
        collision_val = parameters["collision"]
        parameters["collision"] = 0
        parameters["collision_count"] += 1
        parameters["time_since_collision"] = 0
        insertUpToN(1, collision_rate, IMAGE_RATE * 100)
        print("Collision:", collision_val, "because pitch high")
        return collision_val
    
    elif stopped() and parameters["reacted"] >= IMAGE_RATE: # Don't respond to collision if already have
        with vertical_lock:
            dark_top = any(top_bottom_brightnesses[i] < parameters["top_bottom_brightness_threshold"] for i in range(0,min(parameters["time_similar"] + 5, len(top_bottom_brightnesses))))
        if parameters["direction"] in [1,2,6] and (any(acc < -COLLISION_THRESHOLD for acc in acc_ys[0:3]) or dark_top or pitches[0] > DEFAULT_PITCH_THRESHOLD):
            parameters["reacted"] = 0
            parameters["collision_count"] += 1
            parameters["time_since_collision"] = 0
            insertUpToN(1, collision_rate, IMAGE_RATE * 100)
            print("Collision 1 because stopped and acc low or brightness low or pitch high")
            return 1
        
        elif parameters["direction"] in [3,4,5] and any(acc > COLLISION_THRESHOLD for acc in acc_ys[0:3]):
            parameters["reacted"] = 0
            parameters["collision_count"] += 1
            parameters["time_since_collision"] = 0
            insertUpToN(1, collision_rate, IMAGE_RATE * 100)
            print("Collision 2")
            return 2
        
        elif (np.ptp(np.array(pitches[0:3])) > PITCH_VARY_THRESHOLD and any(pitch < 0 for pitch in pitches[0:3]) and any(pitch > 0 for pitch in pitches[0:3])) or (np.ptp(np.array(rolls[0:3])) > ROLL_VARY_THRESHOLD and any(roll < 0 for roll in rolls[0:3]) and any(roll > 0 for roll in rolls[0:3])): # Pitch varying
            parameters["reacted"] = 0
            parameters["stuck"] = True
            parameters["collision_count"] += 1
            parameters["time_since_collision"] = 0
            insertUpToN(1, collision_rate, IMAGE_RATE * 100)
            print("Collision 4")
            return 4
        
        elif parameters["current_behaviour"] != 5 and ((parameters["direction"] in [1,4]) or parameters["turn_speed"] >= 1):
            parameters["reacted"] = 0
            insertUpToN(0, collision_rate, IMAGE_RATE * 100)
            print("Collision 3")
            return 3

    insertUpToN(0, collision_rate, IMAGE_RATE * 100)
    return 0

# Behaviour choosing - Returns the behaviour the rover should follow
def getBehaviour(parameters):
    parameters["iter_index"] += 1
    abnormality_value = detectAbnormality(parameters)
    
    emotion_dist = softmax(getEmotionState())
    happy = False

    if emotion_dist[1] > 0.3 and config_params["boredom"]: # If bored
        boredom_num = random()
        if 0 <= boredom_num < 0.45 and config_params["happiness"]:
            parameters["probe_angle"] += 1
            if parameters["probe_angle"] > MAX_PROBE_ANGLE:
                parameters["probe_angle"] = MAX_PROBE_ANGLE
        elif 0.45 <= boredom_num < 0.9 and config_params["happiness"]:
            parameters["default_behaviour_length"] -= 1
            if parameters["default_behaviour_length"] < MIN_BEHAVIOUR_LENGTH:
                parameters["default_behaviour_length"] = MIN_BEHAVIOUR_LENGTH
        elif 0.9 <= boredom_num < 0.95 and parameters["probe_direction_time"] > parameters["previous_probe_length"] * 2:
            # Recompute probe direction
            parameters["probe_direction"] = 0
        elif 0.95 <= boredom_num < 0.97 and parameters["current_trapped_behaviour"] in [3,4] and parameters["probe_direction_time"] > parameters["previous_probe_length"] * 2:
            # Probe in opposite direction
            parameters["probe_direction"] = -parameters["probe_direction"]
        elif 0.97 <= boredom_num <= 1:
            # Choose different behaviour
            suggested = randint(0,6)
            while suggested == parameters["current_trapped_behaviour"]:
                suggested = randint(0,6)
            parameters["suggested_behaviour"] = suggested
    
    elif emotion_dist[2] > 0.3 and config_params["fear"]: # If scared
        fear_num = random()
        if 0 <= fear_num < 1/3 and (config_params["anger"] or config_params["happiness"]):
            parameters["pitch_threshold"] -= 1
            if parameters["pitch_threshold"] < MIN_PITCH_THRESHOLD:
                parameters["pitch_threshold"] = MIN_PITCH_THRESHOLD
        elif 1/3 <= fear_num < 2/3 and config_params["happiness"]:
            parameters["back_out_distance"] += 1
            if parameters["back_out_distance"] > MAX_BACK_OUT_DISTANCE:
                parameters["back_out_distance"] = MAX_BACK_OUT_DISTANCE
        elif 2/3 <= fear_num <= 1 and (config_params["anger"] or config_params["happiness"]):
            parameters["speed"] -= 0.01
            if parameters["speed"] < MIN_SPEED:
                parameters["speed"] = MIN_SPEED

    elif emotion_dist[0] > 0.3 and config_params["anger"]: # If angry
        anger_num = random()
        if 0 <= anger_num < 0.495 and (config_params["fear"] or config_params["happiness"]):
            parameters["pitch_threshold"] += 1
            if parameters["pitch_threshold"] > MAX_PITCH_THRESHOLD:
                parameters["pitch_threshold"] = MAX_PITCH_THRESHOLD
        elif 0.495 <= anger_num < 0.99 and (config_params["fear"] or config_params["happiness"]):
            parameters["speed"] += 0.01
            if parameters["speed"] > 1:
                parameters["speed"] = 1
        elif 0.99 <= anger_num <= 1:
            parameters["suggested_behaviour"] = 7 # Attempt vault

    elif emotion_dist[3] > 0.3 and config_params["happiness"]: # If happy
        happy = True
        happiness_num = random()
        if 0 <= happiness_num < 0.2 and config_params["boredom"]:
            parameters["probe_angle"] -= 1
            if parameters["probe_angle"] < MIN_PROBE_ANGLE:
                parameters["probe_angle"] = MIN_PROBE_ANGLE
        elif 0.2 <= happiness_num < 0.4 and config_params["boredom"]:
            parameters["default_behaviour_length"] += 1
            if parameters["default_behaviour_length"] > MAX_BEHAVIOUR_LENGTH:
                parameters["default_behaviour_length"] = MAX_BEHAVIOUR_LENGTH
        elif 0.4 <= happiness_num < 0.6 and (config_params["anger"] or config_params["fear"]):
            if parameters["pitch_threshold"] > DEFAULT_PITCH_THRESHOLD:
                parameters["pitch_threshold"] -= 1
            elif parameters["pitch_threshold"] < DEFAULT_PITCH_THRESHOLD:
                parameters["pitch_threshold"] += 1
        elif 0.6 <= happiness_num < 0.8 and (config_params["anger"] or config_params["fear"]):
            if parameters["speed"] > DEFAULT_SPEED:
                parameters["speed"] -= 0.01
            elif parameters["speed"] < DEFAULT_SPEED:
                parameters["speed"] += 0.01
        elif 0.8 <= happiness_num <= 1 and config_params["fear"]:
            parameters["back_out_distance"] -= 1
            if parameters["back_out_distance"] < MIN_BACK_OUT_DISTANCE:
                parameters["back_out_distance"] = MIN_BACK_OUT_DISTANCE
        
    if parameters["trapped"]:
        parameters["time_trapped"] += 1
        parameters["time_free"] = parameters["time_free"]//2
    else:
        parameters["time_free"] += 1
        parameters["time_trapped"] = parameters["time_trapped"]//2
    if parameters["time_trapped"] < 0:
        parameters["time_trapped"] = 0
    elif parameters["time_trapped"] > MAX_BEHAVIOUR_LENGTH:
        parameters["time_trapped"] = MAX_BEHAVIOUR_LENGTH
    if parameters["time_free"] < 0:
        parameters["time_free"] = 0
    elif parameters["time_free"] > MAX_BEHAVIOUR_LENGTH:
        parameters["probe_direction"] = 0
        parameters["suggested_behaviour"] = -1
        parameters["time_free"] = MAX_BEHAVIOUR_LENGTH

    parameters["how_long_dark"] = 0

    if abnormality_value == 0:
        if parameters["time_since_behaviour_started"] < parameters["behaviour_length"] or (parameters["current_behaviour"] in [3,4] and parameters["turning"] != 3):
            return parameters["current_behaviour"]
        else:
            parameters["trapped"] = False
            parameters["stuck"] = False
            parameters["turning"] = 0
            if happy:
                parameters["probe_direction"] = 0
                parameters["suggested_behaviour"] = -1
            if parameters["speed"] > DEFAULT_SPEED:
                parameters["speed"] -= 0.01
            elif parameters["speed"] < DEFAULT_SPEED:
                parameters["speed"] += 0.01
            if parameters["back_out_distance"] > DEFAULT_BACK_OUT_DISTANCE:
                parameters["back_out_distance"] -= 1
            elif parameters["back_out_distance"] < DEFAULT_BACK_OUT_DISTANCE:
                parameters["back_out_distance"] += 1
            if parameters["current_behaviour"] in [3,4]:
                turning["integral"] = 0
            parameters["obstacle_type"] = 0
            return 0
        
    else:
        parameters["trapped"] = True
        parameters["reversed_by"] = 0
        parameters["turning"] = 0
        parameters["extra_probe_angle"] = 0
        roll_on_obstacle = parameters["roll_on_obstacle"]
        parameters["roll_on_obstacle"] = 0
        with horizontal_lock:
            sides_brightness = left_right_brightnesses[0]

        parameters["how_long_dark"] = howLongDarkBeforeCollision()
        parameters["extra_probe_angle"] += (parameters["how_long_dark"] - parameters["time_similar"] + 1)
            
        if not parameters["suggested_behaviour"] in [-1, 7]:
            suggested = parameters["suggested_behaviour"]
            parameters["suggested_behaviour"] = -1
            if suggested == 3:
                if roll_on_obstacle < 0:
                    parameters["extra_probe_angle"] -= abs(roll_on_obstacle) # Probe more finely if large roll on obstacle
                if sides_brightness < 0:
                    parameters["extra_probe_angle"] -= abs(sides_brightness)/4 # Probe more finely as possibly near gap
                if parameters["current_behaviour"] == 3 and parameters["turning"] in [1,2]:
                    parameters["extra_probe_angle"] -= min((parameters["bearing_turned_from"] - bearings[0]) % 360, (bearings[0] - parameters["bearing_turned_from"]) % 360)
            elif suggested == 4:
                if roll_on_obstacle > 0:
                    parameters["extra_probe_angle"] -= roll_on_obstacle # Probe more finely if large roll on obstacle
                if sides_brightness > 0:
                    parameters["extra_probe_angle"] -= sides_brightness/4 # Probe more finely as possibly near gap
                if parameters["current_behaviour"] == 4 and parameters["turning"] in [1,2]:
                    parameters["extra_probe_angle"] -= min((parameters["bearing_turned_from"] - bearings[0]) % 360, (bearings[0] - parameters["bearing_turned_from"]) % 360)
            return suggested
        
        if abnormality_value == 1: # Do probe
            if parameters["suggested_behaviour"] == 7:
                if parameters["how_long_dark"] - parameters["time_similar"] + 1 < 2:
                    return 7
                else:
                    parameters["suggested_behaviour"] = -1

            parameters["extra_probe_angle"] += (parameters["how_long_dark"] - parameters["time_similar"] + 1)
            
            if parameters["probe_direction"] == -1:
                if roll_on_obstacle < 0:
                    parameters["extra_probe_angle"] -= abs(roll_on_obstacle) # Probe more finely if large roll on obstacle
                if sides_brightness < 0:
                    parameters["extra_probe_angle"] -= abs(sides_brightness)/4 # Probe more finely as possibly near gap
                if parameters["current_behaviour"] == 3 and parameters["turning"] in [1,2]:
                    parameters["extra_probe_angle"] -= min((parameters["bearing_turned_from"] - bearings[0]) % 360, (bearings[0] - parameters["bearing_turned_from"]) % 360)
                return 3
            elif parameters["probe_direction"] == 1:
                if roll_on_obstacle > 0:
                    parameters["extra_probe_angle"] -= roll_on_obstacle # Probe more finely if large roll on obstacle
                if sides_brightness > 0:
                    parameters["extra_probe_angle"] -= sides_brightness/4 # Probe more finely as possibly near gap
                if parameters["current_behaviour"] == 4 and parameters["turning"] in [1,2]:
                    parameters["extra_probe_angle"] -= min((parameters["bearing_turned_from"] - bearings[0]) % 360, (bearings[0] - parameters["bearing_turned_from"]) % 360)
                return 4
            
            angle_to_goal_dir = math.sin((config_params["goal_bearing"] - bearings[0])*math.pi/180)
            angle_to_goal_size = -math.cos((config_params["goal_bearing"] - bearings[0])*2*math.pi/180)
            if abs(roll_on_obstacle) > 5 or abs(sides_brightness) > 10 or angle_to_goal_size > 0:
                if abs(roll_on_obstacle) > abs(sides_brightness)/4 and abs(roll_on_obstacle) > (angle_to_goal_size + 1) * 45:
                    parameters["extra_probe_angle"] -= abs(roll_on_obstacle) # Probe more finely if large roll on obstacle
                    if roll_on_obstacle < 0:
                        return 3
                    else:
                        return 4
                elif abs(sides_brightness)/4 > angle_to_goal_size * 90:
                    parameters["extra_probe_angle"] -= sides_brightness/4 # Probe more finely as possibly near gap
                    if sides_brightness < 0:
                        return 3
                    else:
                        return 4
            if angle_to_goal_dir < 0:
                return 3
            else:
                return 4
                    
        elif abnormality_value == 2:
            # Hit obstacle behind so go forwards
            if parameters["current_behaviour"] in [3,4]:
                # If probing, go forward a bit, then turn
                parameters["extra_probe_angle"] += ((parameters["back_out_distance"] - parameters["reversed_by"]) * 2)
                parameters["back_out_distance"] -= ((parameters["back_out_distance"] - parameters["reversed_by"]) // 2)
                if parameters["back_out_distance"] < MIN_BACK_OUT_DISTANCE:
                    parameters["back_out_distance"] = MIN_BACK_OUT_DISTANCE
                return -1
            return 1
        
        elif abnormality_value == 3:
            parameters["reversed_by"] = IMAGE_RATE + 1
            return 5
        
        elif abnormality_value == 4:
            return 6
        
    return parameters["current_behaviour"]

# Return the speed the rover should turn at at the current moment, given the current and desired bearings
def getTurnSpeed(current_bearing, desired_bearing):
    error = min((current_bearing - desired_bearing) % 360, (desired_bearing - current_bearing) % 360)
    turning["integral"] += error
    derivative = error - turning["previous_error"]
    total = error * turning["Kp"] + turning["integral"] * turning["Ki"] * parameters["turn_amount"] + derivative * turning["Kd"] + 0.7
    turning["previous_error"] = error
    if total > 1:
        total = 1
    elif total < 0.7:
        total = 0.7
    return total

# Behaviour following
# Return the direction the rover should move in at the current point, given the behaviour
def getMovement(parameters):
    parameters["previous_behaviour"] = parameters["current_behaviour"]

    behaviour = getBehaviour(parameters)

    if behaviour >= 0:
        if behaviour != 0:
            if behaviour == parameters["current_trapped_behaviour"]:
                parameters["time_doing_behaviour"] += 1
            else:
                parameters["time_doing_behaviour"] = 0
                parameters["time_since_behaviour_started"] = 0
            parameters["previous_trapped_behaviour"] = parameters["current_trapped_behaviour"]
            parameters["current_trapped_behaviour"] = behaviour
        else:
            parameters["time_since_behaviour_started"] = 0

        parameters["current_behaviour"] = behaviour

    # behaviour = parameters["current_behaviour"]

    if pitches[0] > pitches[1] and (pitches[0] > parameters['pitch_threshold'] or (pitches[0] + (pitches[0] - pitches[1])) > parameters["pitch_threshold"]): # If pitch angle too high # If when pitch high, roll varies in direction, could probe more finely in direction of roll
        # Pitch angle too high
        if parameters["direction"] in [1,2,6] and parameters["obstacle_type"] != -1: # Going up obstacle forwards
            parameters["roll_on_obstacle"] = rolls[0]
            parameters["collision"] = 1
            parameters["obstacle_type"] = 1
            return 4
        elif parameters["obstacle_type"] != 1: # Going down hole backwards
            parameters["collision"] = 2
            parameters["obstacle_type"] = -1
            return 1
    
    elif pitches[0] < pitches[1] and (pitches[0] < -parameters["pitch_threshold"] or (pitches[0] + (pitches[0] - pitches[1])) < -parameters["pitch_threshold"]): # If pitch angle too low # If on a branch, try to wiggle off (small branch)
        # Pitch angle too low
        if parameters["direction"] in [3,4,5] and parameters["obstacle_type"] != -1: # Going up obstacle backwards
            parameters["roll_on_obstacle"] = rolls[0]
            parameters["collision"] = 2
            parameters["obstacle_type"] = 1
            return 1
        elif parameters["obstacle_type"] != 1: # Going down hole forwards
            parameters["collision"] = 1
            parameters["obstacle_type"] = -1
            return 4
    
    elif rolls[0] > parameters["roll_threshold"]: # If roll angle too high - left side up
        parameters["roll_on_obstacle"] = rolls[0]
        parameters["turn_amount"] = WIDE_TURN_AMOUNT
        if pitches[0] > 0: # If pitch upwards
            parameters["collision"] = 1
            return 3
        else: # If pitch downwards
            parameters["collision"] = 2
            return 2
        
    elif rolls[0] < -parameters["roll_threshold"]: # If roll angle too low - right side up
        parameters["roll_on_obstacle"] = rolls[0]
        parameters["turn_amount"] = WIDE_TURN_AMOUNT
        if pitches[0] > 0: # If pitch upwards
            parameters["collision"] = 1
            return 5
        else: # If pitch downwards
            parameters["collision"] = 2
            return 6
        
    elif np.ptp(np.array(rolls[0:3])) > ROLL_VARY_THRESHOLD and any(roll < 0 for roll in rolls[0:3]) and any(roll > 0 for roll in rolls[0:3]):
        if pitches[0] * 2 > parameters["pitch_threshold"] and acc_ys[0] < -0.1 and parameters["direction"] in [1,2,6]:
            parameters["collision"] = 1
            return 4
        elif pitches[0] * 2 < -parameters["pitch_threshold"] and acc_ys[0] > 0.1 and parameters["direction"] in [3,4,5]:
            parameters["collision"] = 2
            return 1

    if parameters["extra_probe_angle"] + parameters["probe_angle"] > MAX_PROBE_ANGLE + 10:
        parameters["extra_probe_angle"] = MAX_PROBE_ANGLE + 10 - parameters["probe_angle"]
    elif parameters["extra_probe_angle"] + parameters["probe_angle"] < MIN_PROBE_ANGLE - 10:
        parameters["extra_probe_angle"] = MIN_PROBE_ANGLE - 10 - parameters["probe_angle"]

    if behaviour == -2:
        # Reverse and continue
        return 4
        
    elif behaviour == -1:
        # Go forwards and continue
        parameters["reversed_by"] = parameters["back_out_distance"] / parameters["speed"] # Finish reversing
        return 1

    elif behaviour == 0: # Move towards goal
        if math.cos((config_params["goal_bearing"] - bearings[0])*math.pi/180) < GOAL_THRESHOLD:
            parameters["turn_amount"] = 1
            parameters["turn_speed"] = getTurnSpeed(bearings[0], config_params["goal_bearing"])
            if math.sin((config_params["goal_bearing"] - bearings[0])*math.pi/180) < 0:
                if not parameters["side"] in [3,6]:
                    parameters["side"] = 6
                if stopped(): # If stopped try the wheels on the other side
                    if parameters["side"] == 6:
                        parameters["side"] = 3
                    else:
                        parameters["side"] = 6
                return parameters["side"]
            else:
                if not parameters["side"] in [2,5]:
                    parameters["side"] = 2
                if stopped(): # If stopped try the wheels on the other side
                    if parameters["side"] == 2:
                        parameters["side"] = 5
                    else:
                        parameters["side"] = 2
                return parameters["side"]
        else: # Go forwards
            turning["integral"] = 0
            return 1
    
    elif behaviour == 1: # Move forwards
        parameters["behaviour_length"] = round(parameters["default_behaviour_length"] / parameters["speed"] / 2)
        parameters["time_since_behaviour_started"] += 1
        return 1
    
    elif behaviour == 2: # Back out
        parameters["behaviour_length"] = round(parameters["default_behaviour_length"] / parameters["speed"] / 3)
        parameters["time_since_behaviour_started"] += 1
        return 4
    
    elif behaviour == 3: # Probe left, try 30 degrees, set angle as a parameter, also how much to reverse by, then turn less if more back  
        if parameters["probe_direction"] != -1:
            parameters["previous_probe_length"] = parameters["probe_direction_time"]
            parameters["probe_direction_time"] = 0
            parameters["probe_direction"] = -1
        else:
            parameters["probe_direction_time"] += 1
        parameters["behaviour_length"] = round(parameters["default_behaviour_length"] / parameters["speed"] / 2)
        if parameters["turning"] == 0:
            if parameters["reversed_by"] < parameters["back_out_distance"] / parameters["speed"]:
                parameters["reversed_by"] += 1
                return 4
            else:
                parameters["bearing_turning_to"] = bearings[0] - parameters["probe_angle"] - parameters["extra_probe_angle"]
                parameters["bearing_turned_from"] = bearings[0]
                parameters["extra_probe_angle"] = 0
                parameters["turning"] = 1
                parameters["reversed_by"] = 0
                turning["integral"] = 0
                parameters["obstacle_type"] = 0
                parameters["turn_speed"] = getTurnSpeed(bearings[0], parameters["bearing_turning_to"])
                parameters["turn_amount"] = 1
                parameters["side"] = 6
                return 6 # Turn left
        elif parameters["turning"] == 1:
            if math.sin((parameters["bearing_turning_to"] - bearings[0])*math.pi/180) < 0: # If not yet turned n degrees, turn more
                # Turning left away from obstacle
                parameters["turn_speed"] = getTurnSpeed(bearings[0], parameters["bearing_turning_to"])
                parameters["turn_amount"] = 1
                if stopped():
                    if parameters["side"] == 3:
                        parameters["side"] = 6
                    else:
                        parameters["side"] = 3
                return parameters["side"]
            else:
                # Turning right slowly back towards obstacle
                parameters["turning"] = 2
                turning["integral"] = 0
                parameters["turn_speed"] = getTurnSpeed(bearings[0], parameters["bearing_turned_from"])
                parameters["turn_amount"] = WIDE_TURN_AMOUNT
                return 2
        elif parameters["turning"] == 2: # Turning right slowly back towards obstacle
            if math.sin((parameters["bearing_turned_from"] - bearings[0])*math.pi/180) > 0: # If not yet turned n degrees, turn more: # If not yet turned n degrees, turn more
                parameters["turn_speed"] = getTurnSpeed(bearings[0], parameters["bearing_turned_from"])
                parameters["turn_amount"] = WIDE_TURN_AMOUNT
                return 2
            else: # Done turning so go forwards
                turning["integral"] = 0
                parameters["turning"] = 3
                parameters["time_since_behaviour_started"] += 1
                return 1
        else:
            parameters["time_since_behaviour_started"] += 1
            return 1
        
    elif behaviour == 4: # Probe right
        if parameters["probe_direction"] != 1:
            parameters["previous_probe_length"] = parameters["probe_direction_time"]
            parameters["probe_direction_time"] = 0
            parameters["probe_direction"] = 1
        else:
            parameters["probe_direction_time"] += 1
        parameters["behaviour_length"] = round(parameters["default_behaviour_length"] / parameters["speed"] / 2)
        if parameters["turning"] == 0:
            if parameters["reversed_by"] < parameters["back_out_distance"] / parameters["speed"]:
                parameters["reversed_by"] += 1
                return 4
            else:
                parameters["bearing_turning_to"] = bearings[0] + parameters["probe_angle"] + parameters["extra_probe_angle"]
                parameters["bearing_turned_from"] = bearings[0]
                parameters["extra_probe_angle"] = 0
                parameters["turning"] = 1
                parameters["reversed_by"] = 0
                turning["integral"] = 0
                parameters["obstacle_type"] = 0
                parameters["turn_speed"] = getTurnSpeed(bearings[0], parameters["bearing_turning_to"])
                parameters["turn_amount"] = 1
                parameters["side"] = 2
                return 2 # Turn right
        elif parameters["turning"] == 1:
            if math.sin((parameters["bearing_turning_to"] - bearings[0])*math.pi/180) > 0: # If not yet turned n degrees, turn more
                # Turning right away from obstacle
                parameters["turn_speed"] = getTurnSpeed(bearings[0], parameters["bearing_turning_to"])
                parameters["turn_amount"] = 1
                if stopped():
                    if parameters["side"] == 5:
                        parameters["side"] = 2
                    else:
                        parameters["side"] = 5
                return parameters["side"]
            else: # Turn left slowly towards obstacle
                parameters["turning"] = 2
                turning["integral"] = 0
                parameters["turn_speed"] = getTurnSpeed(bearings[0], parameters["bearing_turned_from"])
                parameters["turn_amount"] = WIDE_TURN_AMOUNT
                return 6
        elif parameters["turning"] == 2:
            if math.sin((parameters["bearing_turned_from"] - bearings[0])*math.pi/180) < 0: # If not yet turned n degrees, turn more
                parameters["turn_speed"] = getTurnSpeed(bearings[0], parameters["bearing_turned_from"])
                parameters["turn_amount"] = WIDE_TURN_AMOUNT
                return 6
            else: # Done turning so go forwards
                turning["integral"] = 0
                parameters["turning"] = 3
                parameters["time_since_behaviour_started"] += 1
                return 1
        else:
            parameters["time_since_behaviour_started"] += 1
            return 1
    
    elif behaviour == 5: # Move forwards and backwards  # If loose leaves or similar
        parameters["behaviour_length"] = round(parameters["default_behaviour_length"] / parameters["speed"])
        parameters["time_since_behaviour_started"] += 1
        parameters["speed"] += 0.01
        if parameters["speed"] > 1:
            parameters["speed"] = 1
        if (stopped() and parameters["reversed_by"] > IMAGE_RATE) or (parameters["direction"] == 4 and parameters["reversed_by"] > MAX_BACK_OUT_DISTANCE) or parameters["direction"] in [2,3,5,6]: # No movement or reversed far enough
            parameters["reversed_by"] = 0
            if parameters["direction"] in [1,2,6]: # If forwards
                return 4 # Go backwards
            elif parameters["direction"] in [3,4,5]: # If backwards
                parameters["obstacle_type"] = 0
                return 1 # Go forwards
            else:
                return 4
        elif parameters["direction"] == 0:
            parameters["reversed_by"] = 0
            return 1 # Get started if not moving before
        parameters["reversed_by"] += 1
        return parameters["direction"] # Keep going the same way
    
    elif behaviour == 6: # Wiggle
        parameters["behaviour_length"] = round(parameters["default_behaviour_length"] / parameters["speed"] / 2)
        parameters["time_since_behaviour_started"] += 1
        if parameters["direction"] == 6:
            return 1
        else:
            return parameters["direction"] + 1
        
    elif behaviour == 7: # Vault
        parameters["behaviour_length"] = round(parameters["default_behaviour_length"] / parameters["speed"])
        parameters["speed"] += 0.01
        if parameters["speed"] > 1:
            parameters["speed"] = 1
        if parameters["reversed_by"] < parameters["back_out_distance"]:
            parameters["reversed_by"] += 1
            return 4
        else:
            parameters["time_since_behaviour_started"] += 1
            parameters["obstacle_type"] = 0
            return 1
        
    else:
        return 0

# Return the speed and steering values for the rover given the direction intended to move in
def getSpeedAndSteering(parameters):
    direction = getMovement(parameters)
    parameters["direction"] = direction
    if direction == 0:
        return 0, 0 # Stop
    elif direction == 1:
        return parameters["speed"], 0 # Forwards
    elif direction == 2:
        return parameters["turn_speed"], parameters["turn_amount"] # Forwards Right
    elif direction == 3:
        return -parameters["turn_speed"], parameters["turn_amount"] # Backwards Right
    elif direction == 4:
        return -parameters["speed"], 0 # Backwards
    elif direction == 5:
        return -parameters["turn_speed"], -parameters["turn_amount"] # Backwards Left
    elif direction == 6:
        return parameters["turn_speed"], -parameters["turn_amount"] # Forwards Left
    
def captureImage(file_path):
    # Add code for the specific camera used
    return

def getSensorData():
    # Add code to retrieve data from the specific sensor used
    return 0, 0, 0, 0, 0, 0

def setMotors(speed, steering):
    # Add code to process the speed and steering values and apply to the motors of the specific rover used
    return

# Start the image processing thread
Thread(target=imageDataLoop).start()

# The following code has been adapted to remove code specific to the rover and controller used in the project
# With a controller, the rover could be stopped with the press of a button, exiting the loop
# Another button could switch between following the algorithm and responding to joystick input
while True:
    
    ######
    speed = 0
    steering = 0

    speed, steering = getSpeedAndSteering(parameters)

    print("Speed: ", speed, ", Steering: ", steering)

    if speed != 0: # Only capture data when rover is moving
        moving = True

        bearing, pitch, roll, acc_x, acc_y, acc_z = getSensorData()
        bearings.insert(0, bearing)
        pitches.insert(0, pitch)
        rolls.insert(0, roll)
        acc_xs.insert(0, acc_x)
        acc_ys.insert(0, acc_y)
        acc_zs.insert(0, acc_z)
    
    else:
        moving = False

    setMotors(speed, steering)

    time.sleep(interval)

    driveLeft = 0
    driveRight = 0
                    
process_loop = False