import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime

current_pitches = []

parameters = {}
parameters["trapped"] = False
parameters["time_trapped"] = 0
parameters["time_free"] = 0
parameters["current_behaviour"] = 0
parameters["previous_behaviour"] = 0
parameters["time_doing_behaviour"] = 0
parameters["time_since_collision"] = 0
parameters["collision_count"] = 0
parameters["iter_index"] = 0

emotions = {}
emotions["anger"] = 0
emotions["boredom"] = 0
emotions["fear"] = 0
emotions["happiness"] = 0

MAX_BEHAVIOUR_LENGTH = 48

# Compute the current emotion state, duplicated here to test already gathered data on versions of the emotion model
def getEmotionState():
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
    emotions["fear"] += ((parameters["collision_rate"] * np.mean(current_pitches[0:3])/45 - parameters["time_since_collision"]/(MAX_BEHAVIOUR_LENGTH/20)))
    if emotions["fear"] > 100:
        emotions["fear"] = 100
    elif emotions["fear"] < 0:
        emotions["fear"] = 0

    # Happiness
    emotions["happiness"] += (parameters["time_free"]/(MAX_BEHAVIOUR_LENGTH*2) + parameters["time_since_collision"]/(MAX_BEHAVIOUR_LENGTH) - parameters["time_trapped"]/(MAX_BEHAVIOUR_LENGTH/10) - (parameters["collision_rate"] * np.mean(current_pitches[0:3])/45) - (oldFear + oldAnger)/200)
    if emotions["happiness"] > 100:
        emotions["happiness"] = 100
    elif emotions["happiness"] < 0:
        emotions["happiness"] = 0

    return np.array([emotions["anger"], emotions["boredom"], emotions["fear"], emotions["happiness"]])

# Reads data from a log file
def readData(filename):
    file = open(filename)

    lines = []

    lines = file.readlines()

    file.close()

    lines.pop()
    l_copy = lines.copy()

    for l in l_copy:
        parts = l.split("\t")
        if len(parts[1]) > 10:
            lines.remove(l)

    data_read = {}
    data_read["datetimes"] = []
    data_read["indexes"] = np.zeros(len(lines)-1,dtype=np.float64)
    data_read["bearings"] = np.zeros(len(lines)-1,dtype=np.float64)
    data_read["pitches"] = np.zeros(len(lines)-1,dtype=np.float64)
    data_read["rolls"] = np.zeros(len(lines)-1,dtype=np.float64)
    data_read["linaccs"] = np.zeros((len(lines)-1,3),dtype=np.float64)
    data_read["gyros"] = np.zeros((len(lines)-1,3),dtype=np.float64)
    data_read["accs"] = np.zeros((len(lines)-1,3),dtype=np.float64)
    data_read["current_behaviours"] = np.zeros(len(lines)-1,dtype=np.float64)
    data_read["time_doing_behaviours"] = np.zeros(len(lines)-1,dtype=np.float64)
    data_read["behaviour_lengths"] = np.zeros(len(lines)-1,dtype=np.float64)
    data_read["time_since_behaviours"] = np.zeros(len(lines)-1,dtype=np.float64)
    data_read["suggested_behaviours"] = np.zeros(len(lines)-1,dtype=np.float64)
    data_read["current_trapped_behaviours"] = np.zeros(len(lines)-1,dtype=np.float64)
    data_read["vertical_brightness_differences"] = np.zeros(len(lines)-1,dtype=np.float64)
    data_read["similarities"] = np.zeros(len(lines)-1,dtype=np.float64)
    data_read["horizontal_differences"] = np.zeros(len(lines)-1,dtype=np.float64)
    data_read["trappeds"] = np.empty(len(lines)-1,dtype=bool)
    data_read["time_trappeds"] = np.zeros(len(lines)-1,dtype=np.float64)
    data_read["time_frees"] = np.zeros(len(lines)-1,dtype=np.float64)
    data_read["time_since_collisions"] = np.zeros(len(lines)-1,dtype=np.float64)
    data_read["collision_counts"] = np.zeros(len(lines)-1,dtype=np.float64)
    data_read["collision_rates"] = np.zeros(len(lines)-1, dtype=np.float64)
    data_read["collisions"] = np.zeros(len(lines)-1,dtype=np.float64)
    data_read["reacteds"] = np.zeros(len(lines)-1,dtype=np.float64)
    data_read["pitch_thresholds"] = np.zeros(len(lines)-1,dtype=np.float64)
    data_read["probe_angles"] = np.zeros(len(lines)-1,dtype=np.float64)
    data_read["back_out_distances"] = np.zeros(len(lines)-1,dtype=np.float64)
    data_read["extra_probe_angles"] = np.zeros(len(lines)-1,dtype=np.float64)
    data_read["roll_on_obstacles"] = np.zeros(len(lines)-1,dtype=np.float64)
    data_read["darks"] = np.zeros(len(lines)-1,dtype=np.float64)
    data_read["probe_directions"] = np.zeros(len(lines)-1,dtype=np.float64)
    data_read["probe_times"] = np.zeros(len(lines)-1,dtype=np.float64)
    data_read["previous_probe_lengths"] = np.zeros(len(lines)-1,dtype=np.float64)
    data_read["bearing_turning_tos"] = np.zeros(len(lines)-1,dtype=np.float64)
    data_read["bearing_turned_froms"] = np.zeros(len(lines)-1,dtype=np.float64)
    data_read["turnings"] = np.zeros(len(lines)-1,dtype=np.float64)
    data_read["reversed_bys"] = np.zeros(len(lines)-1,dtype=np.float64)
    data_read["obstacle_types"] = np.zeros(len(lines)-1,dtype=np.float64)
    data_read["time_similars"] = np.zeros(len(lines)-1,dtype=np.float64)
    data_read["speeds"] = np.zeros(len(lines)-1,dtype=np.float64)
    data_read["turn_speeds"] = np.zeros(len(lines)-1,dtype=np.float64)
    data_read["turn_amounts"] = np.zeros(len(lines)-1,dtype=np.float64)
    data_read["directions"] = np.zeros(len(lines)-1,dtype=np.float64)
    data_read["sides"] = np.zeros(len(lines)-1,dtype=np.float64)
    data_read["angers"] = np.zeros(len(lines)-1,dtype=np.float64)
    data_read["boredoms"] = np.zeros(len(lines)-1,dtype=np.float64)
    data_read["fears"] = np.zeros(len(lines)-1,dtype=np.float64)
    data_read["happinesses"] = np.zeros(len(lines)-1,dtype=np.float64)
    data_read["emotion_dists"] = np.empty((len(lines)-1,4),dtype=np.float64)

    str_to_bool = {"True": True, "False": False}

    starttime = None
    endtime = None
    time_total = 0
    interventions = 0
    flips = 0
    high_pitches = 0
    desired_bearing_time = 0
    similar_time = 0

    i = 0
    for j in range(0,len(l_copy)-2):
        parts = l_copy[j].split("\t")
        if len(parts[1]) < 10:
            data_read["datetimes"].append(parts[0])
            data_read["indexes"][i] = parts[1]
            angleParts = parts[5].split(" ")
            data_read["bearings"][i] = angleParts[0]
            data_read["pitches"][i] = angleParts[1]
            data_read["rolls"][i] = float(angleParts[2])
            data_read["linaccs"][i] = np.array(parts[7].split(" "))
            data_read["gyros"][i] = np.array(parts[9].split(" "))
            data_read["accs"][i] = np.array(parts[11].split(" "))
            behaviours = parts[13].split(' ')
            data_read["current_behaviours"][i] = behaviours[0]
            data_read["time_doing_behaviours"][i] = behaviours[1]
            data_read["behaviour_lengths"][i] = behaviours[2]
            data_read["time_since_behaviours"][i] = behaviours[3]
            data_read["suggested_behaviours"][i] = behaviours[4]
            data_read["current_trapped_behaviours"][i] = behaviours[5]
            image_data = parts[15].split(' ')
            data_read["vertical_brightness_differences"][i] = image_data[1]
            data_read["similarities"][i] = image_data[2]
            data_read["horizontal_differences"][i] = image_data[3]
            trapped_data = parts[17].split(' ')
            data_read["trappeds"][i] = str_to_bool[trapped_data[0]]
            data_read["time_trappeds"][i] = trapped_data[2]
            data_read["time_frees"][i] = trapped_data[3]
            collision_data = parts[19].split(' ')
            data_read["time_since_collisions"][i] = collision_data[0]
            data_read["collision_counts"][i] = collision_data[1]
            data_read["collision_rates"][i] = collision_data[2]
            data_read["collisions"][i] = collision_data[3]
            data_read["reacteds"][i] = collision_data[4]
            param_data = parts[21].split(' ')
            data_read["pitch_thresholds"][i] = param_data[0]
            data_read["probe_angles"][i] = param_data[1]
            data_read["back_out_distances"][i] = param_data[2]
            data_read["extra_probe_angles"][i] = param_data[3]
            data_read["roll_on_obstacles"][i] = param_data[4]
            data_read["darks"][i] = param_data[5]
            data_read["probe_directions"][i] = param_data[6]
            data_read["probe_times"][i] = param_data[7]
            data_read["previous_probe_lengths"][i] = param_data[8]
            data_read["bearing_turning_tos"][i] = param_data[9]
            data_read["bearing_turned_froms"][i] = param_data[10]
            data_read["turnings"][i] = param_data[11]
            data_read["reversed_bys"][i] = param_data[12]
            data_read["obstacle_types"][i] = param_data[13]
            data_read["time_similars"][i] = param_data[14]
            speed_data = parts[23].split(' ')
            data_read["speeds"][i] = speed_data[0]
            data_read["turn_speeds"][i] = speed_data[1]
            data_read["turn_amounts"][i] = speed_data[2]
            data_read["directions"][i] = speed_data[3]
            data_read["sides"][i] = speed_data[4]
            emotion_data = parts[25].split(' ')
            data_read["angers"][i] = emotion_data[0]
            data_read["boredoms"][i] = emotion_data[1]
            data_read["fears"][i] = emotion_data[2]
            data_read["happinesses"][i] = emotion_data[3]

            if data_read["pitches"][i] > 45:
                high_pitches += 1

            if math.cos((data_read["bearings"][i] - data_read["bearings"][0])*math.pi/180) > 0.98:
                desired_bearing_time += 1

            if data_read["graph_similarities"][i] < 40:
                similar_time += 1

            i += 1
        else:
            if parts[1].strip() == "Using Algorithm" and starttime == None:
                starttime = datetime.strptime(parts[0].strip(), "%Y-%m-%d %H:%M:%S.%f")
            elif parts[1].strip() == "Using Controller" and starttime != None:
                endtime = datetime.strptime(parts[0].strip(), "%Y-%m-%d %H:%M:%S.%f")
                time_total += (endtime - starttime).total_seconds()
                interventions += 1
                if abs(data_read["rolls"][i-1]) == 90:
                    flips += 1
                starttime = None

    if starttime != None:
        endtime = datetime.strptime(lines[len(lines)-1].split("\t")[0].strip(), "%Y-%m-%d %H:%M:%S.%f")
        time_total += (endtime - starttime).total_seconds()
        interventions += 1

    previous_index = -1
    increment = 0.1
    for j in range(0, len(data_read["indexes"])):
        if data_read["indexes"][j] == previous_index:
            data_read["indexes"][j] += increment
            increment += 0.1
        else:
            previous_index = data_read["indexes"][j]
            increment = 0.1
        parameters["iter_index"] = j
        parameters["collision_count"] = data_read["collision_counts"][j]
        parameters["collision_rate"] = data_read["collision_rates"][j]
        parameters["current_behaviour"] = data_read["current_behaviours"][j]
        if j > 0:
            parameters["previous_behaviour"] = data_read["current_behaviours"][j-1]
        parameters["time_doing_behaviour"] = data_read["time_doing_behaviours"][j]
        parameters["time_free"] = data_read["time_frees"][j]
        parameters["time_trapped"] = data_read["time_trappeds"][j]
        parameters["time_since_collision"] = data_read["time_since_collisions"][j]
        parameters["trapped"] = data_read["trappeds"][j]
        current_pitches.insert(0, data_read["pitches"][j])

        # The following code can be used to recalculate the emotions at each increment based on the data read in:
        # emotions_list = getEmotionState()
        # angers[j] = emotions["anger"]
        # boredoms[j] = emotions["boredom"]
        # fears[j] = emotions["fear"]
        # happinesses[j] = emotions["happiness"]

        emotions_list = np.array([data_read["angers"][j],data_read["boredoms"][j],data_read["fears"][j],data_read["happinesses"][j]])
        data_read["emotion_dists"][j] = np.exp(emotions_list)/sum(np.exp(emotions_list))

        data_read["time_total"] = time_total
        data_read["interventions"] = interventions - 1
        data_read["flips"] = flips
        data_read["high_pitches"] = high_pitches/i*100
        data_read["desired_bearing_time"] = desired_bearing_time/i*100
        data_read["similar_time"] = similar_time/i*100

    return data_read

# Makes graphs of the data from a log file of the attributes listed in the input "graphs", 
# centred on a specific index "point", with a range either side of size "span"
def makeGraphs(filepath, point, span, graphs):
    data = readData(filepath)
    start = point - span
    end = point + span

    rows = len(graphs)
    fig, ax = plt.subplots(nrows=rows, ncols=1, figsize=(160,80))
    for i in range(0,rows):
        ax[i].grid(True)
        ax[i].set_xlabel("Data index", fontsize=10)
        ax[i].set_xticks(np.arange(len(data[1])))
        ax[i].set_xlim(left=start,right=end)
        ax[i].axvline(x=point, color='red', linestyle='--')
    j = 0
    for i in graphs:
        if i == "bearing":
            ax[j].plot(data["indexes"], data["bearings"], c='b')
            ax[j].set_ylabel("Compass bearing (°)", fontsize=10)
            ax[j].set_ylim(bottom=0,top=360)
        elif i == "pitch":
            ax[j].plot(data["indexes"], data["pitches"], c='b')
            ax[j].set_ylabel("Pitch angle (°)", fontsize=10)
            ax[j].set_ylim(bottom=-100,top=100)
        elif i == "roll":
            ax[j].plot(data["indexes"], data["rolls"], c='b')
            ax[j].set_ylabel("Roll angle (°)", fontsize=10)
            ax[j].set_ylim(bottom=-100,top=100)
        elif i == "acc_x":
            ax[j].plot(data["indexes"], data["linaccs"][:,0], c='b')
            ax[j].set_ylabel("Linear acceleration x axis (m/s^2)", fontsize=10)
            ax[j].set_ylim(bottom=-10,top=10)
        elif i == "acc_y":
            ax[j].plot(data["indexes"], data["linaccs"][:,1], c='b')
            ax[j].set_ylabel("Linear acceleration y axis (m/s^2)", fontsize=10)
            ax[j].set_ylim(bottom=-15,top=15)
        elif i == "acc_z":
            ax[j].plot(data["indexes"], data["linaccs"][:,2], c='b')
            ax[j].set_ylabel("Linear acceleration z axis (m/s^2)", fontsize=10)
            ax[j].set_ylim(bottom=-0.1,top=0.1)
        elif i == "gyro_x":
            ax[j].plot(data["indexes"], data["gryos"][:,0], c='b')
            ax[j].set_ylabel("Gyroscope x axis output (rad/s)", fontsize=10)
            ax[j].set_ylim(bottom=-250,top=250)
        elif i == "gyro_y":
            ax[j].plot(data["indexes"], data["gyros"][:,1], c='b')
            ax[j].set_ylabel("Gyroscope y axis output (rad/s)", fontsize=10)
            ax[j].set_ylim(bottom=-200,top=200)
        elif i == "gyro_z":
            ax[j].plot(data["indexes"], data["gyros"][:,2], c='b')
            ax[j].set_ylabel("Gyroscope z axis output (rad/s)", fontsize=10)
            ax[j].set_ylim(bottom=-150,top=150)
        elif i == "accmeter_x":
            ax[j].plot(data["indexes"], data["accs"][:,0], c='b')
            ax[j].set_ylabel("Accelerometer x axis output (m/s^2)", fontsize=10)
            ax[j].set_ylim(bottom=-10,top=10)
        elif i == "accmeter_y":
            ax[j].plot(data["indexes"], data["accs"][:,1], c='b')
            ax[j].set_ylabel("Accelerometer y axis output (m/s^2)", fontsize=10)
            ax[j].set_ylim(bottom=-10,top=10)
        elif i == "accmeter_z":
            ax[j].plot(data["indexes"], data["accs"][:,2], c='b')
            ax[j].set_ylabel("Accelerometer z axis output (m/s^2)", fontsize=10)
            ax[j].set_ylim(bottom=-10,top=30)
        elif i == "behaviour":
            ax[j].plot(data["indexes"], data["current_behaviours"], 'xb')
            ax[j].set_ylabel("Current behaviour", fontsize=10)
            ax[j].set_ylim(bottom=0,top=10)
        elif i == "behaviour_time_doing":
            ax[j].plot(data["indexes"], data["time_doing_behaviours"], c='b')
            ax[j].set_ylabel("Time doing current behaviour", fontsize=10)
        elif i == "behaviour_length":
            ax[j].plot(data["indexes"], data["behaviour_lengths"], c='b')
            ax[j].set_ylabel("Behaviour length", fontsize=10)
        elif i == "behaviour_time_since_start":
            ax[j].plot(data["indexes"], data["time_since_behaviours"], c='b')
            ax[j].set_ylabel("Time since behaviour started", fontsize=10)
        elif i == "behaviour_suggested":
            ax[j].plot(data["indexes"], data["suggested_behaviours"], 'xb')
            ax[j].set_ylabel("Suggested behaviour", fontsize=10)
        elif i == "behaviour_trapped":
            ax[j].plot(data["indexes"], data["indexes"], 'xb')
            ax[j].set_ylabel("Current trapped behaviour", fontsize=10)
        elif i == "brightness_vertical":
            ax[j].plot(data["indexes"], data["vertical_brightness_differences"], c='b')
            ax[j].set_ylabel("Vertical brightness difference", fontsize=10)
            ax[j].set_ylim(bottom=-255,top=255)
        elif i == "similarity":
            ax[j].plot(data["indexes"], data["similarites"], c='b')
            ax[j].set_ylabel("Image similarity", fontsize=10)
            ax[j].set_ylim(bottom=0,top=255)
        elif i == "brightness_horizontal":
            ax[j].plot(data["indexes"], data["horizontal_brightness_differences"], c='b')
            ax[j].set_ylabel("Horizontal brightness difference", fontsize=10)
            ax[j].set_ylim(bottom=-255,top=255)
        elif i == "trapped_free_time":
            ax[j].plot(data["indexes"], data["time_trappeds"], c='b')
            ax[j].plot(data["indexes"], data["time_frees"], c='r')
            ax[j].set_ylabel("Time trapped (blue), Time free (red)", fontsize=8)
        elif i == "collision_time":
            ax[j].plot(data["indexes"], data["time_since_collisions"], c='b')
            ax[j].set_ylabel("Time since collision", fontsize=10)
        elif i == "collision_count":
            ax[j].plot(data["indexes"], data["collision_counts"], c='b')
            ax[j].set_ylabel("Collision count", fontsize=10)
        elif i == "collision_rate":
            ax[j].plot(data["indexes"], data["collision_rates"], c='b')
            ax[j].set_ylabel("Collision rate", fontsize=10)
            ax[j].set_ylim(bottom=0,top=100)
        elif i == "collision_value":
            ax[j].plot(data["indexes"], data["collisions"], 'xb')
            ax[j].set_ylabel("Collision value passed in", fontsize=10)
        elif i == "time_since_reacted":
            ax[j].plot(data["indexes"], data["reacteds"], c='b')
            ax[j].set_ylabel("Time since reacted to abnormality", fontsize=10)
        elif i == "pitch_threshold":
            ax[j].plot(data["indexes"], data["pitch_thresholds"], c='b')
            ax[j].set_ylabel("Pitch threshold", fontsize=10)
            ax[j].set_ylim(bottom=0,top=45)
        elif i == "probe_angle":
            ax[j].plot(data["indexes"], data["probe_angles"], c='b')
            ax[j].set_ylabel("Probe angle", fontsize=10)
            ax[j].set_ylim(bottom=0,top=90)
        elif i == "back_out_distance":
            ax[j].plot(data["indexes"], data["back_out_distances"], c='b')
            ax[j].set_ylabel("Back out distance", fontsize=10)
        elif i == "extra_probe_angle":
            ax[j].plot(data["indexes"], data["extra_probe_angles"], c='b')
            ax[j].set_ylabel("Extra probe angle", fontsize=10)
            ax[j].set_ylim(bottom=-90,top=90)
        elif i == "roll_on_obstacle":
            ax[j].plot(data["indexes"], data["roll_on_obstacles"], 'xb')
            ax[j].set_ylabel("Roll on obstacle", fontsize=10)
        elif i == "time_dark":
            ax[j].plot(data["indexes"], data["darks"], 'xb')
            ax[j].set_ylabel("Time dark", fontsize=10)#
        elif i == "probe_direction":
            ax[j].plot(data["indexes"], data["probe_directions"], 'xb')
            ax[j].set_ylabel("Probe direction", fontsize=10)
        elif i == "probe_time":
            ax[j].plot(data["indexes"], data["probe_times"], c='b')
            ax[j].set_ylabel("Time probing in direction", fontsize=10)
        elif i == "previous_probe_length":
            ax[j].plot(data["indexes"], data["previous_probe_lengths"], c='b')
            ax[j].set_ylabel("Previous probe length", fontsize=10)
        elif i == "bearing_to":
            ax[j].plot(data["indexes"], data["bearing_turning_tos"], 'xb')
            ax[j].set_ylabel("Bearing turning to", fontsize=10)
        elif i == "bearing_from":
            ax[j].plot(data["indexes"], data["bearing_turned_froms"], 'xb')
            ax[j].set_ylabel("Bearing turned from", fontsize=10)
        elif i == "probe_step":
            ax[j].plot(data["indexes"], data["turnings"], c='b')
            ax[j].set_ylabel("Probing step", fontsize=10)
        elif i == "reversed_distance":
            ax[j].plot(data["indexes"], data["reversed_bys"], c='b')
            ax[j].set_ylabel("Distance reversed", fontsize=10)
        elif i == "obstacle_type":
            ax[j].plot(data["indexes"], data["obstacle_types"], 'xb')
            ax[j].set_ylabel("Obstacle type", fontsize=10)
        elif i == "time_similar":
            ax[j].plot(data["indexes"], data["time_similars"], c='b')
            ax[j].set_ylabel("Time similar", fontsize=10)
        elif i == "speed":
            ax[j].plot(data["indexes"], data["speeds"], c='b')
            ax[j].set_ylabel("Speed", fontsize=10)
            ax[j].set_ylim(bottom=0,top=1)
        elif i == "direction":
            ax[j].plot(data["indexes"], data["directions"], c='b')
            ax[j].set_ylabel("Direction", fontsize=10)
        elif i == "side_turning":
            ax[j].plot(data["indexes"], data["sides"], c='b')
            ax[j].set_ylabel("Side turning", fontsize=10)
        elif i == "emotions":
            ax[j].plot(data["indexes"], data["angers"], c='r')
            ax[j].plot(data["indexes"], data["boredoms"], c='y')
            ax[j].plot(data["indexes"], data["fears"], c='b')
            ax[j].plot(data["indexes"], data["happinesses"], c='g')
            ax[j].set_ylabel("Emotions", fontsize=10)
        elif i == "emotions_softmax":
            ax[j].plot(data["indexes"], data["emotion_dists"][:,0], c='r')
            ax[j].plot(data["indexes"], data["emotion_dists"][:,1], c='y')
            ax[j].plot(data["indexes"], data["emotion_dists"][:,2], c='b')
            ax[j].plot(data["indexes"], data["emotion_dists"][:,3], c='g')
            ax[j].set_ylabel("Emotion softmax", fontsize=10)
        j += 1
    plt.tight_layout()
    return fig

# Makes a set of box plots for tests in multiple locations comparing with and without emotions
# Each input is a list holding lists representing locations, each holds file paths of data logs of trials in that location
# One input is for trials with the emotion model present and the other trials without it
# This function requires the lists to both be the same length, with the lists representing each location being in the same order
# The number of data logs in each location list does not matter, but more improves the box plots
def makeComparisonBoxPlots(no_emotion_filepaths, emotion_filepaths):
    collected_data = [[],[]]
    labels = []
    location_count = 1
    for location in range(0,len(collected_data[0])):
        collected_data[0].append([])
        collected_data[1].append([])
        labels.append("Site " + str(location_count) + "\nBasic")
        labels.append("Site " + str(location_count) + "\nEmotions")

    for x in range(0,len(emotion_filepaths)):
        for y in range(0,len(emotion_filepaths[x])):
            collected_data[0][x].append(readData(emotion_filepaths[x][y]))

    for x in range(0,len(no_emotion_filepaths)):
        for y in range(0,len(no_emotion_filepaths[x])):
            collected_data[1][x].append(readData(no_emotion_filepaths[x][y]))

    full_list = []
    for metric in ["time_total", "interventions", "flips", "high_pitches", "desired_bearing_time", "similar_time"]:
        combined_list = []
        for location in range(0,len(collected_data[0])):
            noemotions = []
            withemotions = []
            for trial in range(0,len(collected_data[0][location])):
                noemotions.append(collected_data[0][location][trial][metric])
                withemotions.append(collected_data[1][location][trial][metric])
            print(noemotions)
            combined_list.append(np.array(noemotions))            
            print(withemotions)
            combined_list.append(np.array(withemotions))
        full_list.append(np.array(combined_list))

    titles = ["Test completion time", "Number of manual interventions", "Number of times flipped over", "Percentage of time close to flipping over", "Percentage of time moving in desired direction", "Percentage of time with similar consecutive images"]
    plt.rcParams.update({'font.size': 23})
    fig, ax = plt.subplots(2,3, figsize=(30,20))
    n = 0
    # Make a 2x3 grid of sets of box plots for each metric, each containing one for with and without the emotion model for each location
    for i in range(0,2):
        for j in range(0,3):
            ax[i][j].boxplot(full_list[n].swapaxes(0,1),tick_labels=labels,flierprops=dict(marker="x",markersize=10))
            ax[i][j].set_title(titles[n])
            n += 1
    plt.tight_layout()
    # plt.show()
    return fig

# Makes a set of box plots for groups of tests in one location each with one emotion missing
# The input list holds four lists, each containing file paths of data log files from trials with one emotion missing
# The order is: no anger, no boredom, no fear, no happiness
# The number of data logs in each list does not matter, but more improves the box plots
def makeAblationStudyBoxPlots(filepaths):
    collected_data = [[],[],[],[]]

    for x in range(0,len(filepaths)):
        for y in range(0,len(filepaths[x])):
            collected_data[x].append(readData(filepaths[x][y]))

    ablation_list = []
    for metric in ["time_total", "interventions", "flips", "high_pitches", "desired_bearing_time", "similar_time"]:
        combined_ablation_list = []
        for missing_emotion in range(0,4):
            trial_data = []
            for trial in range(0,len(collected_data[missing_emotion])):
                trial_data.append(collected_data[missing_emotion][trial][metric])
            print(trial_data)
            combined_ablation_list.append(np.array(trial_data))
        ablation_list.append(np.array(combined_ablation_list))

    titles = ["Test completion time", "Number of manual interventions", "Number of times flipped over", "Percentage of time close to flipping over", "Percentage of time moving in desired direction", "Percentage of time with similar consecutive images"]
    labels = ["No Anger","No Boredom","No Fear","No Happiness"]
    plt.rcParams.update({'font.size': 23})
    fig, ax = plt.subplots(2,3, figsize=(30,20))
    n = 0
    # Make a 2x3 grid of sets of box plots for each metric, each containing box plots for each missing emotion
    for i in range(0,2):
        for j in range(0,3):
            ax[i][j].boxplot(ablation_list[n].swapaxes(0,1),tick_labels=labels,flierprops=dict(marker="x",markersize=10))
            ax[i][j].set_title(titles[n])
            n += 1
    plt.tight_layout()
    # plt.show()
    return fig

filepath = "" # Add path to robot_data.txt file here
makeGraphs(filepath, 130, 20, ["bearing", "pitch", "roll", "brightness_horizontal", "brightness_vertical", "similarity", "emotions", "behaviour"]) # Example inputs

# Fill with file paths
filepaths1 = [["","","","",""],["","","","",""],["","","","",""]]
filepaths2 = [["","","","",""],["","","","",""],["","","","",""]]
filepaths3 = [["","","","",""],["","","","",""],["","","","",""],["","","","",""]]

makeComparisonBoxPlots(filepaths1, filepaths2)
makeAblationStudyBoxPlots(filepaths2)
