import math
import cv2
import time
from picamzero import Camera
from exif import Image
from datetime import datetime
from os import listdir
import os

result = 0
flightheight = 416920
focallength = 6    
sensorwidth = 6.287
sensorheight = 4.712
imagewidth = 1.412
imageheight = 1.9377
""" imagewidth = 1.073
imageheight = 0.805 """
""" imagewidth = 4056
imageheight = 3040 """
cam = Camera()
cam.resolution = (4056, 3040)
time.sleep(2)
TimeToRun = 10 * 60

def round_to_significant_digits(num, sig):
    if num == 0:
        return 0
    else:
        return round(num, sig - int(math.floor(math.log10(abs(num)))) - 1)

def get_time(image):
    with open(image, 'rb') as image_file:
        img = Image(image_file)
        time_str = img.get("datetime_original")
        time = datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
    return time

def get_time_difference(image_1, image_2):
    time_1, time_2 = get_time(image_1), get_time(image_2)
    return abs((time_2 - time_1).total_seconds()) if time_1 and time_2 else 0


def convert_to_cv(image):
    return cv2.imread(image, 0)

def calculate_features(image_1_cv, image_2_cv, feature_number=1000):
    orb = cv2.ORB_create(nfeatures=feature_number)
    keypoints_1, descriptors_1 = orb.detectAndCompute(image_1_cv, None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(image_2_cv, None)
    return keypoints_1, keypoints_2, descriptors_1, descriptors_2

def calculate_matches(descriptors_1, descriptors_2):
    if descriptors_1 is None or descriptors_2 is None:
        return []
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    return sorted(brute_force.match(descriptors_1, descriptors_2), key=lambda x: x.distance)


def find_matching_coordinates(keypoints_1, keypoints_2, matches):
    coordinates_1, coordinates_2 = [], []
    for match in matches:
        coordinates_1.append(keypoints_1[match.queryIdx].pt)
        coordinates_2.append(keypoints_2[match.trainIdx].pt)
    return coordinates_1, coordinates_2

def calculate_mean_distance(coordinates_1, coordinates_2):
    distances = [math.hypot(x1 - x2, y1 - y2) for (x1, y1), (x2, y2) in zip(coordinates_1, coordinates_2)]
    return sum(distances) / len(distances) if distances else 0

def calculate_GSD():
    gsdh = (flightheight * sensorheight) / (focallength * imageheight)
    gsdw = (flightheight * sensorwidth) / (focallength * imagewidth)
    return max(gsdh, gsdw)

def calculate_speed_in_kmps(feature_distance, GSD, time_difference):
    distance = feature_distance * GSD / 100000
    speed = distance / time_difference
    return speed

start_time = time.time()
speeds = []
output_file = "result.txt"

with open(output_file, "w") as file:
    while time.time() - start_time < TimeToRun:
        prefix = f"new_sequence_{int(time.time())}"
        cam.capture_sequence(prefix, num_images=2, interval=1)
        image_1 = f'./{prefix}-1.jpg'
        image_2 = f'./{prefix}-2.jpg'
        time_difference = get_time_difference(image_1, image_2) # Get time difference between images
        image_1_cv, image_2_cv = convert_to_cv(image_1), convert_to_cv(image_2) # Create OpenCV image objects
        keypoints_1, keypoints_2, descriptors_1, descriptors_2 = calculate_features(image_1_cv, image_2_cv, 1000) # Get keypoints and descriptors
        matches = calculate_matches(descriptors_1, descriptors_2) # Match descriptors
        coordinates_1, coordinates_2 = find_matching_coordinates(keypoints_1, keypoints_2, matches)
        average_feature_distance = calculate_mean_distance(coordinates_1, coordinates_2)
        speed = calculate_speed_in_kmps(average_feature_distance, 12648, time_difference)
        speeds.append(speed)
        os.remove(image_1)
        os.remove(image_2)
        time.sleep(1)

    avrgSpeed = 0
    for speed in speeds:
        avrgSpeed += speed
    avrgSpeed /= len(speeds)
    print(avrgSpeed)
    file.write(f"{round_to_significant_digits(avrgSpeed, 5)}")
    print(f"Resultaat opgeslagen in {output_file}")
print(listdir)