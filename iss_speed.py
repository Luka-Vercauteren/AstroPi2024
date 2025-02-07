from exif import Image
from datetime import datetime
import cv2
import math

flightheight = 416920
focallength = 6	
sensorwidth = 7.857		
sensorheigth = 5.284
imagewidth = 1.073
imageheigth = 0.805

def get_time(image):
    with open(image, 'rb') as image_file:
        img = Image(image_file)
        time_str = img.get("datetime_original")
        time = datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
    return time   
    
def get_time_difference(image_1, image_2):
    time_1 = get_time(image_1)
    time_2 = get_time(image_2)
    time_difference = time_2 - time_1
    return time_difference.seconds

def convert_to_cv(image_1, image_2):
    image_1_cv = cv2.imread(image_1, 0)
    image_2_cv = cv2.imread(image_2, 0)
    return image_1_cv, image_2_cv

def calculate_features(image_1, image_2, feature_number):
    orb = cv2.ORB_create(nfeatures = feature_number)
    keypoints_1, descriptors_1 = orb.detectAndCompute(image_1_cv, None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(image_2_cv, None)
    return keypoints_1, keypoints_2, descriptors_1, descriptors_2

def calculate_matches(descriptors_1, descriptors_2):
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = brute_force.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

image_1 = 'photo_0683.jpg'
image_2 = 'photo_0684.jpg'

def display_matches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches):
    match_img = cv2.drawMatches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches[:100], None)
    resize = cv2.resize(match_img, (1600,600), interpolation = cv2.INTER_AREA)
    cv2.imshow('matches', resize)
    cv2.waitKey(0)
    cv2.destroyWindow('matches')
    display_matches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches)
    
def find_matching_coordinates(keypoints_1, keypoints_2, matches):
    coordinates_1 = []
    coordinates_2 = []
    for match in matches:
        image_1_idx = match.queryIdx
        image_2_idx = match.trainIdx
        (x1,y1) = keypoints_1[image_1_idx].pt
        (x2,y2) = keypoints_2[image_2_idx].pt
        coordinates_1.append((x1,y1))
        coordinates_2.append((x2,y2))
    return coordinates_1, coordinates_2

def calculate_mean_distance(coordinates_1, coordinates_2):
    all_distances = 0
    merged_coordinates = list(zip(coordinates_1, coordinates_2))
    for coordinate in merged_coordinates:
        x_difference = coordinate[0][0] - coordinate[1][0]
        y_difference = coordinate[0][1] - coordinate[1][1]
        distance = math.hypot(x_difference, y_difference)
        all_distances = all_distances + distance
    return all_distances / len(merged_coordinates)

def calculate_GSD(flightheight, focallength, sensorwidth, sensorheigth, imagewidth, imageheigth):
    gsdh = (flightheight * sensorheigth)/(focallength * imageheigth)
    gsdw = (flightheight * sensorwidth)/(focallength * imagewidth)
    print(f"{gsdh} : {gsdw}")
    if gsdh >= gsdw:
        GSD = gsdh
    if gsdh < gsdw:
        GSD = gsdw
    return GSD

def calculate_speed_in_kmps(feature_distance, GSD, time_difference):
    distance = feature_distance * GSD / 100
    speed = distance / time_difference
    return speed

time_difference = get_time_difference(image_1, image_2)
image_1_cv, image_2_cv = convert_to_cv(image_1, image_2)
keypoints_1, keypoints_2, descriptors_1, descriptors_2 = calculate_features(image_1_cv, image_2_cv, 1000)
matches = calculate_matches(descriptors_1, descriptors_2)
#display_matches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches)
coordinates_1, coordinates_2 = find_matching_coordinates(keypoints_1, keypoints_2, matches)
average_feature_distance = calculate_mean_distance(coordinates_1, coordinates_2)
GSD = calculate_GSD(flightheight, focallength, sensorwidth, sensorheigth, imagewidth, imageheigth)
speed = calculate_speed_in_kmps(average_feature_distance, GSD, time_difference)
print(speed)
