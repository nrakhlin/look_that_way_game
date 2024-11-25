from enum import Enum
import random
import time
import sys
import select
import threading

import cv2  # OpenCV
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np


class landmarker_and_result:
    def __init__(self):
        self.result = mp.tasks.vision.HandLandmarkerResult
        self.landmarker = mp.tasks.vision.HandLandmarker
        self.createLandmarker()

    def createLandmarker(self):
        # callback function
        def update_result(
            result: mp.tasks.vision.HandLandmarkerResult,
            output_image: mp.Image,
            timestamp_ms: int,
        ):
            self.result = result

        # HandLandmarkerOptions (details here: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/python#live-stream)
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path="/Users/nrakhlin/Documents/Senior/embeddedos/final_project_scratch/hand_landmarker.task"
            ),  # path to model
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,  # running on a live stream
            num_hands=1,  # track both hands
            min_hand_detection_confidence=0.3,  # lower than value to get predictions more often
            min_hand_presence_confidence=0.3,  # lower than value to get predictions more often
            min_tracking_confidence=0.3,  # lower than value to get predictions more often
            result_callback=update_result,
        )  # Unlike video or image mode, the detect function in live stream mode doesn’t return anything. Instead, it runs asynchronously, providing the results to the function that we pass as the result_callback argument.

        # initialize landmarker
        self.landmarker = self.landmarker.create_from_options(options)

    def detect_async(self, frame):
        # convert np frame to mp image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        # detect landmarks
        self.landmarker.detect_async(
            image=mp_image, timestamp_ms=int(time.time() * 1000)
        )

    def close(self):
        # close landmarker
        self.landmarker.close()


def draw_landmarks_on_image(
    rgb_image, detection_result: mp.tasks.vision.HandLandmarkerResult
):
    """Courtesy of https://github.com/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb"""
    try:
        if detection_result.hand_landmarks == []:
            return rgb_image
        else:
            hand_landmarks_list = detection_result.hand_landmarks
            annotated_image = np.copy(rgb_image)

            # Loop through the detected hands to visualize.
            for idx in range(len(hand_landmarks_list)):
                hand_landmarks = hand_landmarks_list[idx]

                # Draw the hand landmarks.
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend(
                    [
                        landmark_pb2.NormalizedLandmark(
                            x=landmark.x, y=landmark.y, z=landmark.z
                        )
                        for landmark in hand_landmarks
                    ]
                )
                mp.solutions.drawing_utils.draw_landmarks(
                    annotated_image,
                    hand_landmarks_proto,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style(),
                )
            return annotated_image
    except:
        return rgb_image


def count_fingers_raised(
    rgb_image, detection_result: mp.tasks.vision.HandLandmarkerResult
):
    """Iterate through each hand, checking if fingers (and thumb) are raised.
    Hand landmark enumeration (and weird naming convention) comes from
    https://developers.google.com/mediapipe/solutions/vision/hand_landmarker."""
    try:
        # Get Data
        hand_landmarks_list = detection_result.hand_landmarks

        # Code to count numbers of fingers raised will go here
        numRaised = 0
        # for each hand...
        for idx in range(len(hand_landmarks_list)):
            # hand landmarks is a list of landmarks where each entry in the list has an x, y, and z in normalized image coordinates
            hand_landmarks = hand_landmarks_list[idx]
            # for each fingertip... (hand_landmarks 4, 8, 12, and 16)
            for i in range(8, 21, 4):
                # make sure finger is higher in image the 3 proceeding values (2 finger segments and knuckle)
                tip_y = hand_landmarks[i].y
                dip_y = hand_landmarks[i - 1].y
                pip_y = hand_landmarks[i - 2].y
                mcp_y = hand_landmarks[i - 3].y
                if tip_y < min(dip_y, pip_y, mcp_y):
                    numRaised += 1
            # for the thumb
            # use direction vector from wrist to base of thumb to determine "raised"
            tip_x = hand_landmarks[4].x
            dip_x = hand_landmarks[3].x
            pip_x = hand_landmarks[2].x
            mcp_x = hand_landmarks[1].x
            palm_x = hand_landmarks[0].x
            if mcp_x > palm_x:
                if tip_x > max(dip_x, pip_x, mcp_x):
                    numRaised += 1
            else:
                if tip_x < min(dip_x, pip_x, mcp_x):
                    numRaised += 1

        # Code to display the number of fingers raised will go here
        annotated_image = np.copy(rgb_image)
        height, width, _ = annotated_image.shape
        text_x = int(hand_landmarks[0].x * width) - 100
        text_y = int(hand_landmarks[0].y * height) + 50
        cv2.putText(
            img=annotated_image,
            text=str(numRaised) + " Fingers Raised",
            org=(text_x, text_y),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=1,
            color=(0, 0, 255),
            thickness=2,
            lineType=cv2.LINE_4,
        )
        return annotated_image
    except:
        return rgb_image


def get_finger_point(rgb_image, detection_result: mp.tasks.vision.HandLandmarkerResult):
    """Iterate through each hand, checking if fingers (and thumb) are raised.
    Hand landmark enumeration (and weird naming convention) comes from
    https://developers.google.com/mediapipe/solutions/vision/hand_landmarker."""
    state = None
    try:
        # Get Data
        hand_landmarks_list = detection_result.hand_landmarks

        # Code to count numbers of fingers raised will go here
        numRaised = 0
        # for each hand...
        for idx in range(len(hand_landmarks_list)):
            # hand landmarks is a list of landmarks where each entry in the list has an x, y, and z in normalized image coordinates

            # look at pointer finger
            hand_landmarks = hand_landmarks_list[idx]
            # for each fingertip... (hand_landmarks 4, 8, 12, and 16)

            finger = hand_landmarks[8]
            dip = hand_landmarks[8 - 1]
            pip = hand_landmarks[8 - 2]
            mcp = hand_landmarks[8 - 3]

            tip_y = finger.y
            tip_x = finger.x

            dip_y = dip.y
            dip_x = dip.x

            pip_y = pip.y
            pip_x = pip.x

            mcp_y = mcp.y
            mcp_x = mcp.x

            # check pointer finger segments
            state = "oop"
            if tip_y > max(dip_x, pip_x, mcp_x):
                state = "down"
            if tip_y < min(dip_y, pip_y, mcp_y):
                state = "up"
            if tip_x < min(dip_x, pip_x, mcp_x) - 0.01:
                state = "right"
            if tip_x > max(dip_x, pip_x, mcp_x) + 0.01:
                state = "left"

        # Code to display the number of fingers raised will go here
        annotated_image = np.copy(rgb_image)
        height, width, _ = annotated_image.shape
        text_x = int(hand_landmarks[0].x * width) - 100
        text_y = int(hand_landmarks[0].y * height) + 50
        cv2.putText(
            img=annotated_image,
            text=state,
            org=(text_x, text_y),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=1,
            color=(0, 0, 255),
            thickness=2,
            lineType=cv2.LINE_4,
        )
        return annotated_image, state
    except:
        return rgb_image, state


class States(Enum):
    STARTUP = 1
    USR_POINTER = 2
    USR_LOOK = 3
    GAMEOVER = 4
    LEADERBOARD = 5
    SETTINGS = 6


class Directions:
    left = "left"
    right = "right"
    up = "up"
    down = "down"

    dirs = [left, right, up, down]

    def select_random_direction():
        return Directions.dirs[random.randint(0, 3)]


look = ""

point = ""

state = States.STARTUP
next_state = None


countdown = 5
score_multiplier = 1
level = 1
score = 0
points_per_score = 1


def print_remaining_time(timeout, stop_event):
    """Function to print remaining time every second"""
    for remaining in range(timeout, 0, -1):
        if stop_event.is_set():  # Check if the countdown should be stopped
            print("\nCountdown stopped.")
            return
        print(f"\nTime remaining: {remaining} seconds", end="\r")
        time.sleep(1)
    print("Time's up! No input received within the given time.")


def get_decisions():
    usr_point = ""
    while usr_point not in Directions.dirs:
        usr_point = input(
            "Please input your direction for pointing (left, right, up, down): \n"
        )
    time.sleep(0.5)
    # ai_point = Directions.select_random_direction()
    ai_point = "right"
    print(f"ai point direction: {ai_point}")
    return usr_point, ai_point


def get_user_input_with_timeout(timeout=3):
    """Function to get user input with a timeout and display remaining time every second"""

    print(f"You have {timeout} seconds to enter your input:")

    # Create an event to signal when to stop the countdown
    stop_event = threading.Event()

    # Start a background thread to print the countdown
    countdown_thread = threading.Thread(
        target=print_remaining_time, args=(timeout, stop_event)
    )
    countdown_thread.start()

    # Wait for input for 'timeout' seconds
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)

    if rlist:
        user_input = sys.stdin.readline().strip()  # Read the input
        stop_event.set()  # Stop the countdown thread once input is received
        print(f"\nUser input: {user_input}")
        return user_input
    else:
        stop_event.set()  # Stop the countdown thread if no input is received
        print(f"\nTime's up! No input received within {timeout} seconds.")
        return None


def get_point_camera(time_length):
    init_time = time.time()
    max_dir, max_count = "down", 0
    dir_dict = {
        "oop": 0,
        "down": 0,
        "up": 0,
        "right": 0,
        "left": 0,
    }
    
    while time_length > time.time() - init_time:
        # pull frame
        ret, frame = cap.read()
        # update landmarker results
        hand_landmarker.detect_async(frame)

        frame, direction = get_finger_point(frame, hand_landmarker.result)
        
        if direction not in ["oop", None]:
            dir_dict[direction] += 1
    
    for key, value in dir_dict.items():
        if value > max_count:
            max_dir, max_count = key, max_count
    
    return max_dir


def get_head_camera(time_length):
    init_time = time.time()
    max_dir, max_count = "down", 0
    dir_dict = {
        "oop": 0,
        "down": 0,
        "up": 0,
        "right": 0,
        "left": 0,
    }
    
    while time_length > time.time() - init_time:
        ret, frame = cap.read()
        ##facemesh
        results = face_mesh.process(frame)
        frame.flags.writeable = True

        # frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

        img_h , img_w, img_c = frame.shape
        face_2d = []
        face_3d = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx ==1 or idx == 61 or idx == 291 or idx==199:
                        if idx ==1:
                            nose_2d = (lm.x * img_w,lm.y * img_h)
                            nose_3d = (lm.x * img_w,lm.y * img_h,lm.z * 3000)
                        x,y = int(lm.x * img_w),int(lm.y * img_h)

                        face_2d.append([x,y])
                        face_3d.append(([x,y,lm.z]))


                #Get 2d Coord
                face_2d = np.array(face_2d,dtype=np.float64)

                face_3d = np.array(face_3d,dtype=np.float64)

                focal_length = 1 * img_w

                cam_matrix = np.array([[focal_length,0,img_h/2],
                                    [0,focal_length,img_w/2],
                                    [0,0,1]])
                distortion_matrix = np.zeros((4,1),dtype=np.float64)

                success,rotation_vec,translation_vec = cv2.solvePnP(face_3d,face_2d,cam_matrix,distortion_matrix)


                #getting rotational of face
                rmat,jac = cv2.Rodrigues(rotation_vec)

                angles,mtxR,mtxQ,Qx,Qy,Qz = cv2.RQDecomp3x3(rmat)

                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360

                #here based on axis rot angle is calculated
                if y < -5:
                    text="right"
                elif y > 5:
                    text="left"
                elif x < -2.5:
                    text="down"
                elif x > 5:
                    text="up"
                else:
                    text="Forward"
            if text != "Forward":
                dir_dict[text] += 1
    for key, value in dir_dict.items():
        if value > max_count:
            max_dir, max_count = key, max_count
    return max_dir

# access webcam
cap = cv2.VideoCapture(0)
# create landmarker
hand_landmarker = landmarker_and_result()

#vars for face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5,min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(color=(128,0,128),thickness=2,circle_radius=1)

while 1:

    print(f"Score: {round(score, 3)}")
    print(f"Level: {level}")
    usr_point = ""
    ai_point = ""
    # currently in state x
    if state == States.STARTUP:
        print("Welcome to Game")
        pass

    elif state == States.USR_POINTER:
        print("User_pointer")
        # usr_point, ai_point = get_decisions()
        # usr_point = get_user_input_with_timeout(countdown)
        usr_point = get_point_camera(5)
        print(f": {usr_point}")
        time.sleep(0.5)
        ai_point = Directions.select_random_direction()
        print(f"ai look direction: {ai_point}")

    elif state == States.USR_LOOK:
        print("user_looker")
        # usr_point, ai_point = get_decisions()
        # usr_point = get_user_input_with_timeout(countdown)
        usr_point = get_head_camera(5)
        print(f": {usr_point}")
        time.sleep(0.5)
        ai_point = Directions.select_random_direction()
        print(f"ai point direction: {ai_point}")

    elif state == States.GAMEOVER:
        print("Sorry, you lose")
        print(f"Final Score: {score}")
        print(f"Final level: {level}")
        break

    time.sleep(1)

    # handle state transition
    if state == States.STARTUP:
        next_state = States.USR_POINTER

    if state == States.USR_POINTER:
        if usr_point == None:
            next_state = States.GAMEOVER
            print(
                "YOU RAN OUTTA TIME WHY WOULD YOU DO THAT YOU SHOULD BE FASTER WHAT THE HECK"
            )
        elif usr_point == ai_point:
            score += score_multiplier * points_per_score
            next_state = state
            print("Great job, you gottem! Try to catch this individual once more!!")
        else:
            next_state = States.USR_LOOK
            print("Nice try but you missed, Mr.")
            print("Switch sides!")

    if state == States.USR_LOOK:
        if usr_point == None:
            next_state = States.GAMEOVER
            print(
                "YOU RAN OUTTA TIME WHY WOULD YOU DO THAT YOU SHOULD BE FASTER WHAT THE HECK"
            )
        elif usr_point == ai_point:
            next_state = States.GAMEOVER
            print("The AI got you why would you let it do that!!")
        else:
            countdown = countdown - countdown * 0.2
            score_multiplier = score * 1.1
            level += 1
            next_state = States.USR_POINTER
            print("Nice job, you evaded the AI!!")

    time.sleep(1)
    print("\n")

    state = next_state
