import math
import time

import cv2
import keyboard
import mediapipe as mp


# =========================================================
# SETTINGS
# =========================================================

CAMERA_INDEX = 0

PINCH_THRESHOLD = 0.045

VOLUME_COOLDOWN = 0.20
MEDIA_COOLDOWN = 1.00


# =========================================================
# MEDIAPIPE
# =========================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# =========================================================
# HELPERS
# =========================================================

def calculate_distance(point1, point2) -> float:
    """
    Calculate distance between two landmarks.
    """

    return math.hypot(
        point1.x - point2.x,
        point1.y - point2.y
    )


def is_pinching(thumb_tip, finger_tip) -> bool:
    """
    Check if thumb and finger are touching.
    """

    distance = calculate_distance(
        thumb_tip,
        finger_tip
    )

    return distance < PINCH_THRESHOLD


# =========================================================
# ACTIONS
# =========================================================

def volume_up():
    keyboard.send("volume up")


def volume_down():
    keyboard.send("volume down")


def video_forward():
    keyboard.send("right")


def video_backward():
    keyboard.send("left")


def mute():
    keyboard.send("volume mute")


# =========================================================
# MAIN
# =========================================================

def main():

    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        print("Failed to open webcam.")
        return

    last_volume_time = 0.0
    last_media_time = 0.0
    last_mute_time = 0.0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:

        while True:

            success, frame = camera.read()

            if not success:
                break

            # Mirror webcam
            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            results = hands.process(rgb_frame)

            current_time = time.time()

            if results.multi_hand_landmarks and results.multi_handedness:

                for hand_landmarks, hand_info in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness
                ):

                    hand_label = hand_info.classification[0].label

                    landmarks = hand_landmarks.landmark

                    thumb_tip = landmarks[
                        mp_hands.HandLandmark.THUMB_TIP
                    ]

                    index_tip = landmarks[
                        mp_hands.HandLandmark.INDEX_FINGER_TIP
                    ]

                    middle_tip = landmarks[
                        mp_hands.HandLandmark.MIDDLE_FINGER_TIP
                    ]

                    ring_tip = landmarks[
                        mp_hands.HandLandmark.RING_FINGER_TIP
                    ]

                    pinky_tip = landmarks[
                        mp_hands.HandLandmark.PINKY_TIP
                    ]

                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

                    # =================================================
                    # INDEX FINGER -> VOLUME
                    # =================================================

                    if is_pinching(thumb_tip, index_tip):

                        if current_time - last_volume_time > VOLUME_COOLDOWN:

                            if hand_label == "Right":
                                volume_up()

                            else:
                                volume_down()

                            last_volume_time = current_time

                    # =================================================
                    # MIDDLE FINGER -> VIDEO SEEK
                    # =================================================

                    if is_pinching(thumb_tip, middle_tip):

                        if current_time - last_media_time > MEDIA_COOLDOWN:

                            if hand_label == "Right":
                                video_forward()

                            else:
                                video_backward()

                            last_media_time = current_time

                    # =================================================
                    # PINKY -> MUTE
                    # =================================================

                    if is_pinching(thumb_tip, pinky_tip):

                        if current_time - last_mute_time > 1.0:

                            mute()

                            last_mute_time = current_time

                    # =================================================
                    # DEBUG TEXT
                    # =================================================

                    cv2.putText(
                        frame,
                        hand_label,
                        (
                            int(thumb_tip.x * frame.shape[1]),
                            int(thumb_tip.y * frame.shape[0])
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )

            # =========================================================
            # UI TEXT
            # =========================================================

            controls = [
                "Right Index Pinch = Volume Up",
                "Left Index Pinch  = Volume Down",
                "Right Middle Pinch = Forward",
                "Left Middle Pinch  = Backward",
                "Pinky Pinch = Mute",
                "ESC = Quit"
            ]

            y = 30

            for text in controls:

                cv2.putText(
                    frame,
                    text,
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )

                y += 30

            cv2.imshow(
                "Gesture System Control",
                frame
            )

            key = cv2.waitKey(1)

            if key == 27:
                break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()