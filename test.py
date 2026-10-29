# Hand Gesture Recognition using Python and OpenCV
#
# Description:
# This program uses the computer's webcam to detect hand gestures in real-time.
# It leverages the MediaPipe library for accurate hand landmark detection and OpenCV
# for video capture and display. The program can recognize three basic gestures:
# "Open Palm", "Fist", and "Pointing Up".
#
# Required Libraries:
# - opencv-python: For capturing video from the webcam and displaying the feed.
# - mediapipe: For the pre-trained hand detection and tracking model.
#
# You can install these libraries using pip:
# pip install opencv-python mediapipe

import cv2
import mediapipe as mp

def main():
    """
    Main function to run the hand gesture recognition application.
    """
    # Initialize MediaPipe Hands solution
    mp_hands = mp.solutions.hands
    # The 'with' statement ensures resources are managed correctly.
    # - static_image_mode=False: Treat video frames as a stream, not static images.
    # - max_num_hands=1: Detect only one hand for simplicity and performance.
    # - min_detection_confidence: Minimum confidence value for the hand detection to be considered successful.
    # - min_tracking_confidence: Minimum confidence value for the hand landmarks to be tracked.
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils

    # Initialize video capture from the default webcam (index 0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    print("Starting hand gesture recognition. Press 'q' to quit.")

    while cap.isOpened():
        # Read a frame from the webcam
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the frame horizontally for a mirror-like view
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB, as MediaPipe requires RGB input
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to find hand landmarks
        results = hands.process(rgb_frame)

        # Draw hand landmarks and recognize gestures
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw the landmarks and connections on the frame
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                # Recognize the gesture
                gesture = recognize_gesture(hand_landmarks)

                # Display the recognized gesture on the frame
                cv2.putText(
                    frame,
                    f"Gesture: {gesture}",
                    (10, 50),  # Position of the text
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,         # Font scale
                    (0, 0, 255), # Text color in BGR (Red)
                    2,         # Text thickness
                    cv2.LINE_AA
                )

        # Display the resulting frame
        cv2.imshow('Hand Gesture Recognition', frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("Program terminated.")

def recognize_gesture(hand_landmarks):
    """
    Recognizes a gesture based on the positions of hand landmarks.

    Args:
        hand_landmarks: A list of landmarks for a detected hand.

    Returns:
        A string representing the recognized gesture ("Open Palm", "Fist", "Pointing Up", or "Unknown").
    """
    # Get the coordinates of the fingertips and relevant lower joints
    # Landmark indices are from the MediaPipe Hands documentation.
    landmarks = hand_landmarks.landmark
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    index_pip = landmarks[6]
    middle_pip = landmarks[10]
    ring_pip = landmarks[14]
    pinky_pip = landmarks[18]

    # --- Gesture Logic ---

    # 1. Check for "Pointing Up"
    # Condition: Index finger is extended up, and other fingers (middle, ring, pinky) are curled down.
    if (index_tip.y < index_pip.y and
            middle_tip.y > middle_pip.y and
            ring_tip.y > ring_pip.y and
            pinky_tip.y > pinky_pip.y):
        return "Pointing Up"

    # 2. Check for "Open Palm"
    # Condition: All fingertips are above their respective PIP (Proximal Interphalangeal) joints.
    if (thumb_tip.x < index_tip.x and # Simple check for thumb being out
        index_tip.y < index_pip.y and
            middle_tip.y < middle_pip.y and
            ring_tip.y < ring_pip.y and
            pinky_tip.y < pinky_pip.y):
        return "Open Palm"

    # 3. Check for "Fist"
    # Condition: All fingertips are below their respective PIP joints.
    if (index_tip.y > index_pip.y and
            middle_tip.y > middle_pip.y and
            ring_tip.y > ring_pip.y and
            pinky_tip.y > pinky_pip.y):
        return "Fist"

    return "Unknown"

if __name__ == '__main__':
    main()
