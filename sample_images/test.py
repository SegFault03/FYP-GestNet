import numpy as np
import tensorflow as tf
import mediapipe as mp
import copy
import itertools
import cv2
import os

def get_images():
    return 

def predict(image):
    labels = [
    'Stop',
    'One',
    'Two',
    'Three',
    'Four',
    'Thumbs Up',
    'Thumbs Down'
    ]
    model_path = 'GestNet.tflite'
    interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=1)

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_details_tensor_index = input_details[0]['index']
    interpreter.set_tensor(
            input_details_tensor_index,
            np.array([image], dtype=np.float32))
    interpreter.invoke()

    output_details_tensor_index = output_details[0]['index']

    result = interpreter.get_tensor(output_details_tensor_index)
    print("Finished prediction")
    result_index = np.argmax(np.squeeze(result))
    print(f'The predicted gesture is: {labels[result_index]}')

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

if __name__ == '__main__':
    print("Starting inference....")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode='store_true',
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    landmarks = []
    image_list = os.listdir('./')
    print("Checking images....")
    for imagepath in image_list:
        if imagepath.split('.')[-1]=='jpg':
            print("Pre-processing images....")
            image = cv2.imread(imagepath, cv2.IMREAD_COLOR)    
            results = hands.process(image)
            if results.multi_hand_landmarks is not None:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                    # Landmark calculation
                    landmark_list = calc_landmark_list(image, hand_landmarks)

                    # Conversion to relative coordinates / normalized coordinates
                    print("Pre-processed images complete.....")
                    pre_processed_image = pre_process_landmark(
                        landmark_list)
                    predict(pre_processed_image)
            break
