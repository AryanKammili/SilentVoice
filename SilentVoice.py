import cv2
import numpy as np
import math
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model/model_unquant.tflite")  # <- Update with your path
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prediction function
def get_prediction(image):
    img_resized = cv2.resize(image, (224, 224))  # Input size must match your model
    img_resized = img_resized.astype(np.float32)
    img_resized = (img_resized / 127.5) - 1  # Normalization (Teachable Machine standard)
    img_resized = np.expand_dims(img_resized, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_resized)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = output_data[0]
    index = np.argmax(prediction)
    return prediction, index

# Your label list (match your training labels)
labels = ["A", "B", "C", "D", "E", "F", "G"]  # <- Update these to match your model

# Set up webcam and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300  

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        imgHeight, imgWidth, _ = img.shape
        x1 = max(0, x - offset)
        y1 = max(0, y - offset)
        x2 = min(imgWidth, x + w + offset)
        y2 = min(imgHeight, y + h + offset)
        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size == 0:
            print("Empty crop, skipping frame.")
            continue

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = min(math.ceil(k * w), imgSize)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap: wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = min(math.ceil(k * h), imgSize)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: hCal + hGap, :] = imgResize

        prediction, index = get_prediction(imgWhite)

        # Draw the result
        cv2.rectangle(imgOutput, (x - offset, y - offset - 70),
                      (x - offset + 400, y - offset + 60 - 50), (0, 255, 0), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y - 30),
                    cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (0, 255, 0), 4)
        
        # -- Inside your while loop, replace cv2.imshow(...) with this block:
        border_thickness = 20
        border_color = (255, 255, 255)  # White
        background_color = (71, 62,49)  # Blue (BGR)

        # Add border to the image
        imgBordered = cv2.copyMakeBorder(
            imgOutput,
            border_thickness,
            border_thickness,
            border_thickness,
            border_thickness,
            cv2.BORDER_CONSTANT,
            value=border_color
        )

        # Create background
        bh, bw = imgBordered.shape[:2]
        background = np.full((bh + 100, bw + 100, 3), background_color, dtype=np.uint8)

        # Center the bordered image on the background
        x_offset = (background.shape[1] - bw) // 2
        y_offset = (background.shape[0] - bh) // 2
        background[y_offset:y_offset + bh, x_offset:x_offset + bw] = imgBordered

        cv2.imshow('Styled Webcam Feed', background)
    cv2.waitKey(1)
