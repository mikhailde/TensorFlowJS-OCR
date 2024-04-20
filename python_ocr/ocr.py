import cv2
import numpy as np
import tensorflow as tf
import time
import math


def non_max_suppression(boxes, probs=None, overlapThresh=0.3):
    if not boxes:
        return []

    if isinstance(boxes[0][0], int):
        boxes = [[float(j) for j in i] for i in boxes]

    pick = []

    x1 = [box[0] for box in boxes]
    y1 = [box[1] for box in boxes]
    x2 = [box[2] for box in boxes]
    y2 = [box[3] for box in boxes]

    area = [(x2[i] - x1[i] + 1) * (y2[i] - y1[i] + 1) for i in range(len(boxes))]
    idxs = y2

    if probs is not None:
        idxs = probs

    idxs = sorted(range(len(idxs)), key=lambda k: idxs[k], reverse=True)

    while idxs:
        i = idxs[-1]
        pick.append(i)

        xx1 = [max(x1[i], x1[j]) for j in idxs[:-1]]
        yy1 = [max(y1[i], y1[j]) for j in idxs[:-1]]
        xx2 = [min(x2[i], x2[j]) for j in idxs[:-1]]
        yy2 = [min(y2[i], y2[j]) for j in idxs[:-1]]

        w = [max(0, xx2[j] - xx1[j] + 1) for j in range(len(xx1))]
        h = [max(0, yy2[j] - yy1[j] + 1) for j in range(len(yy1))]

        overlap = [w[j] * h[j] / area[j] for j in range(len(w))]

        idxs = [j for j in range(len(idxs)) if j != len(idxs) - 1 and overlap[j] <= overlapThresh]

    return [list(map(int, boxes[i])) for i in pick]

def preprocess_image(image_path, target_size):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    
    image = image.astype("float32")
    
    mean = [123.68, 116.779, 103.939][::-1]
    for i in range(3):
        image[:,:,i] -= mean[i]
    
    image = image.reshape((1,) + image.shape)
    
    return image

def load_model_and_infer(model_path, image):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], image)
    interpreter.invoke()
    return interpreter

def save_detected_objects(scores, geometry, rW, rH, orig):
    rects = []
    confidences = []
    for y in range(len(scores[0][0])):
        scoresData = scores[0][0][y]
        xData0 = geometry[0][0][y]
        xData1 = geometry[0][1][y]
        xData2 = geometry[0][2][y]
        xData3 = geometry[0][3][y]
        anglesData = geometry[0][4][y]
        for x in range(len(scoresData)):
            if scoresData[x] < 0.5:
                continue
            offsetX, offsetY = x * 4.0, y * 4.0
            angle = anglesData[x]
            cos, sin = math.cos(angle), math.sin(angle)
            h, w = xData0[x] + xData2[x], xData1[x] + xData3[x]
            endX = int(offsetX + cos * xData1[x] + sin * xData2[x])
            endY = int(offsetY - sin * xData1[x] + cos * xData2[x])
            startX, startY = int(endX - w), int(endY - h)
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    boxes = non_max_suppression(rects, probs=confidences)
    for i, (startX, startY, endX, endY) in enumerate(boxes):
        startX, startY, endX, endY = int(startX * rW), int(startY * rH), int(endX * rW), int(endY * rH)
        object_image = orig[startY:endY, startX:endX]
        cv2.imwrite(f"object_{i}.png", object_image)

image_path = 'test2.png'
orig = cv2.imread(image_path)
image = preprocess_image(image_path, (320, 320))
model_path = 'east_model_float16.tflite'
interpreter = load_model_and_infer(model_path, image)

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

scores = interpreter.tensor(output_details[0]['index'])().tolist()
geometry = interpreter.tensor(output_details[1]['index'])().tolist()

scores = np.transpose(scores, (0, 3, 1, 2))
geometry = np.transpose(geometry, (0, 3, 1, 2))

(H, W) = orig.shape[:2]
rW, rH = W / 320, H / 320

save_detected_objects(scores, geometry, rW, rH, orig)