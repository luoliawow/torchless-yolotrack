import cv2
import numpy as np

from .results import Boxes

def preprocess(frame):
    # Read the input image
    original_image: np.ndarray = frame
    [height, width, _] = original_image.shape

    # Prepare a square image for inference
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image

    # Preprocess the image and prepare blob for model
    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
    return blob

def post_process(infered_results):
    rows = infered_results.shape[1]

    boxes = []
    scores = []
    class_ids = []
    xyxy = []

    # Iterate through output to collect bounding boxes, confidence scores, and class IDs
    for i in range(rows):
        current = infered_results[0][i]
        classes_scores = current[4:]
        maxClassIndex = np.argmax(classes_scores)
        maxScore = classes_scores[maxClassIndex]
        if maxScore >= 0.25:
            box = [
                current[0] - (0.5 * current[2]),
                current[1] - (0.5 * current[3]),
                current[2],
                current[3],
            ]
            _xyxy = [
                current[0] - (0.5 * current[2]),
                current[1] - (0.5 * current[3]),
                current[0] + (0.5 * current[2]),
                current[1] + (0.5 * current[3]),
            ]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)
            xyxy.append(_xyxy)


    result_ids = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)
    results = np.concatenate([np.array(xyxy).reshape(-1, 4), np.array(scores).reshape(-1,1), np.array(class_ids).reshape(-1,1)], axis=1)[result_ids]
    return results



def inference(model, frame):
    # Preprocess the input frame
    preprocessed = preprocess(frame)

    # Set the input to the model
    model.setInput(preprocessed)

    # Perform inference and get the outputs
    outputs = model.forward()

    # Reshape the outputs
    outputs = np.array([cv2.transpose(outputs[0])])

    inferenced = post_process(outputs)
    boxes = Boxes(inferenced, frame.shape[:2])
    return boxes

