import certifi
import ssl
import os

# Set the SSL certificate file path
os.environ['SSL_CERT_FILE'] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context

import cv2
import face_recognition
import numpy as np
import dlib
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

# Load the pre-trained model
weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
model = fasterrcnn_resnet50_fpn(weights=weights)
model.eval()

# Define COCO classes (80 classes)
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def get_prediction(img, threshold=0.5):
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    pred = model([img])
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold]
    pred_boxes = pred_boxes[:len(pred_t)]
    pred_class = pred_class[:len(pred_t)]
    return pred_boxes, pred_class

def load_and_encode_images(image_files):
    known_face_encodings = []
    known_face_names = []
    
    for file in image_files:
        image = face_recognition.load_image_file(file['path'])
        encodings = face_recognition.face_encodings(image)
        if encodings:
            encoding = encodings[0]
            known_face_encodings.append(encoding)
            known_face_names.append(file['name'])
            print(f"Loaded and encoded {file['name']} from {file['path']}")
        else:
            print(f"No faces found in the image {file['path']}")
    
    return known_face_encodings, known_face_names

def recognize_faces_and_objects_in_video(known_face_encodings, known_face_names):
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

    video_capture = cv2.VideoCapture(0)  
    
    while True:
        ret, frame = video_capture.read()  
        if not ret:
            print("Failed to capture image from camera. Exiting.")
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        dets = detector(rgb_frame, 1)
        print(f"Number of faces detected: {len(dets)}")

        face_encodings = []
        for det in dets:
            shape = sp(rgb_frame, det)
            face_descriptor = facerec.compute_face_descriptor(rgb_frame, shape)
            face_encodings.append(np.array(face_descriptor))
        
        for det, face_encoding in zip(dets, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            
            left, top, right, bottom = det.left(), det.top(), det.right(), det.bottom()
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        
        boxes, classes = get_prediction(rgb_frame, threshold=0.6)
        for box, cls in zip(boxes, classes):
            (x1, y1), (x2, y2) = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, cls, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Video', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()

# Example usage
image_files = [
    {'path': 'rohan_shah.jpg', 'name': 'Rohan Shah'},
    {'path': 'rohan_shah.png', 'name': 'Rohan Shah'},
]

known_face_encodings, known_face_names = load_and_encode_images(image_files)

if not known_face_encodings:
    print("No known face encodings loaded. Exiting.")
else:
    recognize_faces_and_objects_in_video(known_face_encodings, known_face_names)
