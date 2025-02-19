from ultralytics import YOLO
import cv2
from sort.sort import *
from util import get_car, read_license_plate, write_csv


results = {}
tracker = Sort()

#load datasets
coco_model = YOLO("yolov8n.pt")
license_plate_detector = YOLO("./models/bestt.pt")

#load video 
cap = cv2.VideoCapture("./ample.mp4")

vehicles = [2,3,5,6,7]
frame_num = -1
ret = True
while ret:
    frame_num+=1
    ret,frame = cap.read()
    if ret:

        results[frame_num] = {}
        
        #detect vehicles
        detections = coco_model(frame)[0]
        detection_info = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detection_info.append([x1, y1, x2, y2, score])
            
        #track vehicles
        track_ids = tracker.update(np.asarray(detection_info))
        
        #detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            
            #assign plates to cars
            car_x1, car_y1, car_x2, car_y2, car_id = get_car(license_plate, track_ids)
            
            if car_id != -1:
            
                #crop the plate
                license_plate_crop = frame[int(y1): int(y2), int(x1): int(x2), :]
            
                #process the plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
                
                #read license plate
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                
                if license_plate_text is not None:
                    results[frame_num][car_id] = {"car":{"bbox":[car_x1, car_y1, car_x2, car_y2] } ,
                                                "license_plate":{"bbox":[x1, y1, x2, y2] ,"text":license_plate_text ,"bbox_score":score ,"text_score":license_plate_text_score  } }


#results
write_csv(results, "./test.csv")