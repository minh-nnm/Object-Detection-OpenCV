import cv2
import numpy as np

img = cv2.imread("lena.png")
cap = cv2.VideoCapture(0)

cap.set(3, 1920)
cap.set(4, 1080)

classNames = []
classFile = "Object_Detection_Files/coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n") # rstrip() loại bỏ khoảng trắng

# print(classNames)

configPath = "Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "Object_Detection_Files/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, frame = cap.read()
    classIds, confs, bbox = net.detect(frame, confThreshold= 0.5)
    # print(classIds, bbox)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(frame, box, color=(0, 255, 0), thickness= 2)
            cv2.putText(frame, classNames[classId-1].upper(), (box[0]-10, box[1]-10),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(frame, str(round(confidence*100, 2)), ((box[0] - 10) + 150, (box[1] - 10)),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)


        cv2.imshow("Output", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break