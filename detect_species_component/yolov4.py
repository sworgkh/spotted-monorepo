from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
import time
import json
 
def yolo_detect(im=None,
                pathIn=None,
                label_path='./cfg/obj.names',
                config_path='./cfg/yolov4-obj.cfg',
                weights_path='./cfg/yolov4-obj_best.weights',
                confidence_thre=0.5,
                nms_thre=0.3):
    labels = open(label_path).read().strip().split("\n")
    if pathIn == None:
        img = im
    else:
        img = cv2.imread(pathIn)
    filename = pathIn.split('/')[-1]
    (H, W) = img.shape[:2]
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    boxes = []
    confidences = []
    classIDs = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > confidence_thre:
                # Restore the coordinates of the bounding box to match the original picture
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                # Calculate the position of the upper left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                # Filter objet with low confidence 
                confidences.append(float(confidence))
                classIDs.append(classID)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_thre, nms_thre)
    lab = []
    loc = []
    resultdata=[]
    data = {}
    data["filename"]=filename
    data["counts"]=len(idxs)
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            info = {'label':labels[classIDs[i]],"confidences":confidences[i],"x":str(x),"y":str(y),"w":str(w),"h":str(h)}
            resultdata.append([info])
            data['data']=resultdata
    res = data
    
    return lab, img, loc, res