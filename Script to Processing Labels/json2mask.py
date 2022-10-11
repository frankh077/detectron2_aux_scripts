## Convertidor JSON VIA elipses a formato para Detectron Mask RCNN

from asyncio import subprocess
import json
import cv2
from matplotlib import pyplot as plt
import numpy as np
import detectron2.structures.boxes
import random
import subprocess


#Nombre del json VIA
filename = "via_project_2Feb2022_16h44m_json.json"
#Cantidad de fotos etiquetadas completamente
finished = 83

def flatten(t):
    return [item for sublist in t for item in sublist]


def Ellipse2BBox(ellipse):
    a = ellipse["rx"]*np.cos(ellipse["theta"])
    b = ellipse["ry"]*np.sin(ellipse["theta"])
    c = ellipse["rx"]*np.sin(ellipse["theta"])
    d = ellipse["ry"]*np.cos(ellipse["theta"])

    width = round(np.sqrt(a**2.0 + b**2.0) * 2.0)
    height = round(np.sqrt(c**2.0 + d**2.0) * 2.0)
    x = round(ellipse["cx"] - width * 0.5)
    y = round(ellipse["cy"] - height * 0.5)
    
    return {"x":x,"y":y,"width":width,"height":height}


def Ellipse2Mask(ellipse,n):    
    dt = 2*np.pi/n

    points = []
    i = 0
    while dt*i < 2*np.pi:
        x = round(ellipse["rx"]*np.cos(i*dt)*np.cos(ellipse["theta"]) - ellipse["ry"]*np.sin(i*dt)*np.sin(ellipse["theta"]) + ellipse["cx"])
        y = round(ellipse["rx"]*np.cos(i*dt)*np.sin(ellipse["theta"]) + ellipse["ry"]*np.sin(i*dt)*np.cos(ellipse["theta"]) + ellipse["cy"])

        points.append([x,y])
        i += 1



    return points


data = json.load(open(filename,"r"))


curr = 0

dataset = []

for k,v in data.items():

    img_path = v["filename"]
    img = cv2.imread(img_path)

    img_data = {'file_name':img_path,'image_id':int(k.split(".")[0]),'height':img.shape[0],'width':img.shape[1],'annotations':[]}

    for i in v["regions"]:
        el = i["shape_attributes"]
        #print(el)
        color = (0,0,255)
        thick = 4
        bbox = Ellipse2BBox(el)
        img = cv2.rectangle(img,(bbox["x"],bbox["y"]),(bbox["x"]+bbox["width"],bbox["y"]+bbox["height"]),(255,0,0),thickness=3)
        mask = Ellipse2Mask(el,30)
        
        annotation = {'iscrowd':0,'segmentation':[flatten(mask)],'bbox':[bbox["x"],bbox["y"],bbox["width"],bbox["height"]],'bbox_mode':detectron2.structures.boxes.BoxMode.XYWH_ABS,'category_id':0}
        img_data["annotations"].append(annotation)

        img = cv2.polylines(img,np.int32([np.array(mask)]),True,(0,255,0),thickness=3,lineType=cv2.LINE_8)

    dataset.append(img_data)

    print('annotations_img_2/'+k)
    subprocess.run(['pwd'])
    cv2.imwrite("annotations_img_2/"+k, img.astype(np.float32))

    curr +=1
    if(curr == finished):
        break


random.shuffle(dataset)
cut = round(len(dataset)/5)
train = dataset[cut:]
val = dataset[:cut]


json.dump({"train":train,"val":val}, open("dataset.json","w"))