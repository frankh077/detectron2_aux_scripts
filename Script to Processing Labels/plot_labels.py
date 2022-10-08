## Convertidor JSON VIA elipses a formato para Detectron Mask RCNN
import json
import cv2
from matplotlib import pyplot as plt
import numpy as np
import detectron2.structures.boxes
import random
from detectron2.structures import BoxMode
import os


finished = 80

filename = '/banano/uvas/racimos/datasets_entrenamiento/datasetCabernet/test/via_region_data.json'
#Cantidad de fotos etiquetadas completamente
path = '/banano/uvas/racimos/datasets_entrenamiento/datasetCabernet/test'
dataset_name = 'datasetCabernet'


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
        #points.append(round(y))
        i += 1



    return points

os.chdir(path)

data = json.load(open(filename,"r"))
curr = 0
dataset = []

for k,v in data.items():#k = "0.jpg"
    img_path = v["filename"]
    #print('IMG_PATH:', img_path)
    try:
        img = cv2.imread(img_path)
        img.shape
    except:
        print(f'NO SE PUEDO LEER {img_path}')
        a=0
    else:
        try:
            region_shape = v['regions'][0]['shape_attributes']['name']
        except:
            print(f'Imagen {img_path} no tiene etiquetas')
        else:
            if region_shape == "ellipse":
                img_data = {'file_name':img_path,'image_id':int(k.split(".")[0]),'height':img.shape[0],'width':img.shape[1],'annotations':[]}
                for i in v["regions"]:
                    el = i["shape_attributes"]
                    bbox = Ellipse2BBox(el)
                    mask = Ellipse2Mask(el,30)
                    annotation = {'iscrowd':0,'segmentation':[flatten(mask)],'bbox':[bbox["x"],bbox["y"],bbox["width"],bbox["height"]],'bbox_mode':detectron2.structures.boxes.BoxMode.XYWH_ABS,'category_id':0}
                    img_data["annotations"].append(annotation)

                dataset.append(img_data)
                curr +=1
                if(curr == finished):
                    break

            if region_shape == "polygon":
                img_data = {'file_name':img_path,'image_id':(k.split(".")[0]),'height':img.shape[0],'width':img.shape[1],'annotations':[]}
                for idx, region in enumerate(v['regions']):
                    points = list(map(list,list(zip(region['shape_attributes']['all_points_x'], region['shape_attributes']['all_points_y']))))
                    px = [a[0] for a in points]
                    py = [a[1] for a in points]
                    annotation = {'iscrowd':0,'segmentation':[flatten(points)],'bbox':[int(np.min(px)), int(np.min(py)), int(np.max(px)), int(np.max(py))],'bbox_mode':BoxMode.XYXY_ABS,'category_id':0}
                    img_data["annotations"].append(annotation)

                    img = cv2.polylines(img,np.int32([np.array(points)]),True,(0,255,0),thickness=3,lineType=cv2.LINE_8)

                dataset.append(img_data)
                cv2.imwrite(dataset_name+"_etiquetas/"+img_path,img.astype(np.float32))
                curr +=1
                if(curr == finished):
                    break

random.shuffle(dataset)
cut = round(len(dataset)/5)
train = dataset[cut:]
val = dataset[:cut]


json.dump({"train":train,"val":val}, open("dataset.json","w"))