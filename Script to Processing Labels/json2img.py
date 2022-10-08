## Convertidor Json VIA Elipses a Imagen
#NOTA: Es para etiqueta eliptica, no mask.

import json
import cv2
from matplotlib import pyplot as plt
import numpy as np

## PARAMETROS ############
#Nombre del json VIA
filename = "via_project_2Feb2022_16h44m_json.json"

data = json.load(open(filename,"r"))

for k,v in data.items():#k = "0.jpg"
    img_path = v["filename"]
    img = cv2.imread(img_path)
    for i in v["regions"]:
        el = i["shape_attributes"]
        #print(el)
        color = (0,0,255)
        thick = 4
        img = cv2.ellipse(img, (el["cx"],el["cy"]), (round(el["rx"]),round(el["ry"])), el["theta"]*180/np.pi,0.0,360.0,color,thickness=thick)

    #cv2.imshow('',img)
    #cv2.waitKey(0)
    cv2.imwrite("annotations_img/"+k,img)
    

