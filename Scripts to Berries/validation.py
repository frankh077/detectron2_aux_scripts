# import some common libraries
import numpy
import cv2
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

#define paths
path = '/path/to/folder/with/images_to_infer'  
path_result = r'/path/to/folder/where/images_are_saved' 
etiquetas = 'json/file/path/with/labels'

modelo = os.path.basename(path_result)

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = '/banano/uvas/eddy/datasets/datasets/output/modelo_rendondas_largas.pth'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.DATASETS.TEST = ("dataset_test", )
cfg.TEST.DETECTIONS_PER_IMAGE = 200
predictor = DefaultPredictor(cfg)

#inferencia y evaluacion
font = cv2.FONT_HERSHEY_SIMPLEX

f = open(etiquetas)
data = json.load(f)
val_images = data['val']
val_dic = {}
val_images_names = [x['file_name'] for x in val_images]
for el in val_images:
    print(f'el :-{el["file_name"]}-')
    if el['file_name'] in val_images_names:
        print(f"el['file_name']: {el['file_name']}")
        val_dic[el['file_name']] = el
data = val_dic

len_labels = []
cdi_lis = []
dv_list = []
error_list = []
image_list = []

df = pd.DataFrame()
cd = 0
cdi = 0

for image in os.listdir(path):
    if image in val_images_names:
        image_list.append(image)
        im_path = str(os.path.join(path, image))
        im = cv2.imread(im_path)
        outputs = predictor(im)
        pred_len = len(outputs["instances"])
        bbox_raw = outputs['instances'].to('cpu')
        bbox_raw = bbox_raw.get_fields()
        bbox_raw = bbox_raw['pred_boxes'].tensor.numpy()
        bbox_raw = list(map(numpy.ndarray.tolist, bbox_raw))
        bbox_raw = list(map(lambda x: list(map(int, x)), bbox_raw))
        cd = cd + len(bbox_raw)
        cdi = len(bbox_raw)


        scores_raw = outputs['instances'].to('cpu')
        scores_raw = scores_raw.get_fields()
        scores_raw = scores_raw['scores'].numpy()

        shapes_images = []
        idx = 0

        for p in data[image]['annotations']:
            idx = idx + 1 
            el = np.array(p['segmentation'][0])
            el_reshaped = el.reshape(round(len(el)/2),2)
            shapes_images.append(el)
            result = np.mean(el_reshaped, axis=0)
            result = np.round(result)
            result = (int(result[0]), int(result[1]))
            im = cv2.polylines(im, [el_reshaped], True, (255,0,0),2)
            im = cv2.putText(im,'idx:'+ str(idx),(result), font, 0.3,(255,255,255),1,cv2.LINE_AA)
        len_labels.append(len(shapes_images))


        idx = 0
        for bbox, score in zip(bbox_raw, scores_raw):
            idx = idx + 1 
            left_top = tuple(bbox[:2])
            right_bottom = tuple(bbox[2:])
            score_height = (bbox[0], bbox[1] - 5) 
            im = cv2.rectangle(im,right_bottom,left_top,(0,0,255),2)
            im = cv2.putText(im,"{:.2f}".format(score) + 'idx:'+ str(idx),score_height, font, 0.3,(0,0,255),1,cv2.LINE_AA)
        
        os.chdir(path_result)
        cv2.imwrite(image, im.astype(np.float32))

        cdi_lis.append(cdi)
        dv = abs(cdi - len(shapes_images))
        dv_list.append(dv)
        if len(shapes_images) != 0:
            error = (dv/(len(shapes_images)))
        else:
            error = 0
        error_list.append(error)


print('CANTIDAD DE DETECCIONES:', cd)

df['labels'] = len_labels
df['detections'] = cdi_lis
df['dv'] = dv_list
df['% error'] = error_list
df = df.set_index([pd.Index(image_list)])
error_mean = df['% error'].mean()
df.loc['total'] = df.sum(axis=0)
x,y = df.shape
df.iloc[x-1, y-1] = error_mean
df.loc['error','labels'] = df['dv'].sum()/df['labels'].sum()

df.to_csv(modelo+'_validacion.csv')
