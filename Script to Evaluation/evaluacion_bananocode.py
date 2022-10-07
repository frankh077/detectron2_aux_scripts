#definir paths
path = '/banano/uvas/racimos/datasets_entrenamiento/test'  #path de imagenes a inferir y evaluar
path_result = r'/banano/uvas/results/evaluacion/modelo_color_giros_crops/inferencia'     #path de carpeta donde se guardará las imagenes inferidas

# import some common libraries
import numpy
import cv2
import os
import json
import pandas as pd
import numpy as np

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import os
import numpy as np
import json
from detectron2.structures import BoxMode

dataset = os.path.basename(path)

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
#cfg.MODEL.WEIGHTS = '/home/omia/banano/uvas/eddy/datasets/datasets/output/model_final.pth'
#cfg.MODEL.WEIGHTS = '/banano/uvas/racimos/datasets_entrenamiento/output__prev/modelo_racimos_v3.pth'
#cfg.MODEL.WEIGHTS = '/banano/uvas/racimos/datasets_entrenamiento/output_consolidadov2_aumentado_color/model_final.pth'
cfg.MODEL.WEIGHTS = '/banano/uvas/racimos/datasets_entrenamiento/evidentes/output/model_final.pth'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.DATASETS.TEST = ("dataset_test", )
cfg.TEST.DETECTIONS_PER_IMAGE = 200
predictor = DefaultPredictor(cfg)

def get_dataset_dicts(directory):
    classes = ['disease']
    dataset_dicts = []
    for idx, filename in enumerate([file for file in os.listdir(directory) if file.endswith('.json')]):
        #print(f'idx: {idx}, filename: {filename}')
        json_file = os.path.join(directory, filename)
        with open(json_file) as f:
            img_anns = json.load(f)

        record = {}
        
        filename = os.path.join(directory, img_anns["imagePath"])
        
        record["file_name"] = filename
        #print(f'idx: {idx}, filename: {filename}')
        record["image_id"] = idx
        record["height"] = 600
        record["width"] = 800
      
        annos = img_anns["shapes"]
        objs = []
        for anno in annos:
            px = [a[0] for a in anno['points']]
            py = [a[1] for a in anno['points']]
            poly = [(x, y) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": classes.index(anno['label']),
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    #print(f'{dataset_dicts}')
    return dataset_dicts

#inferencia y evaluacion
dataset_dicts = get_dataset_dicts(path)
font = cv2.FONT_HERSHEY_SIMPLEX

len_labels = []
cdi_lis = []
dv_list = []
error_list = []
image_list = []
c_precision = []
c_pred_reg = []
c_preci_reg = []

df = pd.DataFrame()
cd = 0
cdi = 0
for image in os.listdir(path):
    if image.lower().endswith(('.png', '.jpg')):
        image_list.append(image)
        im = cv2.imread(os.path.join(path, image))
        outputs = predictor(im)
        pred_len = len(outputs["instances"])
        bbox_raw = outputs['instances'].to('cpu')
        bbox_raw = bbox_raw.get_fields()
        bbox_raw = bbox_raw['pred_boxes'].tensor.numpy()
        bbox_raw = list(map(numpy.ndarray.tolist, bbox_raw))
        bbox_raw = list(map(lambda x: list(map(int, x)), bbox_raw))#esta
        cd = cd + len(bbox_raw)
        cdi = len(bbox_raw)

        scores_raw = outputs['instances'].to('cpu')
        scores_raw = scores_raw.get_fields()
        scores_raw = scores_raw['scores'].numpy()

        shapes_images = []
        idx = 0
        for d in dataset_dicts:
            file_name = os.path.basename(d["file_name"])
            if image == file_name:
                for cord in d['annotations']:
                    instp = list(cord['bbox'])
                    if isinstance(instp[0], numpy.int64):
                        x1,y1,x2,y2 = instp
                        instp = [instp]
                    elif isinstance(instp[0], int):
                        x1,y1,x2,y2 = instp
                        instp = [instp]
                    shapes_images.append(instp)
                    #print(file_name,':', len(instp))

                    for i in instp:
                        idx = idx + 1
                        x1,y1,x2,y2 = i
                        lt = (x1,y1)
                        rb = (x2,y2)
                        score_height = (lt[0], lt[1] - 5)
                        im = cv2.rectangle(im,lt,rb,(255,0,0),2)
                        #im = cv2.putText(im,'Real'+str(idx),score_height, font, 0.3,(255,0,0),1,cv2.LINE_AA)
                        im = cv2.putText(im,str(idx),score_height, font, 0.6,(255,0,0),1,cv2.LINE_AA)

        len_labels.append(len(shapes_images))


        idx = 0
        for bbox, score in zip(bbox_raw, scores_raw):
            idx = idx + 1 
            left_top = tuple(bbox[:2])
            right_bottom = tuple(bbox[2:])
            score_height = (bbox[0], bbox[1] - 5) 
            im = cv2.rectangle(im,right_bottom,left_top,(0,0,255),2)
            #im = cv2.putText(im,"{:.2f}".format(score),score_height, font, 0.3,(0,0,255),1,cv2.LINE_AA)
            im = cv2.putText(im,str(idx),score_height, font, 0.6,(0,0,255),1,cv2.LINE_AA)
            
        os.chdir(path_result)
        cv2.imwrite(image, im.astype(np.float32))

        # #print('DETECCIONES POR IMAGEN:', cdi)
        # cdi_lis.append(cdi)
        # #diferencia de detecciones - reales
        # dv = abs(cdi - len(shapes_images))
        # dv_list.append(dv)
        # #precision
        # precision = (1-abs(cdi-len(shapes_images))/len(shapes_images))
        # c_precision.append(precision)
        # #multiplicacion de detecciones con coeficiente de regresion lineal
        # pred_reg = (cdi*1.15)
        # c_pred_reg.append(pred_reg)
        # #precision de regresion lineal
        # preci_reg = (1-abs(pred_reg-len(shapes_images))/len(shapes_images))
        # c_preci_reg.append(preci_reg)
        # if len(shapes_images) != 0:
        #     error = (dv/(len(shapes_images)))
        # else:
        #     error = 0
        # error_list.append(error)

print('CANTIDAD DE DETECCIONES:', cd)

# df['labels'] = len_labels
# df['detections'] = cdi_lis
# df['dv'] = dv_list
# df['% error'] = error_list
# df = df.set_index([pd.Index(image_list)])
# error_mean = df['% error'].mean()
# df.loc['total'] = df.sum(axis=0)
# x,y = df.shape
# df.iloc[x-1, y-1] = error_mean
# df.loc['error','labels'] = df['dv'].sum()/df['labels'].sum()

# df['labels'] = len_labels
# df['detections'] = cdi_lis
# df['precision'] = c_precision
# df['pred_reg'] = c_pred_reg
# df['precision reg'] = c_preci_reg
# df = df.set_index([pd.Index(image_list)])
# precision_individual = df['precision'].mean()
# precision_total = df['detections'].sum()/df['labels'].sum()
# precision_reg_individual = df['precision reg'].mean()
# precision_reg_total = (1-abs((df['pred_reg'].sum()-df['labels'].sum())/df['labels'].sum()))
# df.loc['total'] = df.sum(axis=0)
# df.loc['Promedio precision individual','labels'] = precision_individual
# df.loc['Precision de los totales','labels'] = precision_total
# df.loc['Promedio precision reg individual','labels'] = precision_reg_individual
# df.loc['Precision reg de los totales','labels'] = precision_reg_total

# df.to_csv(dataset+'_evaluacion.csv')