
# import some common libraries. 
import cv2
import os
import json
import pandas as pd
import numpy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

input_path = '/path/to/folder/with/images' 
output_path = r'/path/to/folder/where/images_are_saved'

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.structures import BoxMode

dataset = os.path.basename(path)

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
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
            json_file = os.path.join(directory, filename)
            with open(json_file) as f:
                img_anns = json.load(f)

            record = {}
            
            filename = os.path.join(directory, img_anns["imagePath"])
            
            record["file_name"] = filename
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
        return dataset_dicts

def crop_200(x1,y1,x2,y2, height, width):

  dx = abs(x2-x1)
  dy = abs(y2-y1)

  if dx < 200:
    cx = int(abs(dx-200)/2)
    x1 = x1 - cx
    x2 = x2 + cx

    if x1<0:
      x1 = 0 
      dx = abs(x2-x1)
      cx = int(abs(dx-200))
      x2 = x2 + cx

    if x2>width:
      x2 = width
      dx = abs(x2-x1)
      cx = int(abs(dx-200))
      x1 = x1 - cx

  else:
    x1 = x1 - 10
    x2 = x2 + 10

    if x1 < 0:
      x1 = 0

    if x2 > width:
      x2 = width
  

  if dy < 200:
    cy = int(abs(dy-200)/2)
    y1 = y1 - cy
    y2 = y2 + cy

    if y1<0:
      y1 = 0
      dy = abs(y2-y1)
      cy = int(abs(dy-200))
      y2 = y2 + cy

    if y2>height:
      y2 = height
      dy = abs(y2-y1)
      cy = int(abs(dy-200))
      y1 = y1 - cy

  else:
    y1 = y1 - 13
    y2 = y2 + 13
    
    if y1 < 0:
      y1 = 0
    
    if y2 > height:
      y2 = height

  return x1,y1,x2,y2

def bb_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def plot_detecction(i, score, path_image, height, width, fpp_crops):
    x1,y1,x2,y2 = i
    font = cv2.FONT_HERSHEY_SIMPLEX
    im = cv2.imread(path_image)
    lt = (x1,y1)
    rb = (x2,y2)
    im = cv2.rectangle(im,lt,rb,(0,0,255),2)
    score_height = (lt[0], lt[1] - 5)
    x1,y1,x2,y2=crop_200(x1,y1,x2,y2, height, width)
    im = cv2.putText(im,"{:.2f}".format(score),score_height, font, 0.6,(0,0,255),1,cv2.LINE_AA)
    cut = im[y1:y2, x1:x2]
    a,b, _ = cut.shape
    if a > 200 or b > 200:
        cut = cv2.resize(cut,(200,200))
    fpp_crops.append(cut)
    return a

def plot_label(i, path_image, height, width, fnr_crops):
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.imread(path_image)
    x1,y1,x2,y2 = i
    lt = (x1,y1)
    rb = (x2,y2)
    score_height = (lt[0], lt[1] - 5)
    img = cv2.rectangle(img,lt,rb,(255,0,0),2)
    img = cv2.putText(img,'Real',score_height, font, 0.6,(255,0,0),1,cv2.LINE_AA)
    x1,y1,x2,y2=crop_200(x1,y1,x2,y2, height, width)
    cutr = img[y1:y2, x1:x2]
    a,b, _ = cutr.shape
    if a > 200 or b > 200:
        cutr = cv2.resize(cutr,(200,200))
    fnr_crops.append(cutr)
    return a

def plot_no_detecction(i, path_image, height, width, fnp_crops, fn):
    x1,y1,x2,y2 = i
    im = cv2.imread(path_image)
    x1,y1,x2,y2=crop_200(x1,y1,x2,y2, height, width)
    cut = im[y1:y2, x1:x2]
    a,b, _ = cut.shape
    if a > 200 or b > 200:
        cut = cv2.resize(cut,(200,200))
    fnp_crops.append(cut)
    fn.append(1)
    return a

def plot_no_label(i, path_image, height, width, fpr_crops, fp):
    x1,y1,x2,y2 = i
    im = cv2.imread(path_image)
    x1,y1,x2,y2=crop_200(x1,y1,x2,y2, height, width)
    cutr = im[y1:y2, x1:x2]
    a,b, _ = cutr.shape
    if a > 200 or b > 200:
        cutr = cv2.resize(cutr,(200,200))
    fpr_crops.append(cutr)
    fp.append(1)
    return a

def plot_mosaics (titulo, listf, dataset):
    split_size = 48
    final = [listf[i:i+split_size] for i in range(0,len(listf),split_size)]
    for idx, part in enumerate(final):
      nc = 8
      fig = plt.figure(figsize=(9, 6), dpi=300)
      grid = ImageGrid(fig, 111, nrows_ncols=(6, nc), axes_pad=0.2,)
      name = dataset+'-'+titulo+'-' + str(idx) + '.png'
      for ax, im in zip(grid, part):
          ax.tick_params(labelbottom= False,labeltop = False, labelleft = False,  labelright = False)
          ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    
      fig.suptitle(titulo, fontsize=15 )
      fig.savefig(name, pad_inches=0)
      plt.cla
      plt.clf
      plt.close(fig)
    return()

print('EMPIEZA EL CALCULO')
dataset_dicts = get_dataset_dicts(path)
font = cv2.FONT_HERSHEY_SIMPLEX

listp = []
listr = []
fnp_crops = []
fnr_crops = []
fpp_crops = []
fpr_crops = []
fn = []
fp = []
tp = []
for d in dataset_dicts:
    path_image = d["file_name"]
    im = cv2.imread(path_image)
    image = os.path.basename(path_image)
    height = im.shape[0]
    width = im.shape[1]
    outputs = predictor(im)
    #Get bboxes from prediction output
    bbox_raw = outputs['instances'].to('cpu')
    bbox_raw = bbox_raw.get_fields()
    bbox_raw = bbox_raw['pred_boxes'].tensor.numpy()
    bbox_raw = list(map(numpy.ndarray.tolist, bbox_raw))
    bbox_raw = list(map(lambda x: list(map(int, x)), bbox_raw))

    #Get scores from prediction output
    scores_raw = outputs['instances'].to('cpu')
    scores_raw = scores_raw.get_fields()
    scores_raw = scores_raw['scores'].numpy()

    #extraer bboxes de etiquetas
    instp_list = []
    for cord in d['annotations']:
      instp = list(cord['bbox'])
      instp_list.append(instp)

    #agrupar bboxes de predicciones con su score
    ke = zip(bbox_raw, scores_raw)
    kel = list(ke)
    instp = instp_list

    #False negative
    if not kel:
        kel = list(instp)
        #graficar etiqueta como prediccion sin bbox
        for i in kel:
            plot_no_detecction(i, path_image, height, width, fnp_crops, fn)

        #graficar etiqueta
        for i in instp:
            plot_label(i, path_image, height, width, fnr_crops)

    #Si no hay etiqueta: Graficar imagen de prediccion como etiqueta
    #Falso positivo
    elif not instp:
        instp = bbox_raw
        for i in instp:
            plot_no_label(i, path_image, height, width, fpr_crops, fp)
      
      #graficar prediccion
        for i, score in kel: 
            plot_detecction(i, score, path_image, height, width, fpp_crops)

    #Cuando hay prediccion y etiqueta: graficar prediccion y etiquetas emparejadas
    #true positive
    else:
      new_raw = []
      new_instp = []
      sep = []
      ser = []
      sepfp = []
      serfp = []
      #Se asigna etiqueta a cada prediccion en caso de iou > 0.2 y se guarda en listas new raw y new instp si no se almacenan en listas sep y ser
      for bbox in bbox_raw:
        bboxtp = None
        max_iou = -1
        max_index = -1
        for i, inst in enumerate(instp):
          iou = bb_iou(bbox, inst)
          if iou > 0.2:
            if iou > max_iou:
                max_iou = iou
                max_index = i
        if max_index >= 0:
          bboxtp = bbox
          new_raw.append(bbox)
          new_instp.append(instp[max_index])
        if bboxtp == None:
              sep.append(bbox)
              ser.append(bbox)
      
      for inst in instp:
          no_fp = None
          for bbox in bbox_raw:
              iou = bb_iou(inst, bbox)
              if iou > 0.2:
                  no_fp = 1
          if no_fp == None:
              sepfp.append(inst)
              serfp.append(inst)

      #Prediccion y etiqueta emparejada iou > 0.2   
      #TRUE POSITIVE IOU
      if len(new_raw) > 0:
        nke = zip(new_raw, scores_raw)
        nkel = list(nke)
        #Graficar prediccion
        for i, score in nkel:
            plot_detecction(i, score, path_image, height, width, listp)
            tp.append(1)

        #Graficar etiqueta
        for i in new_instp:
            plot_label(i, path_image, height, width, listr)

      #Cuando no se encuentra pareja: graficar deteccion como real cuando iou < 0.2
      #FALSE POSITIVE IOU
      if len(sep) > 0:
        seke = zip(sep, scores_raw)
        sekel = list(seke)
        #graficar prediccion
        for i, score in sekel: 
            plot_detecction(i, score, path_image, height, width, fpp_crops)

        #Graficar etiqueta
        for i in ser:
            plot_no_label(i, path_image, height, width, fpr_crops, fp)

      #FALSE NEGATIVE IOU
      if len(sepfp) > 0:

          #graficar deteccion
        for i in sepfp:
            plot_no_detecction(i, path_image, height, width, fnp_crops, fn)

          #graficar etiqueta
        for i in serfp:
            plot_label(i, path_image, height, width, fnr_crops)

listf = [item for sublist in zip(listp, listr) for item in sublist]
fp_crops = [item for sublist in zip(fpp_crops, fpr_crops) for item in sublist]
fn_crops = [item for sublist in zip(fnp_crops, fnr_crops) for item in sublist]

print("False Negative:", len(fn))
print("False Positive:", len(fp))
print("True Positive:", len(tp))
print('len(listf):', len(listf))

#Dibujar mosaicos
os.chdir(path_result)

#Dibujar Verdaderos Positivos
titulo = 'True Positives'
plot_mosaics (titulo, listf, dataset)

#Dibujar Falsos Negativos
titulo = 'False Negatives'
plot_mosaics (titulo, fn_crops, dataset)

#Dibujar Falsos Positivos
titulo = 'False Positives'
plot_mosaics (titulo, fp_crops, dataset)