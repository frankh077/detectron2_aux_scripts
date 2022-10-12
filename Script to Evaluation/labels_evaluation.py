import detectron2
from detectron2.utils.logger import setup_logger
import numpy as np
import cv2
import matplotlib.pyplot as plt
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import pdb
import os
import json
from detectron2.structures import BoxMode
import numpy
import glob
from detectron2.utils.visualizer import ColorMode
from scipy.spatial import ConvexHull, convex_hull_plot_2d

setup_logger()

weights_path = '/exterior/conteo_plantas/9febp2/9feb2_2/model_final.pth' 
images_path = '/exterior/conteo_plantas/9febp2/9feb2_2/originales'
new_labels = 0

def get_images_from_path(images_path):
    list_files = glob.glob(images_path + '/*')
    list_images = []
    for file in list_files:
        print(f'file: {file}')
        if len(file.split("."))> 1:
            if file.split(".")[1] == 'JPG':
                list_images.append(file)

    print(f'Lista de imagenes en el directorio: {list_images}')
    return list_images

class Predictor:
    def __init__(self, weigths_path):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.weights_path = weigths_path
        print(f'weights_path: {self.weights_path}')
        self.cfg.MODEL.WEIGHTS = self.weights_path
        self.cfg.SOLVER.IMS_PER_BATCH = 2
        self.cfg.SOLVER.BASE_LR = 0.00025
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01
        self.predictor = DefaultPredictor(self.cfg)
    
    def make_pred(self, im, name):
        font = cv2.FONT_HERSHEY_SIMPLEX 
        outputs = self.predictor(im) 
        #Get bboxes from prediction output
        bbox_raw = outputs['instances'].to('cpu') 
        bbox_raw = bbox_raw.get_fields() 
        bbox_raw = bbox_raw['pred_boxes'].tensor.numpy()
        bbox_raw = list(map(numpy.ndarray.tolist, bbox_raw))
        bbox_raw = list(map(lambda x: list(map(int, x)), bbox_raw))

        print(f'bbox_raw: {bbox_raw}')
        #Get scores from prediction output
        scores_raw = outputs['instances'].to('cpu')
        scores_raw = scores_raw.get_fields()
        scores_raw = scores_raw['scores'].numpy()
        
        for bbox, score in zip(bbox_raw, scores_raw): 
            left_top = tuple(bbox[:2])
            right_bottom = tuple(bbox[2:])
            score_height = (bbox[0], bbox[1] - 10) 
            im = cv2.rectangle(im,right_bottom,left_top,(0,0,255),15) 
            im = cv2.putText(im,"{:.2f}".format(score),score_height, font, 2,(0,0,255),5,cv2.LINE_AA)
            cv2.imwrite(name.split('.')[0] + '-inf.JPG' ,im)

    def get_annotation(self, image_path):
        #Funcion que extrae las etiquetas del json de cada imagen.
        #Retorna una lista de etiquetas, en donde cada etiqueta es un grupo de puntos.

        bboxes = []
        json_path = image_path.split(".")[0] + '.json'
        
        with open(json_path) as json_file:
            data = json.load(json_file)
        
        for bbox in data["shapes"]:
            bboxes.append(bbox["points"])

        return bboxes

    def get_bboxFromAnnotation(self,annotations):
        x = []
        y = []
        bboxes = [] 

        for shape in annotations:
            for point in shape:
                x.append(point[0]) 
                y.append(point[1])
            bboxes.append([min(x),min(y),max(x),max(y)])

        return bboxes
    def get_iou(self, bb1, bb2):
        bb1 = {'x1':bb1[0], 'x2':bb1[2], 'y1':bb1[1], 'y2':bb1[3]}
        bb2 = {'x1':bb2[0], 'x2':bb2[2], 'y1':bb2[1], 'y2':bb2[3]}
        assert bb1['x1'] < bb1['x2']
        assert bb1['y1'] < bb1['y2']
        assert bb2['x1'] < bb2['x2']
        assert bb2['y1'] < bb2['y2']

        # determine the coordinates of the intersection rectangle
        x_left = max(bb1['x1'], bb2['x1'])
        y_top = max(bb1['y1'], bb2['y1'])
        x_right = min(bb1['x2'], bb2['x2'])
        y_bottom = min(bb1['y2'], bb2['y2'])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # compute the area of both AABBs
        bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
        bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        assert iou >= 0.0
        assert iou <= 1.0
        print(f'iou: {iou}. bbox:{bb1} annotattion: {bb2}')
        return iou

    def bbox_in_annotation(self, bbox, annotations):
        flag = False
        for annotation in annotations:
            if self.get_iou(annotation,bbox) > 0.2:
                flag = True

        return flag

    def writeImage(self,im,im_path,bbox_fp,annotations):
        font = cv2.FONT_HERSHEY_SIMPLEX
        #First, draw annotations
        for bbox in annotations:
            left_top = tuple(bbox[:2])
            right_bottom = tuple(bbox[2:])
            score_height = (bbox[0], bbox[1] - 10) 
            im = cv2.rectangle(im,right_bottom,left_top,(255,0,0),2)
            im = cv2.putText(im,'label',score_height, font, 0.7,(255,0,0),2,cv2.LINE_AA)

        #Second, draw bbox false positive
        left_top = tuple(bbox_fp[:2])
        right_bottom = tuple(bbox_fp[2:])
        score_height = (bbox_fp[0], bbox_fp[1] - 10)  
        im = cv2.rectangle(im,right_bottom,left_top,(0,0,255),2)
        im = cv2.putText(im,'FP',score_height, font, 0.7,(0,0,255),2,cv2.LINE_AA)

        cv2.imwrite(im_path.split('.')[0].split("/")[-1] + '-inf.JPG' ,im)


    def verify_annotation(self,bboxes,mask,annotations, im, im_path):
        for bbox in bboxes:
            if not self.bbox_in_annotation(bbox, annotations):
                print(f'False positive found: {bbox} not in {annotations}')
                self.writeImage(im,im_path,bbox,annotations)

    def add_labels(self, image_path, polygon_to_add):
        json_path = image_path.split(".")[0] + '.json'
        
        with open(json_path) as json_file:
            data = json.load(json_file)

        new_shape = {"label": "disease", "points": polygon_to_add, "group_id": None, "shape_type": "polygon"}  
        print(f'new_shape: {new_shape}')
        data['shapes'].append(new_shape)

        with open(json_path, 'w') as fp:
            json.dump(data, fp)
    
    def draw_mask(self, image, mask):
        pts = np.array(mask, np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(image,[pts],True,(0,255,255))
        cv2.imwrite('test.JPG',image)
        

    def get_comparacion(self, image_data, image_path):
        bbox, mask = self.get_bbox_mask(image_data)
        annotations = self.get_annotation(image_path)
        bbox_annotation = self.get_bboxFromAnnotation(annotations)
        print(f'bbox: {bbox}, annotations: {bbox_annotation}')
        bbox_falso_positivo = self.verify_annotation(bbox,mask,bbox_annotation, image_data, image_path)

        

    def get_bbox_mask(self,im):
        mask_coordinates = []
        outputs = self.predictor(im)
        #Get bboxes from prediction output
        bbox_raw = outputs['instances'].to('cpu')
        mask = outputs['instances'].pred_masks.to('cpu').numpy()
        num, h, w= mask.shape
        bin_mask= np.zeros((h, w))
    
        for m in mask:
            bin_mask+= m
        counter = 0
        sampler = 10
        for x, row in enumerate(bin_mask):
            for y,column in enumerate(row):
                if bin_mask[x][y] > 0:
                    mask_coordinates.append([y,x])


        hull = ConvexHull(mask_coordinates)
        polygon = []
        for vert in hull.vertices:
            polygon.append(mask_coordinates[vert])
        
        bbox_raw = bbox_raw.get_fields()
        bbox_raw = bbox_raw['pred_boxes'].tensor.numpy()
        bbox_raw = list(map(numpy.ndarray.tolist, bbox_raw))
        bbox_raw = list(map(lambda x: list(map(int, x)), bbox_raw))
      
        return bbox_raw, polygon


    def get_double_instace(self, im_path):
        json_path = im_path.split(".")[0] + '.json'
        
        with open(json_path) as json_file:
            data = json.load(json_file)

        json_file.close()

        if len(data['shapes']) > 1:
            print(f'len > 1: {im_path}')


predictor = Predictor(weights_path)

images = get_images_from_path(images_path)

for image in images:
  predictor.make_pred(cv2.imread(image), image)

