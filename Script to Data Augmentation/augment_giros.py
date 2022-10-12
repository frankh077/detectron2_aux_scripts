from detectron2.data import transforms as T
import cv2
import json
import numpy as np
from pprint import pprint
import copy
import os

input_path = '/path/to/folder/with/images' 
output_path = r'/path/to/folder/where/images_are_saved'
json_file = 'file/json/'

idx = 213
f = open(json_file)
json_data = json.load(f) 

augs = {16:T.AugmentationList([T.RandomRotation(angle=16, interp=None)]),
        12:T.AugmentationList([T.RandomRotation(angle=12, interp=None)]),
        8:T.AugmentationList([T.RandomRotation(angle=8, interp=None)]),
        4:T.AugmentationList([T.RandomRotation(angle=4, interp=None)]),
        356:T.AugmentationList([T.RandomRotation(angle=356, interp=None)]),
        352:T.AugmentationList([T.RandomRotation(angle=352, interp=None)]),
        348:T.AugmentationList([T.RandomRotation(angle=348, interp=None)]),
        344:T.AugmentationList([T.RandomRotation(angle=344, interp=None)])
        }

def get_poly(ans):
    an_seg = []
    for an in ans:
        an = np.array(an['segmentation'])
        an_formated = np.reshape(an,(round(len(an[0])/2),2))
        an_seg.append(an_formated)
    return an_seg
    
def generate_img(img, deg, img_name):
    new_img_name = 'rotated_' + str(deg) + '_'+img_name
    img_path = os.path.join(output_path, new_img_name)
    return new_img_name


def compute_bbox(poly_int32):
    poly_x = []
    poly_y = []
    for pair in poly_int32:
        poly_x.append(int(pair[0]))
        poly_y.append(int(pair[1]))
    
    return [int(np.min(poly_x)), int(np.min(poly_y)), int(np.max(poly_x)), int(np.max(poly_y))]

def add_label(deg, img_name, dataset, folder, polygons, shape):
    global idx,json_data
    annotations = []
    new_annotations = {'iscrowd': 0,
                       'segmentation': [],
                       'bbox': [],
                       'bbox_mode':0,
                       'category_id':0}

    new_dict = {'file_name': img_name,
                'image_id': idx,
                'height':shape[0],
                'width':shape[1],
                'annotations': []}
    idx += 1
    for poly in polygons:
        poly_int32 = (poly.astype(int))
        poly_flatten = poly_int32.flatten()
        bbox = compute_bbox(poly_int32)
        new_annotations = {'iscrowd': 0,
                       'segmentation': [poly_flatten.tolist()],
                       'bbox': bbox,
                       'bbox_mode':0,
                       'category_id':0}
        annotations.append(new_annotations)
    new_dict['annotations'] = annotations
    json_data[folder].append(new_dict)

    
    
def augment_data(label, folder):
    #load image
    augmentations = []
    img_path = os.path.join(input_path, label['file_name'])
    img_data = cv2.imread(img_path)
    #Extract mask
    mask = get_poly(label['annotations'])
    for k,v in augs.items():
        input = T.AugInput(img_data)
        transform = v(input)
        img_tran = input.image
        polygons_transformed = transform.apply_polygons(mask)
        new_img_name = generate_img(img_tran, k, label['file_name'])
        add_label(k,new_img_name,label['file_name'], folder,polygons_transformed, img_tran.shape)
    #move original image
    output_img_path = os.path.join(output_path, img_path.split('/')[-1])
    cv2.imwrite(output_img_path, img_data)

    with open(output_path + '/new_json.json','w') as outfile:
        json.dump(json_data, outfile)
        

json_data_ori = copy.deepcopy(json_data)
for folder in json_data_ori:
    for label in json_data_ori[folder]:
        print(f"label:{label['file_name']}")
        aug_data = augment_data(label,folder) 
