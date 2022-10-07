from detectron2.data import transforms as T
import cv2
import json
import numpy as np
from pprint import pprint
import copy
import os

input_path = 'consolidado_1'
output_path = 'consolidado_1_giros2'
json_file = 'consolidado_1/dataset.json'

idx = 213
f = open(json_file)
json_data = json.load(f) #cargar las etiquetas en un dicccionario

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
        #print(f"an['segmentation']: {an}")
        an_formated = np.reshape(an,(round(len(an[0])/2),2))
        an_seg.append(an_formated)
    return an_seg
    
def generate_img(img, deg, img_name):
    new_img_name = 'rotated_' + str(deg) + '_'+img_name
    img_path = os.path.join(output_path, new_img_name)
    #print(f'saving: {img_path}')
    #cv2.imwrite(img_path, img)
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
        #poly_int32 = np.int8(poly.astype(int))
        ##print(f'poly.astype(int): {poly.astype(int)}')
        poly_int32 = (poly.astype(int))
        ##print(f'poly_int32: {poly_int32}')
        #print(f'poly_int32[0]: {poly_int32[0][0]}')
        #print(f'poly_int32[1]: {poly_int32[0][1]}')
        poly_flatten = poly_int32.flatten()
        #poly_flatten = np.int8(poly.astype(int)).flatten()
        bbox = compute_bbox(poly_int32)
        #print(f'poly_flatten: {poly_flatten}')
        new_annotations = {'iscrowd': 0,
                       'segmentation': [poly_flatten.tolist()],
                       'bbox': bbox,
                       'bbox_mode':0,
                       'category_id':0}
        #new_annotations['segmentation'] = [poly_flatten.tolist()]
        #new_annotations['bbox'] = bbox
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
        #img_tran = None
        polygons_transformed = transform.apply_polygons(mask)
        new_img_name = generate_img(img_tran, k, label['file_name'])
        add_label(k,new_img_name,label['file_name'], folder,polygons_transformed, img_tran.shape)
    #move original image
    output_img_path = os.path.join(output_path, img_path.split('/')[-1])
    cv2.imwrite(output_img_path, img_data)
    #json_obj = json.dumps(json_data) #write new jsonfile

              
  #print(f'{[folder]} -- new_label: {img_name}')

    with open(output_path + '/new_json.json','w') as outfile:
        json.dump(json_data, outfile)
        #outfile.write(json_obj)
        
#image_label
#image_label = {'file_name': None, 'image_id': None, ''}


#folders = ['test','train']
json_data_ori = copy.deepcopy(json_data)
for folder in json_data_ori:
    #print(f'el:{folder}')
    for label in json_data_ori[folder]:
        print(f"label:{label['file_name']}")
        aug_data = augment_data(label,folder) 
        #print(el['file_name'])
        #print(el)
        #for k,v in el.items():
        #    print(f'k: {k} -- v: {v}\n')
        #print(f'el:{el}\n')