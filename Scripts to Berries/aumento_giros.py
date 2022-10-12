from detectron2.data import transforms as T
import cv2
import json
import numpy as np
from pprint import pprint
import copy
import os

input_path = 'consolidado_1'
output_path = 'consolidado_1_giros3'
json_file = 'consolidado_1/dataset.json'

idx = 213
f = open(json_file)
json_data = json.load(f)  # cargar las etiquetas en un dicccionario

# definir los giros que se requieren, key == angle (46,46).
augs = {46: T.AugmentationList([T.RandomRotation(angle=46, interp=None)]),
        44: T.AugmentationList([T.RandomRotation(angle=44, interp=None)]),
        42: T.AugmentationList([T.RandomRotation(angle=42, interp=None)]),
        40: T.AugmentationList([T.RandomRotation(angle=40, interp=None)]),
        38: T.AugmentationList([T.RandomRotation(angle=38, interp=None)]),
        36: T.AugmentationList([T.RandomRotation(angle=36, interp=None)]),
        34: T.AugmentationList([T.RandomRotation(angle=34, interp=None)]),
        32: T.AugmentationList([T.RandomRotation(angle=32, interp=None)]),
        30: T.AugmentationList([T.RandomRotation(angle=30, interp=None)]),
        28: T.AugmentationList([T.RandomRotation(angle=28, interp=None)]),
        26: T.AugmentationList([T.RandomRotation(angle=26, interp=None)]),
        24: T.AugmentationList([T.RandomRotation(angle=24, interp=None)]),
        22: T.AugmentationList([T.RandomRotation(angle=22, interp=None)]),
        20: T.AugmentationList([T.RandomRotation(angle=20, interp=None)]),
        18: T.AugmentationList([T.RandomRotation(angle=18, interp=None)]),
        16: T.AugmentationList([T.RandomRotation(angle=16, interp=None)]),
        14: T.AugmentationList([T.RandomRotation(angle=14, interp=None)]),
        12: T.AugmentationList([T.RandomRotation(angle=12, interp=None)]),
        10: T.AugmentationList([T.RandomRotation(angle=10, interp=None)]),
        8: T.AugmentationList([T.RandomRotation(angle=8, interp=None)]),
        6: T.AugmentationList([T.RandomRotation(angle=6, interp=None)]),
        4: T.AugmentationList([T.RandomRotation(angle=4, interp=None)]),
        2: T.AugmentationList([T.RandomRotation(angle=2, interp=None)]),
        358: T.AugmentationList([T.RandomRotation(angle=358, interp=None)]),
        356: T.AugmentationList([T.RandomRotation(angle=356, interp=None)]),
        354: T.AugmentationList([T.RandomRotation(angle=354, interp=None)]),
        352: T.AugmentationList([T.RandomRotation(angle=352, interp=None)]),
        350: T.AugmentationList([T.RandomRotation(angle=350, interp=None)]),
        348: T.AugmentationList([T.RandomRotation(angle=348, interp=None)]),
        346: T.AugmentationList([T.RandomRotation(angle=346, interp=None)]),
        344: T.AugmentationList([T.RandomRotation(angle=344, interp=None)]),
        342: T.AugmentationList([T.RandomRotation(angle=342, interp=None)]),
        340: T.AugmentationList([T.RandomRotation(angle=340, interp=None)]),
        338: T.AugmentationList([T.RandomRotation(angle=338, interp=None)]),
        336: T.AugmentationList([T.RandomRotation(angle=336, interp=None)]),
        334: T.AugmentationList([T.RandomRotation(angle=334, interp=None)]),
        332: T.AugmentationList([T.RandomRotation(angle=332, interp=None)]),
        330: T.AugmentationList([T.RandomRotation(angle=330, interp=None)]),
        328: T.AugmentationList([T.RandomRotation(angle=328, interp=None)]),
        326: T.AugmentationList([T.RandomRotation(angle=326, interp=None)]),
        324: T.AugmentationList([T.RandomRotation(angle=324, interp=None)]),
        322: T.AugmentationList([T.RandomRotation(angle=322, interp=None)]),
        320: T.AugmentationList([T.RandomRotation(angle=320, interp=None)]),
        318: T.AugmentationList([T.RandomRotation(angle=318, interp=None)]),
        316: T.AugmentationList([T.RandomRotation(angle=316, interp=None)])
        }


def get_poly(ans):
    an_seg = []
    for an in ans:
        an = np.array(an['segmentation'])
        an_formated = np.reshape(an, (round(len(an[0])/2), 2))
        an_seg.append(an_formated)
    return an_seg


def generate_img(img, deg, img_name):
    new_img_name = 'rotated_' + str(deg) + '_'+img_name
    img_path = os.path.join(output_path, new_img_name)
    print(f'saving: {img_path}')
    cv2.imwrite(img_path, img)
    return new_img_name


def compute_bbox(poly_int32):
    poly_x = []
    poly_y = []
    for pair in poly_int32:
        poly_x.append(int(pair[0]))
        poly_y.append(int(pair[1]))

    return [int(np.min(poly_x)), int(np.min(poly_y)), int(np.max(poly_x)), int(np.max(poly_y))]


def add_label(deg, img_name, dataset, folder, polygons, shape):
    global idx, json_data
    annotations = []
    new_annotations = {'iscrowd': 0,
                       'segmentation': [],
                       'bbox': [],
                       'bbox_mode': 0,
                       'category_id': 0}

    new_dict = {'file_name': img_name,
                'image_id': idx,
                'height': shape[0],
                'width': shape[1],
                'annotations': []}
    idx += 1
    for poly in polygons:
        poly_int32 = (poly.astype(int))
        poly_flatten = poly_int32.flatten()
        bbox = compute_bbox(poly_int32)
        new_annotations = {'iscrowd': 0,
                           'segmentation': [poly_flatten.tolist()],
                           'bbox': bbox,
                           'bbox_mode': 0,
                           'category_id': 0}
        annotations.append(new_annotations)
    new_dict['annotations'] = annotations
    json_data[folder].append(new_dict)


def augment_data(label, folder):
    augmentations = []
    img_path = os.path.join(input_path, label['file_name'])
    img_data = cv2.imread(img_path)
    mask = get_poly(label['annotations'])
    for k, v in augs.items():
        input = T.AugInput(img_data)
        transform = v(input)
        img_tran = input.image
        polygons_transformed = transform.apply_polygons(mask)
        new_img_name = generate_img(img_tran, k, label['file_name'])
        add_label(k, new_img_name, label['file_name'],
                  folder, polygons_transformed, img_tran.shape)
    # move original image
    output_img_path = os.path.join(output_path, img_path.split('/')[-1])
    cv2.imwrite(output_img_path, img_data)

    with open(output_path + '/new_json.json', 'w') as outfile:
        json.dump(json_data, outfile)
        # outfile.write(json_obj)

# image_label
# image_label = {'file_name': None, 'image_id': None, ''}


#folders = ['test','train']
json_data_ori = copy.deepcopy(json_data)
for folder in json_data_ori:
    # print(f'el:{folder}')
    for label in json_data_ori[folder]:
        print(f"label:{label['file_name']}")
        aug_data = augment_data(label, folder)
