from detectron2.data import transforms as T
import cv2
import json
import numpy as np
import copy
import os
import glob

input_path = 'consolidado_v2_evidentes_aumentado_dropout_color_consolidado'
output_path = 'consolidado_v2_evidentes_aumentado_dropout_color_consolidado_rotaciones'
json_file = 'consolidado_1/dataset.json'
prefix_name = 'A_rotation'


augs = {16:T.AugmentationList([T.RandomRotation(angle=45, interp=None)]),
        12:T.AugmentationList([T.RandomRotation(angle=40, interp=None)]),
        8:T.AugmentationList([T.RandomRotation(angle=35, interp=None)]),
        4:T.AugmentationList([T.RandomRotation(angle=30, interp=None)]),
        356:T.AugmentationList([T.RandomRotation(angle=25, interp=None)]),
        352:T.AugmentationList([T.RandomRotation(angle=20, interp=None)]),
        348:T.AugmentationList([T.RandomRotation(angle=15, interp=None)]),
        344:T.AugmentationList([T.RandomRotation(angle=10, interp=None)])
        }

def get_poly(ans):
    an_seg = []
    for an in ans:
        an = np.array(an['segmentation'])
        an_formated = np.reshape(an,(round(len(an[0])/2),2))
        an_seg.append(an_formated)
    return an_seg

def generate_img(img, deg, img_name, folder):
    new_img_name = 'rotated_' + str(deg) + '_'+img_name
    img_path = os.path.join(output_path, folder,new_img_name)
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

def mask_to_label(mask, name, size):
    print(f'mask shape: {mask}')
    shapes = []
    baselabel = {"version": "4.2.9", 
                 "flags": {}, 
                 "shapes": [], 
                 "imagePath": name, 
                 "imageData": "test", 
                 "imageHeight": size[0], 
                 "imageWidth": size[1]}
                 
    #shapes es una lista de diccionarios
    for c in mask:
        clist = c.tolist()
        clist = [[int(x[0]),int(x[1])] for x in clist]
        if len(clist) > 2:
            shapes.append({"label": "disease", "points": clist, "group_id": 'null', "shape_type": "polygon"})
    
    baselabel['shapes'] = shapes
    return baselabel
    
def get_poly(ans):
    an_seg = []
    for an in ans:
        an = np.array(an['points'])
        an_seg.append(an)
    return an_seg
    
def augmentation_rotation(transform,img_path, label_path, angle, folder):
    img_name = img_path.split('/')[-1].split('.')[0]
    folder = img_path.split('/')[-2]
    new_name = prefix_name + '_'+str(angle)+'_'+img_name
    #Read the image 
    img_data = cv2.imread(img_path)
    label = json.load(open(label_path))
    mask = get_poly(label['shapes'])

    #load image
    augmentations = []

    #Extract mask
    input = T.AugInput(img_data)
    transform = transform(input)
    img_tran = input.image
    polygons_transformed = transform.apply_polygons(mask)
    new_img_name = generate_img(img_tran, angle, label['imagePath'], folder)
    new_label_name = new_img_name.split('.')[0] + '.json'
    new_label=mask_to_label(polygons_transformed,new_img_name, img_tran.shape)
    with open(os.path.join(output_path,folder, new_label_name),'w') as outfile:
        json.dump(new_label, outfile)
    
def copy_original(image):
    pass 

def main():

    file_error = 0
    transform = {10:T.AugmentationList([T.RandomRotation(angle=10, interp=None)]),
                 8:T.AugmentationList([T.RandomRotation(angle=8, interp=None)]),
                 6:T.AugmentationList([T.RandomRotation(angle=6, interp=None)]),
                 4:T.AugmentationList([T.RandomRotation(angle=4, interp=None)]),
                 2:T.AugmentationList([T.RandomRotation(angle=2, interp=None)]),
                 358:T.AugmentationList([T.RandomRotation(angle=358, interp=None)]),
                 356:T.AugmentationList([T.RandomRotation(angle=356, interp=None)]),
                 354:T.AugmentationList([T.RandomRotation(angle=354, interp=None)]),
                 352:T.AugmentationList([T.RandomRotation(angle=352, interp=None)]),
                 350:T.AugmentationList([T.RandomRotation(angle=350, interp=None)])
                 }
    ##Get files
    files = {'train': glob.glob(os.path.join(input_path, 'train') + "/*"),'test':glob.glob(os.path.join(input_path, 'test') + "/*")}
    files_name = {'train': [x.split('/')[-1] for x in files['train']],'test': [x.split('/')[-1] for x in files['test']]}
     
    for k in files.keys():
        for idx,file in enumerate(files[k]):
            print(f'{k}) {idx+1}/{len(files[k])}')
            #Select only images
            file_ext = file.split('.')[-1]
            if file_ext in ['jpg', 'JPG', 'PNG', 'png']:
                img_name = file.split('.')[0]
                #Check if the image has a label
                json_path = img_name + '.json'
                if os.path.exists(json_path):
                    for t in transform.keys():
                        #If exists proceed with augmentation
                        print(f'Input: {file}, transform: {transform[t]}')
                        augmentation_rotation(transform[t], file, json_path, t, k)
                        copy_original(file)
                else:
                    file_error +=1
                    print(f'Error: {img_name + ".json"} does not exists')
    print(f'No se encontraron {file_error} imagenes')

if __name__ == '__main__':
  main()