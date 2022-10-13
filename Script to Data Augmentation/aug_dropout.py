# import some common libraries
import albumentations as A
import cv2
import json
import numpy as np
from PIL import Image, ImageDraw
import os
import shutil
import glob


##Augment Params
crop_width = 512 
crop_height = 512

input_folder = 'folder/with/images'
output_folder = 'folder/where/images_are_saved'

prefix_name = 'A-dropout_'
def polys_to_mask(polys, img):
    height, width = img.shape[:2]
    masks2 = np.zeros([height,width])
    
    #draw boundaries
    for poly in polys:
        nparray = np.array(poly['points'])
        masks2 = cv2.fillConvexPoly(masks2, nparray, (255, 255, 255))

    return masks2

def mask_to_label(mask, name):

    shapes = []
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    baselabel = {"version": "4.2.9", 
                 "flags": {}, 
                 "shapes": [], 
                 "imagePath": name+'.jpg', 
                 "imageData": "test", 
                 "imageHeight": 600, 
                 "imageWidth": 800}
                 
    for c in contours:
        clist = c.tolist()
        clist = [x[0] for x in clist]
        if len(clist) > 2:
            shapes.append({"label": "disease", "points": clist, "group_id": 'null', "shape_type": "polygon"})
    
    baselabel['shapes'] = shapes
    return baselabel



def apply_transform(transform, image, masks):
    transformed = transform(image=image, mask=masks)
    transformed_image = transformed['image']
    transformed_mask = transformed['mask']
    return transformed_image, transformed_mask

def write_results(new_img, new_label, new_name, folder):
    output_path = os.path.join(output_folder, folder)
    output_image = os.path.join(output_path, new_name + '.jpg')
    output_label = os.path.join(output_path, new_name + '.json')
    cv2.imwrite(output_image, new_img)
    with open(output_label, "w") as outfile:
        json.dump(new_label, outfile)

def augmentation_GridDropout(transform,img_path, label_path):
    img_name = img_path.split('/')[-1].split('.')[0]
    folder = img_path.split('/')[-2]
    new_name = prefix_name + img_name
    #Read the image 
    image = cv2.imread(img_path)
    label = json.load(open(label_path))['shapes']
    masks = polys_to_mask(label, image)
    new_im, new_mask = apply_transform(transform, image, masks) #Apply transformation, return new mask and new image
    new_label=mask_to_label(new_mask,new_name) # Convert mask to label
    write_results(new_im, new_label,new_name,folder) #Write the new image and new label

def main():
    file_error = 0
    transform = A.Compose([A.GridDropout(ratio=0.5, unit_size_min=None, unit_size_max=None, 
                  holes_number_x=8, holes_number_y=5, shift_x=0, shift_y=0, 
                  random_offset=True, fill_value=0, mask_fill_value=0, always_apply=True, p=1)])

    ##Get files
    files = {'train': glob.glob(os.path.join(input_folder, 'train') + "/*"),'test':glob.glob(os.path.join(input_folder, 'test') + "/*")}
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
                    #If exists proceed with augmentation
                    augmentation_GridDropout(transform, file, json_path)
                else:
                    file_error +=1
                    print(f'Error: {img_name + ".json"} does not exists')
    print(f'No se encontraron {file_error} imagenes')


 
if __name__ == '__main__':
  main()



