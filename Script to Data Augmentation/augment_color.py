from itertools import combinations, permutations
from itertools import permutations
import cv2
import numpy as np
import argparse
import os
import glob 
import json
from os.path import exists

# Read image given by user
parser = argparse.ArgumentParser(description='Code for Changing the contrast and brightness for a dataset.')
parser.add_argument('--input', help='Path to images folder.', default='.')

def get_prefix(param, value, old_path):
    img_path_base = old_path.split('/')[:-1]
    img_path_tail = ('-').join(((old_path.split('/')[-1]).split('.'))[:-1]) + '.png' ##Correcion extension
    print(f'img_path_tail: {img_path_tail}')
    signo = None

    if value > 1:
        signo = '+'
    elif value < 1:
        signo = '-'
    
    img_path_nuevo = signo+param+'_'+img_path_tail

    ##corregir extension
    img_path = '/'.join(img_path_base) + '/'+img_path_nuevo

    print(f'new path: {img_path}, new name : {img_path_nuevo}')
    
    return img_path,img_path_nuevo

def correct_json(json_name, new_img_name):
    ##Corrige el nombre de la imagen a la que apunta agregando el prefijo del aumento y tambien el nombre del json
    f = open(json_name)
    new_json_base = json_name.split('/')[:-1]
    new_json_tail = json_name.split('/')[-1].split('.')[0] + '.json'
    new_json_path = ('/').join(new_json_base) + '/' + new_json_tail
    new_file_name = ('-').join(new_img_name.split('.')[:-1]) + '.json'
    print(f'**prev_file_name: {json_name}')
    print(f'**new_file_name: {new_file_name}')
    data = json.load(f)
    data['imagePath'] = new_img_name
    new_path2 = os.path.join(("/").join(new_json_base),new_file_name)
    os.remove(json_name)
    print(f'old_path: {json_name}, new_path: {new_path2}')
    print(f'**************escribiendo json: {new_json_path}')
    with open(new_path2, "w") as outfile:
        json.dump(data, outfile)

def aplica_brillo(input, brightness):
    print(f'In brillo')
    carpetas = ['test', 'train']
    for carpeta in carpetas:
        path_1 = input + '/' + carpeta + '/'
        print(f'path_1: {path_1}')
        files = list(next(os.walk(path_1)))
        for file in files[2]:
            print(f'file: {file}')
            if file:
                if file.find('jpg') != -1:
                    img_path = path_1 + file
                    print(f'aplicando ajuste brillo a: {img_path}')
                    input_img = cv2.imread(img_path, 1)
                    new_image = np.zeros(input_img.shape, input_img.dtype)
                    new_image = cv2.convertScaleAbs(input_img, alpha=1, beta=brightness)
                    ##Agregar prefijo aumento
                    new_img_path,  new_img_name= get_prefix('brillo', brightness, img_path)
                    print(f'**************escribiendo imagen: {new_img_path}')
                    os.remove(img_path)
                    cv2.imwrite(new_img_path, new_image)

                    ##corregir json
                    json_path = ('.').join(img_path.split('.')[:-1]) + '.json'
                    correct_json(json_path, new_img_name)

def aplica_contraste(input, contraste):
    print(f'En contraste')
    carpetas = ['test', 'train']
    for carpeta in carpetas:
        path_1 = input + '/' + carpeta + '/'
        files = list(next(os.walk(path_1)))
        for file in files[2]:
            if file:
                if file.find('.jpg') != -1:
                    img_path = path_1 + file
                    input_img = cv2.imread(img_path, 1)
                    print(f'aplicando ajuste contraste a: {img_path}')
                    new_image = np.zeros(input_img.shape, input_img.dtype)
                    new_image = cv2.convertScaleAbs(input_img, alpha=contraste, beta=0)
                    new_img_path,  new_img_name= get_prefix('contraste', contraste, img_path)
                    print(f'**************escribiendo imagen: {new_img_path}')
                    cv2.imwrite(new_img_path, new_image)

                    ##corregir json
                    json_path = ('.').join(img_path.split('.')[:-1]) + '.json'
                    correct_json(json_path, new_img_name)

def aplica_saturacion(input, saturation):
    print(f'En saturacion')
    carpetas = ['test', 'train']
    for carpeta in carpetas:
        path_1 = input + '/' + carpeta + '/'
        print(f'path_1: {path_1}')
        files = list(next(os.walk(path_1)))
        for file in files[2]:
            print(f'file: {file}')
            if file:
                if file.find('.jpg') != -1:
                    img_path = path_1 + file
                    input_img = cv2.imread(img_path, 1)
                    new_image = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV).astype("float32")
                    #saturation = saturation / 10
                    (h, s, v) = cv2.split(new_image)
                    s = s*saturation
                    s = np.clip(s,0,255)
                    new_image = cv2.merge([h,s,v])   
                    new_image = cv2.cvtColor(new_image.astype("uint8"), cv2.COLOR_HSV2BGR)
                    new_img_path,  new_img_name= get_prefix('saturacion', saturation, img_path)
                    print(f'**************escribiendo imagen: {new_img_path}')
                    cv2.imwrite(new_img_path, new_image)

                    ##corregir json
                    json_path = ('.').join(img_path.split('.')[:-1]) + '.json'
                    correct_json(json_path, new_img_name)

def aplica_gamma(input, gamma):
    print(f'En gamma')
    carpetas = ['test', 'train']
    for carpeta in carpetas:
        path_1 = input + '/' + carpeta + '/'
        print(f'path_1: {path_1}')
        files = list(next(os.walk(path_1)))
        for file in files[2]:
            if file:
                if file.find('.jpg') != -1:
                    img_path = path_1 + file
                    input_img = cv2.imread(img_path, 1)
                    #gamma = gamma/10
                    lookUpTable = np.empty((1,256), np.uint8)
                    for i in range(256):
                        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
            
                    new_image = cv2.LUT(input_img, lookUpTable)
                    new_img_path,  new_img_name= get_prefix('gamma', gamma, img_path)
                    print(f'**************escribiendo imagen: {new_img_path}')
                    cv2.imwrite(new_img_path, new_image)

                    ##corregir json
                    json_path = ('.').join(img_path.split('.')[:-1]) + '.json'
                    correct_json(json_path, new_img_name)
    
def apply_image_adjust(input):
    brillo = [40,-40]
    contraste = [1.4,0.6]
    saturacion = [2.4,0.7]
    gamma = [2.0,0.6]

    #Obtener los nombres de las carpetas
    path = input+'/'
    carpetas = list(next(os.walk(path))[1])
    carpetas = ['+brillo','-brillo','+contraste','-contraste','+gamma','-gamma','+saturacion','-saturacion']

    print(f'carpetas: {carpetas}')
    for carpeta in carpetas:
        if carpeta[1:] == 'brillo':
            path = os.path.join(input, carpeta)
            if carpeta[0] == '+':
                aplica_brillo(path, brillo[0])
            else:
                aplica_brillo(path, brillo[1])   
        if carpeta[1:] == 'contraste':
            path = os.path.join(input, carpeta)
            if carpeta[0] == '+':
                aplica_contraste(path, contraste[0])
            else:
                aplica_contraste(path, contraste[1])
        if carpeta[1:] == 'saturacion':
            path = os.path.join(input, carpeta)
            if carpeta[0] == '+':
                aplica_saturacion(path, saturacion[0])
            else:
                aplica_saturacion(path, saturacion[1])   
        if carpeta[1:] == 'gamma':
            path = os.path.join(input, carpeta)
            if carpeta[0] == '+':
                aplica_gamma(path, gamma[0])
            else:
                aplica_gamma(path, gamma[1])
           
def funcBrightContrast(bright=0):
    bright = cv2.getTrackbarPos('bright', 'Test')
    contrast = cv2.getTrackbarPos('contrast', 'Test')
    saturation = cv2.getTrackbarPos('saturation', 'Test')
    gamma = cv2.getTrackbarPos('gamma', 'Test')
    effect = apply_brightness_contrast(img,bright,contrast, saturation, gamma)
    cv2.imshow('Effect', effect)


def apply_brightness_contrast(input_img, brightness, contrast, saturation, gamma):
    #brillo y contraste
    brightness = brightness - 100
    contrast = (contrast + 100) / 100
    new_image = np.zeros(input_img.shape, input_img.dtype)
    new_image = cv2.convertScaleAbs(input_img, alpha=contrast, beta=brightness)
    #saturacion
    new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2HSV).astype("float32")
    saturation = saturation / 10
    (h, s, v) = cv2.split(new_image)
    s = s*saturation
    s = np.clip(s,0,255)
    new_image = cv2.merge([h,s,v])   
    new_image = cv2.cvtColor(new_image.astype("uint8"), cv2.COLOR_HSV2BGR)
    
    #correccion gamma
    gamma = gamma/10
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    
    new_image = cv2.LUT(new_image, lookUpTable)
    cv2.putText(new_image,'B:{},C:{},S:{},G:{}'.format(brightness,contrast,saturation, gamma),(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return new_image

if __name__ == '__main__':
    args = parser.parse_args()
    path = args.input
    print(f'input: {path}')
    apply_image_adjust(path)

cv2.waitKey(0)