# detectron2_aux_scripts

In this repository you will find all the auxiliary  scripts that was used with detectron2  for berries detection in bunches of grapes images, allows the detection of circular, oval and elongated grapes shapes.

<p float="middle" >
  <img src="https://github.com/frankh077/detectron2_aux_scripts/blob/main/pictures/berries_alarg.jpg"  width="400" height="450" />
   <img src="https://github.com/frankh077/detectron2_aux_scripts/blob/main/pictures/bunch.jpg" width="550" height="450" />

</p>

## Table of Contents

1. [Scrip to Data Augmentation](https://github.com/frankh077/detectron2_aux_scripts/tree/main/Script%20to%20Data%20Augmentation)
2. [Script to Data Evaluation](https://github.com/frankh077/detectron2_aux_scripts/tree/main/Script%20to%20Evaluation)
3. [Script to Data Processing Labels](https://github.com/frankh077/detectron2_aux_scripts/tree/main/Script%20to%20Processing%20Labels)
4. [Script to Train](https://github.com/frankh077/detectron2_aux_scripts/tree/main/Script%20to%20Train)

5. [Scripts to Berries](https://github.com/frankh077/detectron2_aux_scripts/tree/main/Scripts%20to%20Berries)

## Installation

These scripts were executed on a computer with the following characteristics:

- OS type: 64-bit Ubuntu/Linux 18.04.6 LTS
- Processor: 11th Gen Intel Core i7-11700 @2.50GHz x 16
- Graphics: NVIDIA GeForce RTX 3060

### Docker commands

Download image

`sudo docker pull eddyerach1/detectron2_banano_uvas:latest`

Build a container from an image

`sudo docker run --gpus all -it -v /home/grapes:/shared  --name detectron2_grapes`
