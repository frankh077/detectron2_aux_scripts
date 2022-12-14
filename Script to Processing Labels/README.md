# Processing Labels

- [Process Labels](https://github.com/frankh077/detectron2_aux_scripts/blob/main/Script%20to%20Processing%20Labels/process_labels.py)
- [Plot Labels](https://github.com/frankh077/detectron2_aux_scripts/blob/main/Script%20to%20Processing%20Labels/plot_labels.py)

Once the labeling was finished, a python code was used to format the label JSON file in such a way that it is compatible with detectron2 for [training](https://github.com/frankh077/detectron2_aux_scripts/blob/main/Script%20to%20Train/train.py).

[Plot Labels](https://github.com/frankh077/detectron2_aux_scripts/blob/main/Script%20to%20Processing%20Labels/plot_labels.py) draws the labels from the json generated from [VGG image annotator](https://www.robots.ox.ac.uk/~vgg/software/via/via.html) tool from The University of Oxford using the polygon region shape tool. And code [process labels](https://github.com/frankh077/detectron2_aux_scripts/blob/main/Script%20to%20Processing%20Labels/process_labels.py) consist split the labels of each image into an individual .json file.
