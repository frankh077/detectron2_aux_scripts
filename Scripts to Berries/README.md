# Scripts to Berries

The following codes were used to count the berries.

- [Inference](https://github.com/frankh077/detectron2_aux_scripts/blob/main/Scripts%20to%20Berries/inference.py)
- [Validation](https://github.com/frankh077/detectron2_aux_scripts/blob/main/Scripts%20to%20Berries/validation.py)
- [Train](https://github.com/frankh077/detectron2_aux_scripts/blob/main/Scripts%20to%20Berries/train.py)
- [Process Labels](https://github.com/frankh077/detectron2_aux_scripts/tree/main/Scripts%20to%20Berries/Script%20to%20Processing%20Labels)

Since there are different types of berries, such as: round, oval and elongated, an enlargement of the dataset with rotated images was performed, for this purpose the following [scripts](https://github.com/frankh077/detectron2_aux_scripts/tree/main/Script%20to%20Data%20Augmentation) were used.

As found a large number of berries per bunch, it was decided to contrast the number of labels vs. the number of detections, since there are labels and detections that overlap. Then, apply [evaluation]() of the model was carried out through the inference of the test dataset. The evaluation code takes as input a json file that is generated from [mask code]() that contains all the dataset labels.
The inference  consist in draw the bounding box  and to calculate detections on images. Finally, to train the model use the this[code](https://github.com/frankh077/detectron2_aux_scripts/blob/main/Scripts%20to%20Berries/train.py)
