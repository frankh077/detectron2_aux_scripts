# Evaluation

For the evaluation there are 3 important codes:

- [IOU Validation](https://github.com/frankh077/detectron2_aux_scripts/blob/main/Script%20to%20Evaluation/iou_validation.py)

- [Inference](https://github.com/frankh077/detectron2_aux_scripts/blob/main/Script%20to%20Evaluation/inference.py)

- [Labels Evaluation](https://github.com/frankh077/detectron2_aux_scripts/blob/main/Script%20to%20Evaluation/labels_evaluation.py)

The evaluation criterion used was the Intersection Over Union [(IOU)](https://github.com/frankh077/detectron2_aux_scripts/blob/main/Script%20to%20Evaluation/iou_validation.py), which consists of the overlap between two bboxes, the larger the region of overlap, the higher the IOU. An IOU score greater than 0.2 between a detection and a label is considered a **true positive**, otherwise it is a **false positive** and labels without detection were considered **false negatives**.
The [inference](https://github.com/frankh077/detectron2_aux_scripts/blob/main/Script%20to%20Evaluation/inference.py) code perform the inferences and plot the detections images.
Finally, the [Labels Evaluation](https://github.com/frankh077/detectron2_aux_scripts/blob/main/Script%20to%20Evaluation/labels_evaluation.py) code has some functions, such as:

- make_pred
- get_annotation
- get_bboxFromAnnotation
- get_iou
- bbox_in_annotation
- writeImage
- verify_annotation
- add_labels
- draw_mask

But in this case, just used the **make_pred** function which generates labels from the images inference, and also draws the bounding boxes with their respective annotations.

# Images

Mosaics of the evaluation through IOU of the Bunches detection model are shown below.

<!-- <p float="middle" >
  <img src="https://github.com/frankh077/detectron2_aux_scripts/blob/main/pictures/test-False%20Negatives-2.png" width="300%"  />
</p> -->

| ![](https://github.com/frankh077/detectron2_aux_scripts/blob/main/pictures/test-False%20Negatives-2.png) | 
|:--:| 
| *False Negatives* |

<!-- <p float="middle" >
  <img  src="https://github.com/frankh077/detectron2_aux_scripts/blob/main/pictures/test-False%20Positives-1.png" width="300%" />
</p> -->

| ![](https://github.com/frankh077/detectron2_aux_scripts/blob/main/pictures/test-False%20Positives-1.png) | 
|:--:| 
| *False Positives* |

<!-- <p float="middle" >
  <img  src="https://github.com/frankh077/detectron2_aux_scripts/blob/main/pictures/test-True%20Positives-0.png" width="300%" />
</p> -->

| ![](https://github.com/frankh077/detectron2_aux_scripts/blob/main/pictures/test-True%20Positives-0.png) | 
|:--:| 
| *True Positives* |
