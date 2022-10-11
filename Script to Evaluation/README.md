For the evaluation there are 3 important codes:

[Inference](https://github.com/frankh077/detectron2_aux_scripts/blob/main/Script%20to%20Evaluation/inference.py)
[IOU Validation](https://github.com/frankh077/detectron2_aux_scripts/blob/main/Script%20to%20Evaluation/iou_validation.py)
[Labels Evaluation](https://github.com/frankh077/detectron2_aux_scripts/blob/main/Script%20to%20Evaluation/labels_evaluation.py)

The evaluation criterion used was the Intersection Over Union [(IOU)](https://github.com/frankh077/detectron2_aux_scripts/blob/main/Script%20to%20Evaluation/iou_validation.py), which consists of the overlap between two bboxes, the larger the region of overlap, the higher the IOU. An IOU score greater than 0.2 between a detection and a label is considered a true positive, otherwise it is a false positive and labels without detection were considered false negatives.
