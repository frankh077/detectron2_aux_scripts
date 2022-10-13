
# Data Augmentations

Due to the complexity of the detection, a robust model is required. For this reason, an extensive augmentation of the consolidated dataset was performed.

## Augmentations

- [Grid Dropout (Albumentations)](https://github.com/frankh077/detectron2_aux_scripts/blob/main/Script%20to%20Data%20Augmentation/aug_dropout.py)
- [Colors (Detectron2)](https://github.com/frankh077/detectron2_aux_scripts/blob/main/Script%20to%20Data%20Augmentation/augment_color.py)
- [Rotations (Detectron2)](https://github.com/frankh077/detectron2_aux_scripts/blob/main/Script%20to%20Data%20Augmentation/aug_rotations.py)

### Dataset augmentation: Grid Dropout

For the Grid Dropout augmentation we used the Albumentations library, because it is not implemented in Detectron2.

### Dataset augmentation: Colors

This code modifies:

- (+/-) brightness
- (+/-) contrast
- (+/-) saturation
- (+/-) gamma
for each images.

### Dataset augmentation: Rotations

The rotations were from 45º to -45º, each 5º a rotation was made for a total of 19 rotations of the dataset.
