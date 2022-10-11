Due to the complexity of the detection, a robust model is required. For this reason, an extensive augmentation of the consolidated dataset was performed.

## Augmentations1

- [Grid Dropout (Albumentations)](#Dropout)
- [Colors (Detectron2)](#Color)
- [Rotations (Detectron2)](#Rotation)

#### Dataset augmentation: Grid Dropout

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
