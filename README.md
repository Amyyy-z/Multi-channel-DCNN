# Multi-channel-DCNN implementation
This repository stores the code for multi-channel CNN architectures implementation, including SIDC, DIDC, and four-channel CNN architectures generated with Xception as base model

VGGs and ResNets are included in this project, other models (i.e., Inception, DenseNet) can be extracted from Keras.models pacakages

CT image segmentation (cropping) tool is also available, so as partial thyroid CT images and labels

## Getting Started
If you are working with ultrasound images, then you can go to the specific multi-channel CNN architectures for implementation.
If you are working with CT images, CT segmentation is suggested to reach better classification performance.

### Pre-requisites for CT segmentation
1. A whole-slide of CT image is required, basic medical knowledges are required to segment the image into left and right sides based on each lobes.
2. Implement segmentation of CT scans through [CT segmentation too.py](https://github.com/Amyyy-z/Multi-channel-DCNN/blob/Multi-channel-CNN/CT%20segmentation%20tool.py).
3. Segmented images will need to be labeled in the meantime.
4. Images will be stored in a new file, and labels will be stored in an Excel sheet.

### Apply SIDC, DIDC, and Four-channel architectures
**Computational environment**: Tensorflow 2.1.0, Python 3.7

**Implementation procedures**:
1. Prepare left-side and right-side images in seperate folders
2. Prepare the labels for both folders
3. SIDC, DIDC, and Four-channels require to input left and right-side images simultaneously, so import both sides of image sets and corresponding labels into arrarys
4.  Match image sets with labels 
5.  Encode labels into one_hot formats
6.  Training and testing splits through cross-validation (**stratified CV if data set is imbalanced**)
7.  Construct the architectures:
    - SIDC architecture at [SIDC.py](https://github.com/Amyyy-z/Multi-channel-DCNN/blob/Multi-channel-CNN/SIDC.py) 
    - DIDC architecture at [DIDC.py](https://github.com/Amyyy-z/Multi-channel-DCNN/blob/Multi-channel-CNN/DIDC.py) 
    - Four-channel achitecture at [Four-channel.py](https://github.com/Amyyy-z/Multi-channel-DCNN/blob/Multi-channel-CNN/Four-channel.py)
8. Train and test the model through stratified CV
9. Print out the confusion matrix to evaluate the model performance

--------------------------------------------------

# Multi-channel Multi-class CNN implementation
Besides multi-channel CNN for binary classification, this repository also contains the work for multi-channel multi-class classification tasks.
The implementation procedures were the same as the above steps.

If you are looking for the multi-class dataset, please download at [Multi-class Dataset.zip](https://github.com/Amyyy-z/Multi-channel-DCNN/blob/Multi-channel-CNN/Multi-class%20Dataset.zip).

The step-by-step implementation procedures can be found through [Multi-channel Multi-class Classification.py](https://github.com/Amyyy-z/Multi-channel-DCNN/blob/Multi-channel-CNN/Multi-channel%20Multi-class%20Classification.py)


### Contact
For any other inquiries, please email: xinyu.zhang@monash.ed

--------------------------------------------------------------------------------------------------------

**Thank you**

