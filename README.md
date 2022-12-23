# Contrastive learning for one-shot materials and textures similarity recognition from images, using MatSim dataset
This net receives the image and region of the object in the image and predicts the descriptor for the material. This can be used to identify visual similarity between materials in different images (Figure 1), or identify the same material on different objects and environments using single example (one-shot). See [One-shot recognition of any material anywhere using contrastive learning with
physics-based rendering](https://arxiv.org/pdf/2212.00648.pdf).
#### PreTrained model can be downloaded from this [url](https://e1.pcloud.link/publink/show?code=kZGUwQZNV7k6Y8NrH5zjCQPQT0Fb0aFatjV) or [2](https://icedrive.net/s/Nv97G1gDvTvhgBzxzx6iRA3CPw6B), [3](https://zenodo.org/record/7390166#.Y5PrsWHMJH7)
The net was trained on the [MatSim dataset](https://zenodo.org/record/7390166#.Y5PrsWHMJH7) that can be downloaded from [1](https://e1.pcloud.link/publink/show?code=kZIiSQZCYU5M4HOvnQykql9jxF4h0KiC5MX), [2](https://icedrive.net/s/A13FWzZ8V2aP9T4ufGQ1N3fBZxDF) or [3](https://zenodo.org/record/7390166#.Y5PrsWHMJH7)
 
![](/Figure1.jpg)
![](/Figure2.jpg)

# Requirement:

Torch: [https://pytorch.org/](https://pytorch.org/)

OpenCV: [pip install opencv-python](https://pytorch.org/)

The net was trained with Conda 4.12.0 Torch 1.12.1+Cuda 113 on  a single RTX 3090.

#### Pretrained model is supplied at folder: logs/Defult.torch
#### Sample data is supplied at subfolder: sample_data 
#### Full MatSim dataset and benchmark can be downloaded from [here](https://e1.pcloud.link/publink/show?code=kZIiSQZCYU5M4HOvnQykql9jxF4h0KiC5MX) 

# Testing:
PreTrained model can be downloaded from this [url](https://e1.pcloud.link/publink/show?code=XZ0BGYZRFqGBQ5oHzRzt652juiFa8BrpFqy) or [2](https://icedrive.net/s/25Y9CP8w1wG51W6R7Bik1t51VANP) or [3] (https://zenodo.org/record/7390166#.Y5PrsWHMJH7)
Sample data are supplied with the code which should allow the model to run out of the box.

## Finding the best similarity to each image in a folder: Test_Model.py
For each image in a folder, find the most similar image in the same folder (most similar material).
### Usage:
Test_Model.py --input_folder sample_data/test  --train_model_path logs/Defult.torch


### parameters:

--input_folder input folder should contain images to compare, in format  x.jpg and their corresponding masks in format x_MASK.jpg  (regions of the object ROI in the mask should have a value of 255 all other pixels should be 0)

--train_model_path  path to the trained model

--use_roi_mask optional If you don't have a mask and want to predict the descriptor for all images set this to False 
### output
This should output list of images in the folder, and for each image, the name of the image with the most similar material.


## Predict descriptor for an image: Test_Predict_descriptor.py
### Usage: 
python Test_Predict_descriptor.py --input_image sample_data/test/Candies_candy_IMG_20220506_165103_792.jpg --input_mask sample_data/test/Candies_candy_IMG_20220506_165103_792_MASK.png --train_model_path logs/Defult.torch

### Parameters

--input_image Path for input RGB image 

--input_mask  Path for the mask of ROI  material region in the image (regions of the object ROI in the mask should have a value of 255 all other pixels should be 0)

--train_model_path  path to the trained model

--use_roi_mask optional If you don't have a mask and want to predict the descriptor for all images set this to False 

### Output

The output is 512 values descriptor vector for the image.

## Predict descriptor for every image  in folder and subfolders recursively and store in a json dictionary file : RunNetOnFolder.py

### usage:
python RunNetOnFolder.py --test_set_path  sample_data/test_set1 --train_model_path logs/Defult.torch 
--out_json_path logs/descriptor_dict.json

### Parameters
--test_set_path path to folder every image in the folder  and sub folder recursively should have format x.jpg and their corresponding masks in format x_MASK.jpg  (regions of the object ROI in the mask should have a value of 255 all other pixels should be 0)

--train_model_path  path to the trained model

--out_json_path path to json file where the dictionary of descriptors to all images in the input folder will be saved. Note the dictionary will have the same tree structure as the folder.

--use_roi_mask optional If you don't have a mask and want to predict the descriptor for all images set this to False 


### Output
Json dictionary file with the descriptors of all images in the folder and subfolders, with the same tree structure as the input folder.

### Evaluation
For evaluation of the the descriptor accuracy see: GetStatiticsOnFile_ByCatWeight.py 


# Traininig: Train.py
## usage:
python Train.py --MatSim_dir_object sample_data/VesselTrainData/ --MatSim_dir_vessel sample_data/ObjectTrainData/ --log_dir logs/
## parameters
--MatSim_dir_object path for the MatSim dataset [synthetic materials on objects  train set](https://e1.pcloud.link/publink/show?code=kZXOwQZ45biYwUEAg0GRCilfErYjh0WycaX)*

--MatSim_dir_vessel path for the MatSim dataset [synthetic material inside transparent vessels train set](https://e1.pcloud.link/publink/show?code=kZJOwQZpA3UO7aVlW0YCF5vGkPVdVJrVqQk)*

--log_dir logs/ logs dir where trained model will be saved.

* In training data from both folders will be sampled in equal probability (if you want to train with only one its possible for both to point for the same folder)

## output
Trainig for about 200,000 iteration (three days on RTX 3090) should give the model described in the paper.



















