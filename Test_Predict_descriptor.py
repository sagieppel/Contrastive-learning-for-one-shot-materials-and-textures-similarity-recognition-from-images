# Predict descriptor for an image
# ...............................Imports..................................................................
import os
import random
import json
import cv2
import numpy as np
import torch
import argparse
import NetAndLoss as NetBuild
import torch.nn.functional as F
##################################Input paramaters#########################################################################################
parser = argparse.ArgumentParser(description='Predict descriptor for an image')
parser.add_argument('--input_image', default=r"sample_data/test/Candies_candy_IMG_20220506_165103_792.jpg", type=str, help='image to use')
parser.add_argument('--input_mask', default= r"sample_data/test/Candies_candy_IMG_20220506_165103_792_MASK.png", type=str, help='mask of the material region in the image')
parser.add_argument('--train_model_path', default= r"logs/Defult.torch", type=str, help='path to trained model')
parser.add_argument('--max_img_size', default= 900, type=int, help=' max image size, larger images will be shrinked')
parser.add_argument('--min_img_size', default= 200, type=int, help=' min image size, smaller images will be resized')
parser.add_argument('--use_roi_mask', default= True, type=bool, help=' read roi mask of the object in from a file x_MASK.png where x.jpg is the image file, otherwise the mask will generated to cover the all image values 0,255')
parser.add_argument('--crop', default= False, type=bool, help=' crop image around ROI mask')
parser.add_argument('--mask', default= True, type=bool, help=' mask image around ROI mask')

args = parser.parse_args()

# use: python Test_Predict_descriptor.py --input_image sample_data/test/Candies_candy_IMG_20220506_165103_792.jpg --input_mask sample_data/test/Candies_candy_IMG_20220506_165103_792_MASK.png --train_model_path logs/Defult.torch
if __name__ == "__main__":
    # load model
    Net = NetBuild.Net()  # Create net and load pretrained
    Net.load_state_dict(torch.load(args.train_model_path))

    # load image and mask

    img = cv2.imread(args.input_image)  # read image
    if args.use_roi_mask:
         mask = cv2.imread(args.input_mask, 0)  # read ROI mask from file
    else:
        mask = img[:, :, 0] * 0 + 255 # ROI region cover all image


    # Crop resize and preproccess image
    h, w = mask.shape
    mn = np.min([h, w])
    mx = np.max([h, w])

    if args.crop:
        [xb, yb, wb, hb] = cv2.boundingRect(mask)

        mask = mask[yb:yb + hb, xb:xb + wb]
        img = img[yb:yb + hb, xb:xb + wb]

    if mx > args.max_img_size:
        r = args.max_img_size / mx
        img = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (int(w * r), int(h * r)), interpolation=cv2.INTER_NEAREST)
    h, w = mask.shape
    mn = np.min([h, w])
    if mn < args.min_img_size:
        r = args.min_img_size / mn
        img = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (int(w * r), int(h * r)), interpolation=cv2.INTER_NEAREST)

    if args.mask:
        img_mask = img.copy()
        for ch in range(3): img_mask[:, :, ch][mask == 0] = 0  ## MASKING


#  Convert image to pytorch

    img = np.expand_dims(img, axis=0).astype(np.float32)
    mask = (np.expand_dims(mask, axis=0) > 0).astype(np.float32)
    with torch.no_grad():
        desc = Net.forward(img, mask, TrainMode=False)
        desc = F.normalize(desc, dim=1)
# display descriptor
    print("descriptor")
    print(str(list(desc[0].cpu().detach().numpy())))

