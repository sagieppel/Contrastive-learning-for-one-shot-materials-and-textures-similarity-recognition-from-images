# recursive create descriptors for all images in folder and subfolder in targer dir (data_folder)
# and return dictionary of descriptors with the same structure as the folder hirarchis and save in json file, this file can be evaluated  by GetStatistics on file
#
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
parser = argparse.ArgumentParser(description='Run on folder and create descriptor to all images in folder and subdirs, folder should have images  as X.jpg and the corresponding mask as X_MASK.png')
parser.add_argument('--test_set_path', default=r"sample_data/test_set1/", type=str, help='path to folder with images and masks')
parser.add_argument('--train_model_path', default= r"logs/Defult.torch", type=str, help='path to trained model')
parser.add_argument('--out_json_path', default= "logs/descriptor_dict.json", type=str, help='file where descriptor dictionary will be saved,  dictionary of descriptors with the same structure as the folder hirarchis and save in json file ')
parser.add_argument('--max_img_size', default= 900, type=int, help=' max image size, larger images will be shrinked')
parser.add_argument('--min_img_size', default= 200, type=int, help=' min image size, smaller images will be resized')
parser.add_argument('--use_roi_mask', default= True, type=bool, help=' read roi mask of the object in from a file x_MASK.png where x.jpg is the image file, otherwise the mask will generated to cover the all image')
parser.add_argument('--crop', default= False, type=bool, help=' crop image around ROI mask')
parser.add_argument('--mask', default= True, type=bool, help=' mask image around ROI mask')
parser.add_argument('--save_to_file', default= True, type=bool, help='Save descriptor to file')
parser.add_argument('--UseAverageMaskUnMask', default= False, type=bool, help='')
args = parser.parse_args()
add_txt = ""  # Text that will be added to the output descriptor file

#usage: python RunNetOnFolder.py --test_set_path  sample_data/test_set1 --train_model_path logs/Defult.torch  --out_json_path logs/descriptor_dict.json
############################################################################################
if args.UseAverageMaskUnMask == True: args.mask= True
# if args. == True: add_txt += "_Crop_"
# if UseAverageMaskUnMask == True:
#     add_txt += "_AverageMaskUnMasked_"
# elif UseMask:
#     add_txt += "_MASKED_"


#############################################################################################
def RecursiveRun(path, Net, args):
    '''recursive create descriptor for all images in folder and subfolder and return dictionary of descriptors with the same structure as the folder hirarchis'''
    DescDic = {"descs": {}, "dirs": {}, "prop": {}}  # dictionary were image descriptors will be stored
    # scan all files in dir
    for nm in os.listdir(path):
        newpath = path + "/" + nm
        # recursively run on subfolders
        if os.path.isdir(newpath):
            print(newpath)
            DescDic["dirs"][nm] = RecursiveRun(newpath, Net, args)
            continue
        # load image and mask
        elif ("_MASK.png" in newpath) and args.use_roi_mask:
            print(newpath)
            mask = cv2.imread(newpath, 0)# read ROI mask from file
            if not os.path.exists(newpath.replace("_MASK.png", ".jpg")):
                print("missing ", newpath.replace("_MASK.png", ".jpg"))
                continue

            img = cv2.imread(newpath.replace("_MASK.png", ".jpg")) # read image
            #  resize image if too large  or small
        elif args.use_roi_mask==False and ".jpg" in newpath: # Use the all image as mask
            img = cv2.imread(newpath)  # read image
            mask= img[:,:,0]*0+255
        else:
            continue

        # cv2.imshow(",", img);
        # cv2.imshow(",", mask);
        # cv2.waitKey()
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
            if not args.UseAverageMaskUnMask:
                img = img_mask.copy()

        # cv2.imshow(",", np.hstack([img]))
        # #cv2.imshow(",", np.hstack([img,img_mask]))
        # cv2.waitKey()

        img = np.expand_dims(img, axis=0).astype(np.float32)
        mask = (np.expand_dims(mask, axis=0) > 0).astype(np.float32)
        with torch.no_grad():
            desc = Net.forward(img, mask, TrainMode=False)
            desc = F.normalize(desc, dim=1)
        if args.UseAverageMaskUnMask:
            img_mask = np.expand_dims(img_mask, axis=0).astype(np.float32)
            with torch.no_grad():
                desc_mask = Net.forward(img_mask, mask, TrainMode=False)
                desc_mask = F.normalize(desc_mask, dim=1)
            desc = desc + desc_mask
            ####  desc=torch.cat([desc, desc_mask], 1)
            desc = F.normalize(desc, dim=1)
        # store descriptor in dictionary
        DescDic["descs"][nm.replace("_MASK.png", "")] = str(list(desc[0].cpu().detach().numpy()))
        DescDic["prop"][nm.replace("_MASK.png", "")] = {"size h,w": str(h) + "," + str(w), "path": newpath}
        #
            # with open("www.json", 'w') as fp:
            #     json.dump(DescDic, fp)
            # with open("www.json") as fp:
            #     data = json.load(fp)
    return DescDic

##############################################################################################################################
def Run(DataPath, Trained_model_path,args, Net=None, outpath=None):  # Pretrain model path
    ''' run Net on the images ain DataPath using trained weight in Trained_model_path, save  '''
    if Net is None:
        Net = NetBuild.Net()# Create net and load pretrained
    if outpath is None:
        outpath=Trained_model_path+".json"
    Net.load_state_dict(torch.load(Trained_model_path))


    Net.eval()
    desc_dic = RecursiveRun(DataPath, Net,args=args)
    if args.save_to_file:
        print("save file to :", outpath)
        with open(outpath, 'w') as fp:
            json.dump(desc_dic, fp)
    return desc_dic

    # with open(outpath) as fp2:
    #     data = json.load(fp2)
    #     print(data)


###########################################################################33

def RunFolder(DataPath, Trained_modelFolder, Net, args):
    '''run on multiple models stored in  Trained_modelFolder, on the images on DataPath, create one descriptor dictionary per trained model'''
    for model_name in os.listdir(Trained_modelFolder):
        if ".torch" in model_name:
            Trained_model_path = Trained_modelFolder + "/" + model_name
            Run(DataPath,Trained_model_path,args=args, Net=Net,
                outpath=Trained_modelFolder + "/" + model_name.replace(".torch", add_txt + "desc.json"))

###############################################################################################################3

if __name__ == "__main__":
    Run(DataPath= args.test_set_path, args=args,Trained_model_path = args.train_model_path ,  outpath = args.out_json_path )

