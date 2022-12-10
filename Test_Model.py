# Given a folder of images (x.jpg) and their corresponding ROI masks (x_MASK.png) find for each image the most similar image in the folder (should run out of the box with sample folder)
# ...............................Imports..................................................................
import numpy as np
import argparse
import RunNetOnFolder as run
##################################Input paramaters#########################################################################################
parser = argparse.ArgumentParser(description='Given a folder of images (x.jpg) and their corresponding ROI masks (x_MASK.png) find for each image the most similar image in the folder (should run out of the box with sample folder)')
parser.add_argument('--input_folder', default=r"sample_data/test", type=str, help='path to folder with images and masks')
parser.add_argument('--train_model_path', default= r"logs/Defult.torch", type=str, help='path to trained model')
parser.add_argument('--max_img_size', default= 900, type=int, help=' max image size, larger images will be shrinked')
parser.add_argument('--min_img_size', default= 200, type=int, help=' min image size, smaller images will be resized')
parser.add_argument('--use_roi_mask', default= True, type=bool, help=' read roi mask of the object in from a file x_MASK.png where x.jpg is the image file, otherwise the mask will generated to cover the all image')
parser.add_argument('--crop', default= False, type=bool, help=' crop image around ROI mask')
parser.add_argument('--mask', default= True, type=bool, help=' mask image around ROI mask')
parser.add_argument('--UseAverageMaskUnMask', default= False, type=bool, help='')
parser.add_argument('--save_to_file', default= False, type=bool, help='Save descriptor to file')
args = parser.parse_args()

# Usage python Test_Model.py --input_folder sample_data/test  --train_model_path logs/Defult.torch

if __name__ == "__main__":
    print("Generating Descriptors for ",args.input_folder, "\\n Using model ",args.train_model_path,"\n\n\n\n")
    desc_dict=run.Run(DataPath = args.input_folder, args=  args, Trained_model_path= args.train_model_path ,  outpath = "")
    desc=desc_dict['descs']
    print("\n------  results---------\n Top 1 match list")
    for nm in desc: # match descriptor
        max_sim=-1
        max_sim_file=""
        for nm2 in desc:
            if nm2!=nm:
                dc1 = desc[nm]
                dc2 = desc[nm2]
                if type(dc2)==str:
                    dc1 = np.fromstring(dc1.replace('[', '').replace(']', ''), dtype=np.float32, sep=', ')
                    dc2 = np.fromstring(dc2.replace('[', '').replace(']', ''), dtype=np.float32, sep=', ')
                sim = ((dc1 * dc2).sum()) # Get cosine similarity
                if sim>=max_sim:
                    max_sim=sim
                    max_sim_file = nm2

        print("Image:",nm," best match with",max_sim_file,". With similarity:",str(max_sim))

