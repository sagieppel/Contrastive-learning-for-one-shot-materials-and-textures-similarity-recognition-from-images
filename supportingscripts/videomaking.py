#  Find best match for each frame in the video, from a folder of sample images.
#  Get for each frame of the video witch image it best match image
# ...............................Imports..................................................................
import os.path

import numpy as np
import argparse
import RunNetOnFolder as run
import cv2
import NetAndLoss as NetBuild
import torch
import torch.nn.functional as F
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
########################################################################################################################################3

# Receive image and net and predict descriptor

###########################################################################################################################33
def getdescriptor(Net, img, mask):
    # img = cv2.imread(newpath.replace("_MASK.png", ".jpg"))  # read image
    # #  resize image if too large  or small
    # elif args.use_roi_mask == False and ".jpg" in newpath:  # Use the all image as mask
    # img = cv2.imread(newpath)  # read image
    ##    mask = img[:, :, 0] * 0 + 255
    # else:
    # continue

    # cv2.imshow(",", img);
    # cv2.imshow(",", mask);
    # cv2.waitKey()
    h, w = mask.shape
    mn = np.min([h, w])
    mx = np.max([h, w])

    if args.crop: # crop using ROI
        [xb, yb, wb, hb] = cv2.boundingRect(mask)

        mask = mask[yb:yb + hb, xb:xb + wb]
        img = img[yb:yb + hb, xb:xb + wb]

    if mx > args.max_img_size: # resize if to large
        r = args.max_img_size / mx
        img = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (int(w * r), int(h * r)), interpolation=cv2.INTER_NEAREST)
    h, w = mask.shape
    mn = np.min([h, w])
    if mn < args.min_img_size:# resize if to small
        r = args.min_img_size / mn
        img = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (int(w * r), int(h * r)), interpolation=cv2.INTER_NEAREST)

    if args.mask: # apply masking
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
    return desc.cpu().numpy()[0]
#######################################################################################################################################3

# receive video and a set of images and for each frame of the video predict  which image best match the frame

########################################################################################################################################
class VideoMatSim():
    def __init__(self, video_file,images_folder,model_path,output_path,frame_gap, roi_ratio):

        # Load Model-----------------------------------------------------
        self.Net = NetBuild.Net()  # Create net and load pretrained
        self.Net.load_state_dict(torch.load(model_path))
        self.Net.eval()

        # Load examples-------------------------------------------------------------------

        self.exm={} # Examples materials used to match to the images of the frames
        for fl in os.listdir(images_folder): # load examples
            self.exm[fl]={}
            self.exm[fl] = {}
            if not os.path.isfile(images_folder +"/"+fl):continue
            self.exm[fl]["im"]=cv2.imread(images_folder +"/"+fl)
            # Set ROI Mask----------------------------------------------------------------------
            self.exm[fl]["mask"] = self.create_mask(self.exm[fl]["im"], roi_ratio) # create mask

            # cv2.imshow("im",self.exm[fl]["im"])
            # self.exm[fl]["im"][:,:,0][self.exm[fl]["mask"]>0]=0
            # self.exm[fl]["im"][:, :, 1][self.exm[fl]["mask"] > 0] = 255
            # cv2.imshow("msk", self.exm[fl]["im"])
            # cv2.waitKey()

            self.exm[fl]["descriptor"] = getdescriptor(self.Net, self.exm[fl]["im"],  self.exm[fl]["mask"]) #get descriptor
            #Load video--------------------------------------------------------------------------
            # Open the video file
        self.cap = cv2.VideoCapture(video_file)
        if not self.cap.isOpened():
            print("Error: Could not open video.")
            exit()
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.itr=0
        self.frame_gap=frame_gap # read one of k frames
        #-----------------------Create output folder------------------------------------
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        self.outdir=output_path
#####################################################################################################################################

 # Create attention mask that cover the center of the image (size determied by roi_ratio)

###################################################################################################################################
    def create_mask(self,im,roi_ratio):
        h, w, d = im.shape
        msk = im[:, :, 0] * 0
        if roi_ratio == 1:
            msk += 255
        rt = (1 - roi_ratio) / 2
        msk[int(h * rt):-int(h * rt), int(w * rt):-int(w * rt)] = 255
        return msk
 ##############################################################################################################################333

# Load next frame from video and match it to referance images, display and save results and return data

#####################################################################################################################################
    def get_next_frame_data(self,roi_ratio,temp=1,show_roi=False,display_im=True,save_to_file=True):
        self.itr+=self.frame_gap
        print("frame",int(self.itr))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(self.itr))
        ret, self.frame = self.cap.read()
        # Read a frame
        print("1")
        # for i in range(self.frame_gap):
        #     ret, self.frame = self.cap.read()
        # If frame is read correctly, ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            return False,False,False
        print("2")
        self.mask = self.create_mask(self.frame, roi_ratio)  # create mask
        # cv2.imshow("frame", self.frame)
        # self.frame[:,:,0][self.mask>0]=0
        # self.frame[:, :, 1][self.mask > 0] = 255
        # cv2.imshow("frame msk",  self.frame)
        # cv2.waitKey()

        desc = getdescriptor(self.Net, self.frame, self.mask)  # get descriptor
        self.similarity = {}

        sumexp = 0
        for ky in self.exm:
            self.similarity[ky]=(self.exm[ky]["descriptor"]*desc).sum()
            sumexp+=np.exp(self.similarity[ky]*temp)
        #----------Find Best Match------------------------------------------------------------------------
        self.probs = {}
        max_prob=0
        best_match=""

        for ky in self.exm:
            self.probs[ky] = np.exp(self.similarity[ky]*temp)/sumexp
            if self.probs[ky]>=max_prob:
                  max_prob= self.probs[ky]
                  best_match = ky
        print("3")
        #-----------Display----------------------------------------------------------------------
        if display_im or save_to_file:
                im=self.frame.copy()
                msk=self.mask.copy()

                rat=1100/max(list(im.shape[:2]))
                msk = cv2.resize(msk, (int(rat * im.shape[1]), int(rat * im.shape[0])))
                im=cv2.resize(im,(int(rat*im.shape[1]),int(rat*im.shape[0])))
                #  ROI square
                if  show_roi:
                    rec = cv2.dilate(msk,np.ones([17,17]))>msk
                    im[:, :, 0][rec] = 255
                    im[:, :, 1][rec] = 0
                    im[:, :, 2][rec] = 0
                #cv2.destroyAllWindows()
                #  Examples images
                exmp=[]
                for ky in self.exm:
                    i1=self.exm[ky]["im"]
                    r=self.frame.shape[0]/i1.shape[0]/3
                    im2 = cv2.resize(i1,(int(r*i1.shape[1]),int(r*i1.shape[0])))
                    # Mark the matched image
                    if ky==best_match:
                        color = [0,255,0 ]
                        gp=50
                        for c in range(3):
                           im2[ :gp, :, c] = color[c]
                           im2[-gp:, :, c] = color[c]
                           im2[:, :gp, c] = color[c]
                           im2[:, -gp:, c] = color[c]
                    # Add upper and lower panel
                    bsep = np.zeros([20, int(im2.shape[1]), 3])
                    tsep = np.zeros([120, int(im2.shape[1]), 3])
                    if ky==best_match:
                        tsep[:,:,1]=255
                        tsep=cv2.resize(cv2.imread("vidcaption.jpg"),(tsep.shape[1],tsep.shape[0]))

                    else:
                        tsep[:, :, 2] = 255
                    im2= np.concatenate([tsep,im2,bsep],axis=0).astype(np.uint8)
                    exmp.append(im2)
                    sidesep=np.zeros([im2.shape[0],int(im2.shape[1]/8),3])
                    exmp.append(sidesep)
                topim=np.concatenate(exmp,axis=1).astype(np.uint8)
                r=im.shape[1]/topim.shape[1]
                topim=cv2.resize(topim,[int(topim.shape[1]*r),int(topim.shape[0]*r)])
                im = np.concatenate([topim,im],axis=0)
                print(best_match)
            #    print(self.probs)
                if display_im:
                    cv2.imshow("",im)
                    cv2.waitKey(1)
                if save_to_file:
                    cv2.imwrite(self.outdir+"/"+str(self.itr)+".jpg",im)
                print("4")
        return True,self.probs,self.similarity
#####################################################################################################################################




if __name__ == "__main__":
   
    video_file = r"/mnt/306deddd-38b0-4adc-b1ea-dcd5efc989f3/VideoMatSim/BreadAndCheese/1.mp4" # video
    images_folder = r"/mnt/306deddd-38b0-4adc-b1ea-dcd5efc989f3/VideoMatSim/BreadAndCheese/3/" # Sample images to match video framea
    model_path = args.train_model_path
    output_path ="/mnt/306deddd-38b0-4adc-b1ea-dcd5efc989f3/VideoMatSim/BreadAndCheese/1mp4to3dir/"
    frame_gap = 2 # Use one of k frames
    roi_ratio_im = 0.4 # Use center region as ROI for sample images (1 use all image)
    roi_ratio_vid = 0.3  # Use center region as ROI for video (1 use all frame)

    vidreader = VideoMatSim(video_file, images_folder, model_path, output_path, frame_gap, roi_ratio_im)
    while (True):

      #  ret, prob, sim = vidreader.get_next_frame_data(roi_ratio = 0.4)
        ret, prob, sim = vidreader.get_next_frame_data(roi_ratio=roi_ratio_vid,show_roi=False,display_im=True,save_to_file=True)
        if ret==False: break
