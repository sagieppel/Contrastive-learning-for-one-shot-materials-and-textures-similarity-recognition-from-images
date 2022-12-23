# Train net to predict materials similarity using the MatSim Dataset set
#...............................Imports..................................................................
import os
import numpy as np
import torch
import SimLoader as Reader
import NetAndLoss as NetBuild
import argparse# Create the parser
parser = argparse.ArgumentParser()
##################################Input paramaters#########################################################################################
#.................................Main Input parametrs...........................................................................................
# Input folders

parser = argparse.ArgumentParser(description='Train on MatSim')
parser.add_argument('--MatSim_dir_object', default= r"sample_data/VesselTrainData/", type=str, help='input folder 1 MatSim synthethic Objects dataset main dir')
parser.add_argument('--MatSim_dir_vessel', default= r"sample_data/ObjectTrainData/", type=str, help='input folder 2 MatSim synthethic Vessels dataset main dir')
parser.add_argument('--MaxPixels', default= 800*800*12, type=int, help='max Size of input matrix in pixels H*W*BatchSize (reduce to solve cuda out of memory)')
parser.add_argument('--MaxImagesInBatch', default = 15, type=int, help='max images in a a batch (reduce to solve cuda out of memory)')
parser.add_argument('--temp', default= 0.2, type=float, help='temperature for softmax')
parser.add_argument('--weight_decay', default= 4e-5, type=float, help='optimizer weight decay')
parser.add_argument('--learning_rate', default= 1e-5, type=float, help='optimizer learning rate')
parser.add_argument('--max_iteration', default= 200001, type=float, help='max training iteration')
parser.add_argument('--log_dir', default= r"logs/", type=str, help='log folder were train model will be saved')
parser.add_argument('--resume_training_from', default= r"", type=str, help='path to model to resume training from')
parser.add_argument('--auto_resume', default= True, type=bool, help='start training from existing last saved model (Defult.torch)')
parser.add_argument('--min_img_size', default= 200, type=int, help='min image size for augmentation')
parser.add_argument('--max_img_size', default= 1100, type=int, help='max image size for augmentation')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Training hardware", device)
args = parser.parse_args()

# usage: python Train.py --MatSim_dir_object sample_data/VesselTrainData/ --MatSim_dir_vessel sample_data/ObjectTrainData/ --log_dir logs/
#************** Set folder for trained model**************************************************************************************************************************************


if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)
# Trained_model_path="" # Path of trained model weights If you want to return to trained model, else should be =""

#=========================Load net weights if exist====================================================================================================================
InitStep=0
if args.auto_resume:
    if os.path.exists(args.log_dir + "/Defult.torch"):
        Trained_model_path=args.log_dir  + "/Defult.torch"
    if os.path.exists(args.log_dir +"/Learning_Rate.npy"):
        Learning_Rate=np.load(args.log_dir +"/Learning_Rate.npy")
    if os.path.exists(args.log_dir +"/itr.npy"): InitStep=int(np.load(args.log_dir +"/itr.npy"))
#...............Other training paramters..............................................................................

TrainLossTxtFile=args.log_dir +"TrainLoss.txt" #Where train losses statitics will be writen

#---------------------Create and Initiate net and create optimizer------------------------------------------------------------------------------------
Net=NetBuild.Net() # Create net and load pretrained


if args.resume_training_from!="": # Optional initiate full net by loading a pretrained net weights
    Net.load_state_dict(torch.load(args.resume_training_from))


optimizer=torch.optim.AdamW(params=Net.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay) # Create  optimizer

#----------------------------------------Create reader for data set--------------------------------------------------------------------------------------------------------------

SimReaderObjects=Reader.Reader(args.MatSim_dir_object)
SimReaderVessels=Reader.Reader(args.MatSim_dir_vessel)
#--------------------------- Create logs files for saving loss statistics during training----------------------------------------------------------------------------------------------------------
f = open(TrainLossTxtFile, "w+")# Training loss log file
f.write("Iteration\tloss\t Learning Rate=")
f.close()
AVGLoss=0 # running average loss
AVGAcr={}
for mode in ['objects','vessels']:
  AVGAcr[mode] = {}
  for i in range(0,5):
      AVGAcr[mode][i/4]=0
#..............Start Training loop: Main Training....................................................................
print("Start Training")
scaler = torch.cuda.amp.GradScaler() # For mixed precision

for itr in range(InitStep,args.max_iteration): # Main training loop

#----------------------Load next batch---------------------------------------------------------------------
   # print(itr)
    if np.random.rand()<0.5: #load from each loader with 50% probability
        mode="objects"
        Images, Masks, frctLst, matDirLst, sceneDirLst=SimReaderObjects.LoadRandomAugmentedBatch(MaxImagesInBatch=args.MaxImagesInBatch,MaxPixelsBatch=args.MaxPixels,minSz=args.min_img_size,MaxSz=args.max_img_size)
    else:
         mode = "vessels"
         Images, Masks, frctLst, matDirLst, sceneDirLst = SimReaderVessels.LoadRandomAugmentedBatch(MaxImagesInBatch=args.MaxImagesInBatch,MaxPixelsBatch=args.MaxPixels,minSz=args.min_img_size,MaxSz=args.max_img_size)

#***********************************************************************************************
    # for i in range(Images.shape[0]):
    #     for j in range(Images.shape[0]):
    #         if i==j: continue
    #         im1 = Images[i].copy()
    #         mask1 = Masks[i]
    #         im2 = Images[j].copy()
    #         mask2 = Masks[j]
    #         im1[:, :, 0] *= mask1
    #         im1[:, :, 1] *= mask1
    #         im2[:, :, 0] *= mask2
    #         im2[:, :, 1] *= mask2
    #         vis.show(np.concatenate([np.hstack([im1,Images[i]]),np.hstack([im2,Images[j]])],0),str(frctLst[i]-frctLst[j])+"  SameDir="+str(matDirLst[i]==matDirLst[j]))
# #******************************Main Train steps***********************************************************************

    with torch.cuda.amp.autocast():
        Net.zero_grad()
        desc = Net.forward(Images, Masks, TrainMode=True)
        Loss,tpfp = Net.SoftLossCosineSimilarity(desc,frctLst, matDirLst,temp=args.temp)
    #-----------------------backpropogate----------------------------------------------------------------------------------------------------------------
        scaler.scale(Loss).backward()  # Backpropogate loss caler used for mix precision
        scaler.step(optimizer)  # Apply gradient descent change to weight scaler used for mix precision
        scaler.update()
######Calculate running average loss############################################################

    fr = 1 / np.min([itr - InitStep + 1, 2000])
    AVGLoss = AVGLoss * (1-fr) + fr * float(Loss.data.cpu().numpy()) # Average loss
    for ii in tpfp:
        if sum(tpfp[ii])>0:
           AVGAcr[mode][ii] = AVGAcr[mode][ii] * (1 - fr ) + fr * tpfp[ii][0]/sum(tpfp[ii]) # Add true positives

#===================save statitics and displaye loss======================================================================================
# --------------Save trained model------------------------------------------------------------------------------------------------------------------------------------------
    if itr % 200 == 0 and itr>0: #Save model weight and other paramters in temp file once every 1000 steps
        print("Saving Model to file in "+args.log_dir+"/Defult.torch")
        torch.save(Net.state_dict(), args.log_dir + "/Defult.torch")
        torch.save(Net.state_dict(), args.log_dir + "/DefultBack.torch")
        print("model saved")
        np.save(args.log_dir+"/Learning_Rate.npy",args.learning_rate)
        np.save(args.log_dir+"/itr.npy",itr)
        torch.cuda.empty_cache() # clean memory
    if itr % 50000 == 0 and itr>0: #Save model weight once every 30k steps permenant (not temp)
        print("Saving Model to file in "+args.log_dir+"/"+ str(itr) + ".torch")
        torch.save(Net.state_dict(), args.log_dir + "/" + str(itr) + ".torch")
        print("model saved")
#......................Write and display train loss..........................................................................
    if itr % 10==0: # Display  and save train loss

        txt="\n"+str(itr)+"\t Average Loss "+str(AVGLoss) +"\t Average Accuracy"+str(AVGAcr)+"\n"
        for mode in ['objects', 'vessels']:
            txt+="\n"+mode +" ) "
            for ii in AVGAcr[mode]:
                 txt+=")  Similarity "+str(ii)+" Accuracy)"+str(AVGAcr[mode][ii])
        print(txt)
        #Write train loss to file
        with open(TrainLossTxtFile, "a") as f:
            f.write(txt)
            f.close()


