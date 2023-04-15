## Reader MatSim dataset



import numpy as np
import os
import cv2
import json
import threading
import Visuallization as vis

##############################################################################################
ClassList=[0,0.25,0.5,0.75,1]


matFileNames={0:"RGB_0_RGB.jpg",0.25:"RGB_0.25_RGB.jpg",0.5:"RGB_0.5_RGB.jpg",0.75:"RGB_0.75_RGB.jpg",1:"RGB_1_RGB.jpg"}
#########################################################################################################################
class Reader:
# Initiate reader and define the main parameters for the data reader
    def __init__(self, MainDir=r""):
        self.epoch = 0 # Training Epoch
        self.itr = 0 # Training iteratation
#-----------------List Files----------------------------------------------------------------------------------------
        self.data=[]
        for dName in os.listdir(MainDir):
               pathName=MainDir+"/"+dName
               if not os.path.exists(pathName+r"/Finished.txt"): continue
               MatsData=[]

               for sceneName in os.listdir(pathName):
                    pathScene=pathName+"/"+sceneName
                    if not os.path.isdir(pathScene): continue
                    sceneData={}
                    sceneData["matDir"] = dName
                    sceneData["sceneDir"] = sceneName

                    if os.path.exists(pathScene+r"//MaskOcluded.png"):
                            sceneData['mask']=pathScene+r"//MaskOcluded.png" # mask of the object with the material
                    else:
                       sceneData['mask']=pathScene+r"/ContentMaskOcluded.png" # Mask of the vessel content
                    if not os.path.exists(sceneData["mask"]):
                               print("Missing",sceneData["mask"])
                               continue
                    if os.path.isdir(pathScene):
                        for frct in matFileNames:
                            matFile=matFileNames[frct]
                            if os.path.exists(pathScene+"/"+matFile):
                                sceneData[frct]=pathScene+"/"+matFile
                            else:
                                print("Missing",pathScene+matFile)
                                continue
                    MatsData.append(sceneData)
               self.data.append(MatsData)
        print("done creating file list")




######################################################Augmented Image add noise##################################################################################################################################
    def Augment(self,Img,Mask):
        Img=Img.astype(np.float32)
        if np.random.rand()<0.5: # flip left right
                Img = np.fliplr(Img)
                Mask = np.fliplr(Mask)
        if np.random.rand()<0.5: # up down
                Img = np.flipud(Img)
                Mask = np.flipud(Mask)

        # if np.random.rand()<0.5: # Change from rgb to bgr
        #     for i in Img:
        #         Img[i] = Img[i][..., :: -1]

        if np.random.rand() < 0.035: # Add noise
                noise = np.random.rand(Img.shape[0],Img.shape[1],Img.shape[2])*0.2+np.ones(Img.shape)*0.9

                Img*=noise
                Img[Img>255]=255
        #
        if np.random.rand() < 0.2: # Gaussian blur

                Img= cv2.GaussianBlur(Img, (5, 5), 0)

        if np.random.rand() < 0.25:  # Dark light

                Img= Img * (0.5 + np.random.rand() * 0.65)
                Img[Img>255]=255

        if np.random.rand() <0.25:# 0.33:  # GreyScale
                r = np.random.rand()

                Gr=Img.mean(axis=2)
                Img[:, :, 0] = Img[:, :, 0] * r + Gr * (1 - r)
                Img[:, :, 1] = Img[:, :, 1] * r + Gr * (1 - r)
                Img[:, :, 2] = Img[:, :, 2] * r + Gr * (1 - r)


        return Img, Mask
########################################################################################################################################################
#############################################################################################################################
#############################################################################################################################
# Crop and resize image and annotation mask and ROI to feet batch size
    def CropResize(self,Img, Mask ,Hb,Wb):
        # ========================resize image if it too small to the batch size==================================================================================
        bbox= cv2.boundingRect(Mask.astype(np.uint8))
        [h, w, d] = Img.shape
#====================================================================================================

        Wbox = int(np.floor(bbox[2]))  # Segment Bounding box width
        Hbox = int(np.floor(bbox[3]))  # Segment Bounding box height
        if Hbox == 0 or Wbox == 0:
            print("empty box")
            exit(0)
        if Wbox==0:  Wbox+=1
        if Hbox == 0: Hbox += 1

        Rs = np.max((Hb / h, Wb / w)) # Check if target size larger then corrent image size
        Bs = np.min((Hb / Hbox, Wb / Wbox)) # Check if target size smaller then bounding box
        if Rs > 1:# or Bs<1:  # Resize image and mask to batch size if mask is smaller then batch or if segment bounding box larger then batch image size
            h = int(np.max((h * Rs, Hb)))
            w = int(np.max((w * Rs, Wb)))
            Img = cv2.resize(Img, dsize=(w, h), interpolation=cv2.INTER_LINEAR)

            Mask = cv2.resize(Mask, dsize=(w, h), interpolation=cv2.INTER_NEAREST)
            bbox = (np.float32(bbox) * Rs.astype(np.float32)).astype(np.int64)

 # =======================Crop image to fit batch size===================================================================================
        x1 = int(np.floor(bbox[0]))  # Bounding box x position
        Wbox = int(np.floor(bbox[2]))  # Bounding box width
        y1 = int(np.floor(bbox[1]))  # Bounding box y position
        Hbox = int(np.floor(bbox[3]))  # Bounding box height

        if Wb > Wbox:
            Xmax = np.min((w - Wb, x1))
            Xmin = np.max((0, x1 - (Wb - Wbox)-1))
        else:
            Xmin = x1
            Xmax = np.min((w - Wb, x1 + (Wbox - Wb)+1))

        if Hb > Hbox:
            Ymax = np.min((h - Hb, y1))
            Ymin = np.max((0, y1 - (Hb - Hbox)-1))
        else:
            Ymin = y1
            Ymax = np.min((h - Hb, y1 + (Hbox - Hb)+1))

        if Ymax<=Ymin: y0=Ymin
        else: y0 = np.random.randint(low=Ymin, high=Ymax + 1)

        if Xmax<=Xmin: x0=Xmin
        else: x0 = np.random.randint(low=Xmin, high=Xmax + 1)

        # Img[:,:,1]*=Mask
        # misc.imshow(Img)

        Img = Img[y0:y0 + Hb, x0:x0 + Wb, :]

        Mask = Mask[y0:y0 + Hb, x0:x0 + Wb]
        if not (Mask.shape[0] == Hb and Mask.shape[1] == Wb): InsMasks = cv2.resize(Mask.astype(float),dsize=(Wb, Hb), interpolation=cv2.INTER_NEAREST)
#------------------------------------------Verify shape match the batch shape----------------------------------------------------------------------------------------
        if not (Img.shape[0] == Hb and Img.shape[1] == Wb): Img = cv2.resize(Img, dsize=(Wb, Hb),interpolation=cv2.INTER_LINEAR)


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
        return Img,Mask

#############################################Load batch for training #####################################################################################################################


    def LoadRandomAugmentedBatch(self,MaxImagesInBatch,MaxPixelsBatch,minSz,MaxSz):
        # -----------------------Choose Image size, and create batch size-------------------------------------------------------------------------------------
        Hb = np.random.randint(minSz, MaxSz)
        Wb = np.random.randint(minSz, MaxSz)
        Bsize=np.min([int(np.floor(MaxPixelsBatch/Hb/Wb)),MaxImagesInBatch])
        Bmsk = np.zeros([Bsize, Hb, Wb], np.float32)
        BImg = np.zeros([Bsize, Hb, Wb, 3], np.float32)
        nSample = 3
        frctLst = []
        matDirLst = []
        sceneDirLst = []
        #------------------------------------------------------------------------------------------------------------------------------
        nim = 0

        while(True):
                #..................Pick random set and two materials from this entry..................................................................
                Entry = self.data[np.random.randint(len(self.data))]  # Pick random set
                for k in range(nSample): # pick nSample from the set (note a batch images are all from the same set)

                    while (True): # make sure that no image is picked twice
                       scene = Entry[np.random.randint(len(list(Entry)))]
                       matDir =  scene["matDir"]
                       sceneDir = scene["sceneDir"]
                       frc = ClassList[np.random.randint(len(ClassList))]

                       duplicate=False
                       for ii in range(len(frctLst)):
                           if frctLst[ii]==frc and matDirLst[ii]== matDir and sceneDirLst[ii] == sceneDir:  duplicate=True
                         #  if matDirLst[ii] == matDir and sceneDirLst[ii] == sceneDir:  duplicate = True
                       if not duplicate: break


                    im = cv2.imread(scene[frc])
                    msk = (cv2.imread(scene["mask"], 0) > 100).astype(np.uint8)
                    if msk.sum()<100: continue

                    im, msk = self.CropResize(im, msk, Hb, Wb)

                    im, msk = self.Augment(im, msk)
                    Bmsk[nim] = msk
                    BImg[nim] = im
                    nim += 1
                    frctLst.append(frc)
                    matDirLst.append(matDir)
                    sceneDirLst.append(sceneDir)



                    if nim==Bsize:
                        return BImg,Bmsk, frctLst, matDirLst, sceneDirLst
  ##############################################################################################################

  #                         Load data for a single pair of vessels sequentially and with no augmantation for evaluation

  ########################################################################################################
#     def LoadSingleClean(self,MaxImageSize=1000000): # For Evaluation
#         self.itr += 1
#         print(self.itr)
#         if self.itr >= len(self.AnnList):
#             self.itr = 0
#             self.epoch += 1
# #.............Load next pair entry.........................................................................
#
#         #...................Load image and vessel masks..............................................................
#         Entry = self.AnnList[self.itr]
#         msk1 = cv2.imread(Entry["Mask1"], 0) > 0
#         msk2 = cv2.imread(Entry["Mask2"], 0) > 0
#         img = cv2.imread(Entry["Image"])
#
#         #....................Resize if to big or small...........................................................
#         Hbc = img.shape[0]
#         Wbc = img.shape[1]
#         if np.random.rand() < 0.3:
#             R = np.random.rand() + 0.5
#             Hbc = int(R * Hbc)
#             Wbc = int(R * Wbc)
#         if Wbc * Hbc > MaxImageSize:  # if too big shrink
#             R = (MaxImageSize / Wbc / Hbc) ** 0.5
#             Hbc = int(R * Hbc)
#             Wbc = int(R * Wbc)
#         if np.min([Wbc, Hbc]) < 100:  # if to small resize
#             R = 100 / np.min([Wbc, Hbc])
#             Hbc = int(R * Hbc)
#             Wbc = int(R * Wbc)
# #...............Create batch.................................................................
#         Bmsk1 = np.zeros([1, Hbc, Wbc], np.float32)
#         Bmsk2 = np.zeros([1, Hbc, Wbc], np.float32)
#         BImg = np.zeros([1, Hbc, Wbc, 3], np.float32)
#
#         Bclass = {}
#         for cls in ClassList:
#             Bclass[cls] = np.zeros([1], np.float32)
# #........................Load data into batch..............................................................
#         Bmsk1[0, :, :] = cv2.resize(msk1.astype(np.uint8), (Wbc, Hbc), interpolation=cv2.INTER_NEAREST)
#         Bmsk2[0, :, :] = cv2.resize(msk2.astype(np.uint8), (Wbc, Hbc), interpolation=cv2.INTER_NEAREST)
#         BImg[0] = cv2.resize(img.astype(np.uint8), (Wbc, Hbc), interpolation=cv2.INTER_LINEAR)
#
#         for cls in ClassList: # Load relations into batch classes
#             if Entry[cls]: Bclass[cls][0] = 1
#         return BImg, Bmsk1, Bmsk2, Bclass
# ###########################Load Image to batch
