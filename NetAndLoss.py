# Model and Loss functions
import torchvision.models as models
import torch
import copy
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
class Net(nn.Module):# Net for region based segment classification
######################Load main net (resnet 50) class############################################################################################################
        def __init__(self): # Load pretrained encoder and prepare net layers
            super(Net, self).__init__()
# ---------------Load pretrained net----------------------------------------------------------
            self.Encoder1 = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)#resnet50(pretrained=True)
           # self.Encoder1 = models.convnext_large(weights=True)  # resnet50(pretrained=True)
            # weight_shape = torch.tensor(self.Encoder1.features[0][0].weight.shape)
            # weight_shape[1] = 1
            # mask_weight=torch.zeros(tuple(weight_shape.numpy()))
            # total_weight= torch.cat([self.Encoder1.features[0][0].weight.data, mask_weight], 1)

#--------------Replace First layer from 3 channel input (RGB) to 4 channel (RGB,ROI)
            old_bias = copy.deepcopy(self.Encoder1.features[0][0].bias.data)
            old_weight = copy.deepcopy(self.Encoder1.features[0][0].weight.data)
            self.Encoder1.features[0][0]= torch.nn.Conv2d(4, 128, kernel_size=(4, 4), stride=(4, 4))
            self.Encoder1.features[0][0].weight.data[:,:3,:,:] = old_weight
            self.Encoder1.features[0][0].weight.data[:, 3, :, :] = 0
            self.Encoder1.features[0][0].bias.data = old_bias
            print("new_bias", self.Encoder1.features[0][0].bias.data.sum())
            print("new_weight", self.Encoder1.features[0][0].weight.data.sum())
#----------------Change final layer to predict 512 descriptor------------------------------------------------------------------------------------------
            #self.Encoder1.fc=nn.Sequential(nn.Linear(2048, 512),nn.ReLU())
            self.Encoder1.classifier[2]=torch.nn.Linear(in_features=1024, out_features=512, bias=True)


#
        ###############################################Run prediction inference using the net ###########################################################################################################
        def forward(self,Images1, Mask1,TrainMode=True):

#------------------------------- Convert from numpy to pytorch-------------------------------------------------------
                if TrainMode:
                    mode=torch.FloatTensor
                else:
                    mode=torch.half

                self.type(mode)
                InpImages1 = torch.autograd.Variable(torch.from_numpy(Images1), requires_grad=False).transpose(2,3).transpose(1, 2).type(mode)
                ROIMask1 = torch.autograd.Variable(torch.from_numpy(Mask1.astype(np.float32)), requires_grad=False).unsqueeze(dim=1).type(mode)
                InpImages1 = InpImages1.to(device)
                ROIMask1 = ROIMask1.to(device)
                self.to(device)

# -------------------------Normalize image-------------------------------------------------------------------------------------------------------
                RGBMean = [123.68, 116.779, 103.939]
                RGBStd = [65, 65, 65]
                for i in range(len(RGBMean)):
                    InpImages1[:, i, :, :] = (InpImages1[:, i, :, :] - RGBMean[i]) / RGBStd[i]  # Normalize image by std and mean
#============================Run net layers===================================================================================================
                x=torch.cat([InpImages1, ROIMask1],1)
                descriptor = self.Encoder1(x) # run net
                descriptor = F.normalize(descriptor, dim=1)
                return descriptor


###########################################################################################################################################################3
        '''cross entropy cosine loss between all descriptor in the batch'''
        def SoftLossCosineSimilarity(self, Desc1, DescFract,DescFolder, temp=1):
                    ###Desc1 = F.normalize(Desc1, dim=1) already normalized
                    # Create GT and predicted matrixes of similartiy between  vectors in the batch
                    prd_correlation= {}

                    GT_correlation = {}
                    for i1 in range(Desc1.shape[0]): # (einsum)
                        for i2 in range(Desc1.shape[0]):
                            prd_correlation[(i1,i2)] = (Desc1[i1] * Desc1[i2]).sum()
                            if DescFolder[i1]==DescFolder[i2]:
                                GT_correlation[(i1,i2)]=1-np.abs(DescFract[i1]-DescFract[i2])
                            else:
                                GT_correlation[(i1,i2)]=0

                    # Calculate loss
                    Loss=0
                    tpfp = {} # Use to store statistics about true and false postives
                    for i in range(0,5):
                         tpfp[i/4]=[0,0]
                    numSamp=0
                    for k1 in range(Desc1.shape[0]):   # check any triplet in of descriptor with anchor negative positive mode
                        for k2 in range(Desc1.shape[0]):
                            if k1==k2: continue
                            if GT_correlation[k1,k2]>0:
                                for k3 in range(Desc1.shape[0]):
                                        if k3<=k1 and  GT_correlation[(k1, k3)]>0: continue # already checked
                                        corl_dif = GT_correlation[(k1, k2)] - GT_correlation[(k1, k3)]
                                        margin=corl_dif/4
                                        if corl_dif!=0: # cant apply loss if all examples are the same distance
                                             logits = torch.cat([prd_correlation[(k1,k2)].unsqueeze(0), prd_correlation[(k1,k3)].unsqueeze(0)], 0) / temp
                                             prob = F.softmax(logits, dim=0)
                                             #if abs(corl_dif)==1 or (prob[0]>0.5) or  (corl_dif > 0 and (prob[1])>0.5):
                                             if (corl_dif < 0 and (prob[0]+margin)>0.5) or  (corl_dif > 0 and (prob[1]+margin)>0.5):
                                                 numSamp+=1

                                                 if corl_dif>0: Loss += -torch.mean((torch.log(prob[0] + 0.0000001)))  # Calculate cross entropy loss
                                                 if corl_dif<0: Loss += -torch.mean((torch.log(prob[1] + 0.0000001)))  # Calculate cross entropy loss
                                             if    (corl_dif<0 and logits[1]>logits[0]) or (corl_dif>0 and logits[0]>logits[1]):
                                                 tpfp[abs(corl_dif)][0] += 1
                                             else:
                                                 tpfp[abs(corl_dif)][1] += 1
                    Loss/=numSamp+0.0000001
                    if Loss == 0: Loss = torch.sum(Desc1 * 0)  # To prevent crash *its complicated :)
                    return  Loss,tpfp
