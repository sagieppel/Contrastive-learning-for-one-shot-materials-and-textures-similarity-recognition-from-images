# Evaluate on MatSim benchmark
# Receive a file json with descriptors arrange in a tree strcutres (that replicate input folder and subfolder structre).
# Assume all descriptor in the same subfolder are of the same subclass
#...............................Imports..................................................................
import os
import random
import json
import numpy as np
import argparse
##os.system("cls")
##################################Input paramaters#########################################################################################
parser = argparse.ArgumentParser(description = 'take descriptor files generate by RunNetOnFolder and get accuracy statitics')
parser.add_argument('--descriptor_file', default="logs/descriptor_dict.json", type=str, help=' input json file with descriptors, generatee by RunNetOnFolder.py')
parser.add_argument('--test_set', default=  1 , type = int, help="Benchmarks set Set 1 subclass and states , Set 2. Random material on random on random objects")
parser.add_argument('--mode', default=  "all" , type = str, help='subclass/all "subclass": Match to only to images of the same superclass, "all" match between all images in the set')
parser.add_argument('--desc_length', default=  512 , type = int, help='descriptor length')
parser.add_argument('--random_match', default=  False , type = bool, help='replace descriptor with random  numbers for testing')
args = parser.parse_args()

# TestSet=1 # Benchmarks set" Set 1 subclass and states , Set 2. Random material on random on random objects
# mode= "all" # subclass/all "subclass": Match to only to images of the same superclass, "all" match between all images in the set
#
#
# # input json file with descriptors, generatee by RunNetOnFolder.py
# InpuFile =   r"logs/descriptor_dict.json"
# desc_length = 512


##############################################################################
if args.test_set==1:
    tree_depth=2#  the descriptor are arrange in tree structure were each layer is a more fine grain class, 1 mean single layer (one class per descriptor) 2 means 2 layer superclas+ subclass per descriptor
elif args.test_set==2:
    tree_depth = 1  #
    args.mode = "subclass" # Set 2


###RandomMatch=False # Replace descriptor with random descriptor
'''
subclass= compare subclass within the same super class,
all= compare all sub class to all subclass 
superclass = compare superclass to other super class
Note: superclasses are first order branches/folders subclass are second order folders
'''
###########################################################################################################################
'''
The following 
'''

def createDescList_1stOrderCoarseGrain_SuperClass(data,RandomMatch=False):
    '''Take the tree structure of the dictionary and replace it with flatt list of image descriptors and their class.
    Use bottum second order sub folders as main class and compare everyfolder to every folder.
    note that image of the same folder belong to the same class, and the lower the subfolder the more specific is the class'''
    descs = [] # list of all descriptors (for each decriptor also have its cat and other related data)
    descCat = {} # List of all descriptors divided by class
    for nm in data["dirs"]:# Use go over all first 2nd subfolders, use sub first order folder as class (but read files from second order subclass_
        subdirs1 = data["dirs"][nm]["dirs"] # second order file list
        descCat[nm] = {"sum_desc": np.zeros(args.desc_length), "num_samples": 0}
        for sdir1 in subdirs1: descCat[nm]["num_samples"]+=len(subdirs1[sdir1]["descs"]) # number of images of the same class
        for sdir1 in subdirs1:
            subdir2=subdirs1[sdir1]["descs"]

            for file_name in subdir2:
               descriptor = subdir2[file_name][1:-1]
               if type(descriptor) == str:
                        descriptor= np.fromstring(descriptor, dtype=np.float32, sep=', ') # get descriptor
               if RandomMatch:   descriptor= np.random.rand(args.desc_length)#
               descs.append({"file": file_name, "class": nm, "descs": descriptor, "num_samples": descCat[nm]["num_samples"],"id": len(descs)})
               descCat[nm]["sum_desc"] += descriptor # average descriptor for class
    return  descs#,descCat


# Put all the descriptors in one folder
def createDescList_2dOrderFineGrainAll(data,RandomMatch=False):
    '''Take the tree structure of the dictionary and replace it with flatt list of descriptor and their class.
    use bottum third order sub folders as main class and compare everyfolder to every folder.
    note that image of the same folder belong to the same class, and the lower the subfolder the more specific is the class'''
    descs = []  # list of all descriptors (for each decriptor also have its cat and other related data)
    descCat = {}  # List of all descriptors divided by class
    for nm in data["dirs"]:
        subdirs1 = data["dirs"][nm]["dirs"]


        for sdir1 in subdirs1:
            descCat[nm+sdir1] = {"sum_desc": np.zeros(args.desc_length), "num_samples": len(subdirs1[sdir1]["descs"])}
            subdir2=subdirs1[sdir1]["descs"]

            for file_name in subdir2:
               descriptor = subdir2[file_name][1:-1]
               if type(descriptor) == str:
                      descriptor= np.fromstring(subdir2[file_name][1:-1], dtype=np.float32, sep=', ') #
               if RandomMatch:  descriptor= np.random.rand(args.desc_length)#
               descs.append({"file": file_name, "class": nm+sdir1, "descs": descriptor, "num_samples": descCat[nm+sdir1]["num_samples"],"id": len(descs)})
               descCat[nm+sdir1]["sum_desc"] += descriptor
    return  descs#,descCat
#####################################################################################################################3
# Put all the descriptors in one folder
def createDescListSubClass(data,RandomMatch=False):
    '''Take the tree structure of the dictionary and replace it with flatt list of descriptor and their class.
    use bottum third order sub folders as main class but compare it only to image of the same coarse grain class.
    Hence only compare images belonging to the same super cclass
    note that image of the same folder belong to the same class, and the lower the subfolder the more specific is the class'''
    descs = []  # list of all descriptors (for each decriptor also have its cat and other related data)
 #   descCat = {}  # List of all descriptors divided by class
    for nm in data["dirs"]: # Use go over all subfolders, use sub folder as class
        subdir = data["dirs"][nm]["descs"]
     #   descCat[nm] = {"sum_desc": np.zeros(1536), "num_samples": len(subdir)}
        for file_name in subdir: # Get descriptor for every file in subfolder
            descriptor = subdir[file_name][1:-1]
            if  type(descriptor) == str:
                   descriptor= np.fromstring(descriptor, dtype=np.float32, sep=', ') #
            if RandomMatch:  descriptor= np.random.rand(args.desc_length)#
            descs.append({"file": file_name, "class": nm, "descs": descriptor, "num_samples": len(subdir),"id": len(descs)})
          #  descCat[nm]["sum_desc"] += descriptor #use sub folder as class
    return  descs#,descCat

#############################################################################3
def  createMatchList(descs):
       '''Match all descriptor to all other descriptor and create for each image descriptor similarity score for all other descriptor'''
       random.shuffle(descs)
       for i1 in range(len(descs)):
            descs[i1]["matches"]=[]
            for i2 in range(len(descs)):
                if i1!=i2:
                    sim=((descs[i1]["descs"]*descs[i2]["descs"]).sum())

                    # sim1 = ((descs[i1]["descs"][1536:] * descs[i2]["descs"][1536:]).sum())
                    # sim2 = ((descs[i1]["descs"][0:1536] * descs[i2]["descs"][0:1536]).sum())*5
                    # sim= sim1+sim2


                    dic1 = {"sim": sim, "class": descs[i2]["class"], "name": descs[i2]["file"]}
                    descs[i1]["matches"].append(dic1)
       return descs

#******************************************************************************************
# def createClassCenterList(descs,descCat):
#        for i1 in range(len(descs)):
#            descs[i1]["class_center_matches"] = []
#            dc = descs[i1]
#            for nm in descCat:
#                sumDesc = descCat[nm]["sum_desc"]+0
#
#                if dc["class"]==nm:
#                    sumDesc-=dc["descs"]
#                    sumDesc /= (descCat[nm]["num_samples"]-1)
#                else:
#                    sumDesc /= (descCat[nm]["num_samples"])
#
#                sim=(sumDesc*dc["descs"]).sum()
#                descs[i1]["class_center_matches"].append({"class":nm,"sim":sim})
#        return  descs
#########################################################
def count_matches(descs,votes):
       stat_fl = {}
     #  stat_cat = {}

       tp = 0
       fn = 0

       tp_class={}
       fn_class={}
       # create statitics by cat and by file
       for i1 in range(len(descs)):
           matches = sorted(descs[i1]["matches"], key=lambda x: x['sim'],reverse=True) # sort file matches by similarity
      ####     class_matches = sorted(descs[i1]["class_center_matches"], key=lambda x: x['sim'],reverse=True)
           # get to match class
      # get predicted category and top category
           real_cat = descs[i1]["class"]
           mxsample = descs[i1]["num_samples"]
           stat_fl[i1] = {"num_samples": mxsample}
           if votes == -1 or votes > mxsample: votes = mxsample
           prd_cats = []
           for iii in range(votes):
               prd_cats.append(matches[iii]["class"])

           top_prd_cat = max(set(prd_cats), key=prd_cats.count)
      # file info tp is wether the predicted and real cat match
           st = {"tp": top_prd_cat == real_cat, "real_cat": real_cat, "prd_cat": top_prd_cat, "max_sample": mxsample}
      # Get statitics
           stat_fl[i1] = st
           if not real_cat in tp_class:
                tp_class[real_cat]=0
                fn_class[real_cat]=0

           if st["tp"]:
               tp += 1
               tp_class[real_cat]+=1
           else:
               fn += 1 # false prediction
               fn_class[real_cat]+=1


       averge_accur_class=0
       for ct in tp_class:
           acr=tp_class[ct] / (fn_class[ct] + tp_class[ct])
           averge_accur_class+=acr
           print("-",ct,acr)
       averge_accur_class/=len(list(tp_class))
       print("average accuracy by equal class ",averge_accur_class)


       return stat_fl,  averge_accur_class
##############################################################3
def GetStatistics(data,votes,mode,RandomMatch=False):
   '''get statics of accuracy '''
# arrange descriptor in list structure
   if mode=="subclass":#all, superclass
          descs =createDescListSubClass(data,RandomMatch)
   if mode=="all":#all, superclass
          descs =createDescList_2dOrderFineGrainAll(data,RandomMatch)
   if mode == "superclass":  # all, superclass
       descs = createDescList_1stOrderCoarseGrain_SuperClass(data,RandomMatch)


#   match with other descriptors and get similarity score
   descs=createMatchList(descs)

#
# #       match with class average
#    descs=createClassCenterList(descs, descCat)



# Sort dictionary and get statitics
   stat_fl, acurFile = count_matches(descs,votes)
   return  stat_fl, acurFile

###################################################################################################################
if __name__ == "__main__":
    with open(args.descriptor_file) as fp2:
        data = json.load(fp2)
      #  print(data)
        st=[]
        sum_mean = 0
        #********************************
        for i in range(1):
            if  (args.mode=="superclass" or args.mode=="all" and tree_depth==2) or (args.mode=="subclass" and tree_depth==1):
               stat_fl,  acurFile = GetStatistics(data, votes=1,mode=args.mode,RandomMatch=args.random_match)
               print("Accuracy by image", acurFile)#, " Accuracy by cat", acurCats)
               sum_mean += acurFile
            #*************************************
            else:
                for i in range(1):
                    for nm in data["dirs"]:
                        stat_fl, acurFile  =GetStatistics(data["dirs"][nm],votes=1,mode=args.mode,RandomMatch=args.random_match)
                        st.append(acurFile)
                        print("-----------scene:",nm,"-------------------------------------------------")
                        print("Accuracy by image",acurFile)

                    print("mean ",sum(st)/len(st))
                    sum_mean+=sum(st)/len(st)
        print("\n\n Average accuracy for all class:",sum_mean/1)
        print(sum_mean/1)
        #
        # print("mean ", sum(stCats) / len(st))
        #     # print(stat_cat)
        #     # statitics[nm]=stat_cat
