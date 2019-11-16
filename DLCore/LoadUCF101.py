import __future__
import cv2
import numpy as np

def LoadlUCF101_OnsetAligned(DataDir,MinFrames=50,FrameStride=1,MinTrainingPerClass=70,Height=80,Width=112,CategoryFilter='All'):
	# Load the UCF101 Data downsampled by 2

  Nframe=int(MinFrames/FrameStride)

  TrainH=open(DataDir+'ucfTrainTestlist/trainlist03.txt','r')
  TestH=open(DataDir+'ucfTrainTestlist/testlist03.txt','r')
  ClassesH=open(DataDir+'ucfTrainTestlist/classInd.txt','r')

  TrainFiles=TrainH.readlines()
  TestFiles=TestH.readlines()
  Classes=ClassesH.readlines()
  TrainH.close()
  TestH.close()
  ClassesH.close()
  
  ClassDict={}
  for l in Classes:
      ClassDict[l.split(' ')[1][:-1]]=int(l.split(' ')[0])
      
  
  SportFilter  = [12,26,36,42,44,49,71,74,76,84]##Easily detectable
  MakeupFilter = [1,2,13,20,34,39,78]
  WorkoutFilter= [10,15,21,37,38,47,48,52,56,70,72,84,99]
  MusicFilter  = [27,59,60,61,62,63,64,65,66,67]
  TrackFilter  = [36,40,45,51,68,93] 
  

  if CategoryFilter == 'Sport':  
    SelFilter=SportFilter;
  elif CategoryFilter == 'Makeup':  
    SelFilter=MakeupFilter;
  elif CategoryFilter == 'Workout':  
    SelFilter=WorkoutFilter;
  elif CategoryFilter == 'Music':  
    SelFilter=MusicFilter;
  elif CategoryFilter == 'Track':  
    SelFilter=TrackFilter;
  elif CategoryFilter == 'All':  
    SelFilter=range(1,102);
  
  SamplePerClass={}
  
  Train_select_all=[]
  for sample in range(len(TrainFiles)):
      vid=cv2.VideoCapture(DataDir+'UCF-101/'+TrainFiles[sample].split(' ')[0])
      Nf=int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
      cind=int(TrainFiles[sample].split(' ')[1])
      if (Nf>MinFrames and (cind in SelFilter)): 
          if cind in SamplePerClass:
              if SamplePerClass[cind]==MinTrainingPerClass:
                   continue
              SamplePerClass[cind] +=1
          else:
              SamplePerClass[cind] = 1
          Train_select_all.append((sample,cind))
              
  vid.release()
  
  ViableClasses= [i for i in SamplePerClass if (SamplePerClass[i]==MinTrainingPerClass)]
  Train_select_viable=[i for (i,j) in Train_select_all if j in ViableClasses] 
  
  Test_select_all=[]
  TestSamplePerClass={}
  for sample in range(len(TestFiles)):
      cind=ClassDict[TestFiles[sample].split('/')[0]]
      if cind in ViableClasses:
          vid=cv2.VideoCapture(DataDir+'UCF-101/'+TestFiles[sample].split(' ')[0][:-1])
          Nf=int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
          if Nf>=MinFrames:
              Test_select_all.append((sample,cind))
              if cind in TestSamplePerClass:
                  TestSamplePerClass[cind] +=1
              else:
                  TestSamplePerClass[cind] = 1
              
  vid.release()      
  
  N_train=len(Train_select_viable)
  N_val=len(Test_select_all)
  
  #Shuffle Training Set
  idx=np.random.permutation(len(Train_select_viable));
  Train_select=[Train_select_viable[i] for i in idx] 
  
  
  
  X_train=np.zeros((N_train,Nframe,Height,Width,3),dtype='float32')
  N_train,F,W,H,C=X_train.shape
  y_train=np.zeros(N_train)
  
  for i in range(N_train):
      sample=Train_select[i]
      vid=cv2.VideoCapture(DataDir+'UCF-101/'+TrainFiles[sample].split(' ')[0])
      y_train[i]=ViableClasses.index(int(TrainFiles[sample].split(' ')[1]))
      for f in range(MinFrames):
          ret,frame=vid.read()  
          if ret==True:
              if (f%FrameStride==0):
                  X_train[i,int(f/FrameStride),:,:,:]=cv2.pyrDown(frame)[60-int(Height/2):60+int(Height/2),80-int(Width/2):80+int(Width/2),:]
          else:
              break
      vid.release()
  
      
  
  X_val=np.zeros((N_val,Nframe,Height,Width,3),dtype='float32')
  N_val,F,W,H,C=X_val.shape
  y_val=np.zeros(N_val)
  
  for i in range(N_val):
      (sample,cind)=Test_select_all[i]
      vid=cv2.VideoCapture(DataDir+'UCF-101/'+TestFiles[sample].split(' ')[0][:-1])
      y_val[i]=ViableClasses.index(cind)
      for f in range(MinFrames):
          ret,frame=vid.read()  
          if ret==True:
              if (f%FrameStride==0):
                  X_val[i,int(f/FrameStride),:,:,:]=cv2.pyrDown(frame)[60-int(Height/2):60+int(Height/2),80-int(Width/2):80+int(Width/2),:]
          else:
              break
      vid.release()
  
  X_train /= 255
  X_val /= 255    
  X_train -= np.mean(X_train,axis=(0,1))
  X_val -= np.mean(X_val,axis=(0,1))
  
  ###Just to make is consistent with CIFAR data
  X_train=X_train.transpose(0,1,4,2,3)
  X_val=X_val.transpose(0,1,4,2,3)
  return {
    'X_train': X_train,
    'y_train': y_train,
    'X_val': X_val,
    'y_val': y_val,
  }
