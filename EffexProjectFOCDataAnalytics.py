# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 13:07:21 2020
This program is developed to analyze the Spindle Data from AFOSR
Project - STTR EFFEX - Funded to Advent Innovations and University of South Carolina
Machine Learning Program - DataFrame
Analysis of FOC data *.csv
@author: Banerjee
"""
# First import the libraries that are required
#%%
#
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk 
import matplotlib.pyplot as plt
import math as mt
import glob

#%matplotlib inline

from sklearn import metrics 
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#%% 
# Loading all the data files in the FOC Folder  
#fileExt="C:\Sourav Working Directory\OneDrive - University of South Carolina\2_Research_2017-2021\ProjectFund_2017-2021\ProjectFunded 2020\AFOSR STTR EFFEX\T4 Data\aaf10f69-1ecf-430d-b5b3-142d72e65a46\FOCAS";

# Getting the list of files in the FOCAS Directory
mylist = [f for f in glob.glob("FOCAS\*.csv")]
# Collecting Number of Files
NumFile=len(mylist)

FOC_df={};
FOCNewData={};
FOCFinalStat={};
# Removing non important data columns - for explanation refer Jupyter code 
for n in range(NumFile):
    FOC_df[n]=pd.read_csv(mylist[n]);
    FOCNewData[n]=FOC_df[n].drop(["DATE",'FRONT IN', 'REAR_IN', 'MACHINE TEMP', 'VIBRATION',
       'SPINDLE COMMAND SPEED','FEED OVERRIDE',"JOG OVERRIDE","RAPID OVERRIDE",
       'RELATIVE1','RELATIVE2','RELATIVE3','RELATIVE4','RELATIVE5','RELATIVE6','RELATIVE7','RELATIVE8',
       'NC MODE', 'SPINDLE POT', 'SPINDLE PTN','MAIN PROGRAM','EXEC PROGRAM','MAIN COMMENT',
       'EXEC BLOCK','SPINDLE FTN','SPINDLE ITN','CUTTING REMOVAL AMOUNT','MSPINDLE LOAD', "SPINDLE DIRECTION","NC STATUS",
       'FRONT OUT', 'REAR OUT', 'VIB-X', 'VIB-Y', 'VIB-Z', 'SPINDLE LOAD',
       'SPINDLE ACTUAL SPEED', 'FEED SPEED', 'FEED ACTUAL SPEED', 'DI_CUT',
       'FEED AXIS STATUS','DISTANCE3', 'DISTANCE4', 'DISTANCE5', 'DISTANCE6', 'DISTANCE7',
       'DISTANCE8','MACHINE ALARM', 'MACHINE WARNING', 'BLOCK COUNTER',
       'EMERGENCY', 'ADDITIONAL INCYCLE STATUS', 'SPINDLE OVERRIDE'],axis=1)
    # Getting Statistics of the Data Columns 
    FOCFinalStat[n]=FOCNewData[n].describe().transpose();
Attribute=FOCNewData[1].columns  # Getting the Remaining Columns 

#%%    
# Collecting the mean value of the ABSOLUTE MACHINE and LOAD data 
MotorCoil={};
NF={};
Abs1={};Abs2={};Abs3={}
Mac1={};Mac2={};Mac3={}
Lod1={};Lod2={};Lod3={}
for n in range(NumFile):
    NF[n]=n+1
    MotorCoil[n]=FOCFinalStat[n].loc['MOTOR COIL','mean']
    Abs1[n]=FOCFinalStat[n].loc['ABSOLUTE1','mean']
    Abs2[n]=FOCFinalStat[n].loc['ABSOLUTE2','mean']
    Abs3[n]=FOCFinalStat[n].loc['ABSOLUTE3','mean']  
    Mac1[n]=FOCFinalStat[n].loc['MACHINE1','mean']
    Mac2[n]=FOCFinalStat[n].loc['MACHINE2','mean']
    Mac3[n]=FOCFinalStat[n].loc['MACHINE3','mean'] 
    Lod1[n]=FOCFinalStat[n].loc['LOAD1','mean']
    Lod2[n]=FOCFinalStat[n].loc['LOAD2','mean']
    Lod3[n]=FOCFinalStat[n].loc['LOAD3','mean'] 
    

#%% 
# Collecting the median value of the ABSOLUTE MACHINE and LOAD data 
MotorCoil={};
NF={};
Abs1={};Abs2={};Abs3={}
Mac1={};Mac2={};Mac3={}
Lod1={};Lod2={};Lod3={}

for n in range(NumFile):
    NF[n]=n+1
    MotorCoil[n]=FOCFinalStat[n].loc['MOTOR COIL','min']
    Abs1[n]=FOCFinalStat[n].loc['ABSOLUTE1','min']
    Abs2[n]=FOCFinalStat[n].loc['ABSOLUTE2','min']
    Abs3[n]=FOCFinalStat[n].loc['ABSOLUTE3','min']  
    Mac1[n]=FOCFinalStat[n].loc['MACHINE1','min']
    Mac2[n]=FOCFinalStat[n].loc['MACHINE2','min']
    Mac3[n]=FOCFinalStat[n].loc['MACHINE3','min'] 
    Lod1[n]=FOCFinalStat[n].loc['LOAD1','min']
    Lod2[n]=FOCFinalStat[n].loc['LOAD2','min']
    Lod3[n]=FOCFinalStat[n].loc['LOAD3','min'] 
    
#%%    
##plt.plot(NF,MeanMotorCoil,color='orange',linewidth=5,linestyle='-')

#Plotting Variation of Motor Coil Temp [C] - A check
plt.scatter(MotorCoil.keys(),MotorCoil.values())
plt.title("Scatter Plot")
plt.xlabel("Days")
plt.ylabel("Variation of Motor Coil Temp [C]")
plt.grid(True)
plt.show()   
 
#%%
#Plotting Variation of ABSOLUTE VIB X - A check
plt.scatter(Abs1.keys(),Abs1.values())
plt.title("Scatter Plot")
plt.xlabel("Days")
plt.ylabel("Variation of ABSOLUTE VIB 1")
plt.grid(True)
plt.show()  
 
#%%
#Plotting Variation of ABSOLUTE VIB Y - A check
plt.scatter(Abs2.keys(),Abs2.values())
plt.title("Scatter Plot")
plt.xlabel("Days")
plt.ylabel("Variation of ABSOLUTE VIB 2")
plt.grid(True)
plt.show() 

#%%
#Plotting Variation of ABSOLUTE VIB Z - A check
plt.scatter(Abs3.keys(),Abs3.values())
plt.title("Scatter Plot")
plt.xlabel("Days")
plt.ylabel("Variation of ABSOLUTE VIB 3")
plt.grid(True)
plt.show() 
#%%
#Plotting Comparison of ABSOLUTE VIB X and Y- A check
plt.scatter(Abs1.values(),Abs2.values())
plt.title("Scatter Plot")
plt.xlabel("ABS 1")
plt.ylabel("ABS 2")
plt.grid(True)
plt.show() 
#%%
#Plotting Comparison of ABSOLUTE VIB Y and LOAD Y- A check
plt.scatter(Abs2.values(),Lod2.values())
plt.title("Scatter Plot")
plt.xlabel("ABS 2")
plt.ylabel("Load 2")
plt.grid(True)
plt.show() 

#%%
#%% PCA ANALYSIS OF MAIN DATA FRAME 
#%%
#%%
# Chronological PCA Analysis 3 Principal Components Of MAIN DATA 
# Data Normalization
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

FOC_MainData_PCA3_df={}
for n in range(NumFile):
    xm = FOCNewData[n].loc[:, :].values
    xm = StandardScaler().fit_transform(xm) # normalizing the features
    where_are_NaNs = np.isnan(xm)
    xm[where_are_NaNs] = 0
    pca_FOC = PCA(n_components=3)
    FOCpca = pca_FOC.fit_transform(xm)
    FOC_MainData_PCA3_df[n] = pd.DataFrame(data = FOCpca,
                                columns = ['principal component 1', 
                                           'principal component 2',
                                           'principal component 3']) 
#%%
# 3D PCA PLOTS from Main Data 

from mpl_toolkits.mplot3d import Axes3D


for n in range(NumFile): 
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    MainTitle='PCA  -  ' 
    DynTitle=mylist[n][6:21]
    NowTtile=MainTitle+DynTitle
    plt.title(NowTtile,fontsize=20)
    ax.scatter(FOC_MainData_PCA3_df[n].loc[:, 'principal component 1']
              ,FOC_MainData_PCA3_df[n].loc[:, 'principal component 2']
              ,FOC_MainData_PCA3_df[n].loc[:, 'principal component 3'], c='r', marker='o')
    ax.set_title(NowTtile,fontsize=10)
    ax.set_xlabel('Principal Component - 1')
    ax.set_ylabel('Principal Component - 2')
    ax.set_zlabel('Principal Component - 3')
    plt.show()
    plt.gcf().clear()
for i in range(0, 360, 45):
    ax.view_init(None, i)
    plt.show()  

#%%
for n in range(NumFile):
    FOC_df[n]['DI_CUT'].replace(0, 'CUT-0',inplace=True)
    FOC_df[n]['DI_CUT'].replace(1, 'CUT-1',inplace=True)     
#%%

# 2D PCA from 3 PCA analysis : PLOTS from Main Data - Clustering w.r.t  DI_CUT

targets = ['CUT-0', 'CUT-1']
colors = ['r', 'g']

from mpl_toolkits.mplot3d import Axes3D


for n in range(NumFile):
    
    
    for target, color in zip(targets,colors):  # Target to fingd CUT-0 or CUT-1  id True
        indicesToKeep = FOC_df[n]['DI_CUT'] == target  # Index Keeps when True
        fig=plt.figure()
        ax=fig.add_subplot(111,projection='3d')
        MainTitle='PCA  -  ' 
        DynTitle=mylist[n][6:21]
        NowTtile=MainTitle+DynTitle
        plt.title(NowTtile,fontsize=20)
        ax.scatter(FOC_MainData_PCA3_df[n].loc[indicesToKeep, 'principal component 1']
                  ,FOC_MainData_PCA3_df[n].loc[indicesToKeep, 'principal component 2'], c = color, s = 50)
                 
        ax.set_title(NowTtile,fontsize=10)
        ax.set_xlabel('Principal Component - 1')
        ax.set_ylabel('Principal Component - 2')
        
        plt.show()
        plt.gcf().clear()

#%%
for i in range(0, 360, 45):
    ax.view_init(None, i)
    plt.show()  


#%%  
# Plotting PCA MAIN Data 3D First 2 PCA 
# 2D PCA from 3 PCA analysis : PLOTS from Main Data - Clustering w.r.t  DI_CUT

targets = ['CUT-0', 'CUT-1']
colors = ['r', 'g']


for n in range(NumFile): 
    for target, color in zip(targets,colors):  # Target to fingd CUT-0 or CUT-1  id True
        indicesToKeep = FOC_df[n]['DI_CUT'] == target  # Index Keeps when True
        plt.figure()
        plt.figure(figsize=(10,10))
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=14)
        plt.xlabel('Principal Component - 1',fontsize=20)
        plt.ylabel('Principal Component - 2',fontsize=20)
        #ax=fig.add_subplot(111,projection='3d')
        MainTitle='PCA  -  ' 
        DynTitle=mylist[n][6:21]
        NowTtile=MainTitle+DynTitle
        plt.title(NowTtile,fontsize=20)
        plt.scatter(FOC_MainData_PCA3_df[n].loc[indicesToKeep, 'principal component 1']
                   ,FOC_MainData_PCA3_df[n].loc[indicesToKeep, 'principal component 2'], c = color, s = 50)
        plt.show()
        plt.gcf().clear()
        

#%%
# Plotting PCA Plot one PCA of MAIN DATA bar plot w.r.t file number 
     #NumAttb=[x for x in range(FOCpca.shape[0])] Required if use plt.bar from matplotlib
     # here seaborn librray is used 
     
for n in range(NumFile):  
    plt.figure(figsize=(10,10))
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=14)
    plt.xlabel('Data Attributes',fontsize=20)
    plt.ylabel('Principal Component - 1',fontsize=20)
    MainTitle='PCA  -  ' 
    DynTitle=mylist[n][6:21]
    NowTtile=MainTitle+DynTitle
    bplot=sns.barplot(x=Attribute,y=FOCpca3_df[n].loc[:, 'principal component 1'])
    plt.title(NowTtile,fontsize=20)
    bplot.set_xticklabels(bplot.get_xticklabels(), rotation=45, fontsize=10);
    plt.show()
    plt.gcf().clear()     

   
#%%






#%% PCA ANALYSIS OF STATISTICAL DATA FRAME 
#%%
#%%
# Chronological PCA Analysis 3 Principal Components Of STATISTICS of DATA COLUMNS 
# Data Normalization

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
FOC_StatData_PCA3_df={}
for n in range(NumFile):
    xs3 = FOCFinalStat[n].loc[:, :].values
    xs3 = StandardScaler().fit_transform(xs3) # normalizing the features
    pca_FOC = PCA(n_components=3)
    FOCpca = pca_FOC.fit_transform(xs3)
    FOC_StatData_PCA3_df[n] = pd.DataFrame(data = FOCpca,
                                columns = ['principal component 1', 
                                           'principal component 2',
                                           'principal component 3']) 
#%%
# 3D PCA PLOTS

from mpl_toolkits.mplot3d import Axes3D


for n in range(NumFile): 
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    MainTitle='PCA  -  ' 
    DynTitle=mylist[n][6:21]
    NowTtile=MainTitle+DynTitle
    plt.title(NowTtile,fontsize=20)
    ax.scatter(FOC_StatData_PCA3_df[n].loc[:, 'principal component 1']
              ,FOC_StatData_PCA3_df[n].loc[:, 'principal component 2']
              ,FOC_StatData_PCA3_df[n].loc[:, 'principal component 3'], c='r', marker='o')
    ax.set_title(NowTtile,fontsize=10)
    ax.set_xlabel('Principal Component - 1')
    ax.set_ylabel('Principal Component - 2')
    ax.set_zlabel('Principal Component - 3')
    plt.show()
    plt.gcf().clear()
for i in range(0, 360, 45):
    ax.view_init(None, i)
    plt.show()    
#%% 
# Chronological PCA 1, 2 3 - Creating Z rendered Data for All Attribute over Time 
# Surface Plot
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
FileNum=[x for x in range(len(mylist))]     #Getting the number of files
NumAttb=[x for x in range(FOCpca.shape[0])] #Getting Names of the Attributes

X, Y = np.meshgrid(FileNum,NumAttb)         # Creating a Mesh Grid

PCA_Statover_time=np.zeros((len(FOCNewData[1].columns),len(mylist)))
for n in range(NumFile):
    PCA_Stat_t =  FOC_StatData_PCA3_df[n].loc[:, :].values
    PCA_Statover_time[:,n]=PCA_Stat_t[:,0]
    
fig = plt.figure()
ax = fig.gca(projection='3d')
   
surf = ax.plot_surface(X, Y, PCA_Statover_time, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)    
#%% 

# 3D PCA Wire Frame Plot Can not be done with Current Vectoriezed Data 
# ERROR: ValueError: Argument Z must be 2-dimensional.

# from mpl_toolkits.mplot3d import Axes3D


# for n in range(NumFile): 
#     fig=plt.figure()
#     ax=plt.axes(projection='3d')
#     MainTitle='PCA  -  ' 
#     DynTitle=mylist[n][6:21]
#     NowTtile=MainTitle+DynTitle
#     plt.title(NowTtile,fontsize=20)
#     ax.plot_wireframe(FOCpca3_df[n].loc[:, 'principal component 1']
#                      ,FOCpca3_df[n].loc[:, 'principal component 2']
#                      ,FOCpca3_df[n].loc[:, 'principal component 3'], c='r', marker='o')
#     ax.set_title(NowTtile,fontsize=10)
#     ax.set_xlabel('Principal Component - 1')
#     ax.set_ylabel('Principal Component - 2')
#     ax.set_zlabel('Principal Component - 3')
#     plt.show()
#     plt.gcf().clear()
# for i in range(0, 360, 45):
#     ax.view_init(None, i)
#     plt.show()    
   #%% 
# Chronological PCA 1, 2 3 - Creating Z rendered Data for All Attribute over Time 
# 3D Bar Plot

# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
# FileNum=[x for x in range(len(mylist))]
# NumAttb=[x for x in range(FOCpca.shape[0])]
# X, Y = np.meshgrid(FileNum,NumAttb)
# PCA_over_time=np.zeros((len(FOCNewData[1].columns),len(mylist)))
# for n in range(NumFile):
#     PCA_t =  FOCpca3_df[n].loc[:, :].values
#     PCA_over_time[:,n]=PCA_t[:,0]
    
# fig = plt.figure()
# ax = fig.gca(projection='3d')
   
# surf = ax.bar3d(FOCpca.shape[0], len(mylist), PCA_over_time, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)    
#%%  
# Plotting 2D PCA Over time - i.e Over the Range of the FOC Files 

 for n in range(NumFile):  
     plt.figure()
     plt.figure(figsize=(10,10))
     plt.xticks(fontsize=12)
     plt.yticks(fontsize=14)
     plt.xlabel('Principal Component - 1',fontsize=20)
     plt.ylabel('Principal Component - 2',fontsize=20)
     MainTitle='PCA  -  ' 
     DynTitle=mylist[n][6:21]
     NowTtile=MainTitle+DynTitle
     plt.title(NowTtile,fontsize=20)
     plt.scatter(FOC_StatData_PCA3_df[n].loc[:, 'principal component 1']
                ,FOC_StatData_PCA3_df[n].loc[:, 'principal component 2'])
     plt.show()
     plt.gcf().clear()
#%%
# Plotting 2D PCA Plot only the one but First PCA bar plot w.r.t file number 
     #NumAttb=[x for x in range(FOCpca.shape[0])] Required if use plt.bar from matplotlib
     # here seaborn librray is used 
     
for n in range(NumFile):  
    
    plt.figure(figsize=(10,10))
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=14)
    plt.xlabel('Data Attributes',fontsize=20)
    plt.ylabel('Principal Component - 1',fontsize=20)
    MainTitle='PCA  -  ' 
    DynTitle=mylist[n][6:21]
    NowTtile=MainTitle+DynTitle
    bplot=sns.barplot(x=Attribute[:],y=-FOC_StatData_PCA3_df[n].loc[:, 'principal component 1'])
    plt.title(NowTtile,fontsize=20)
    bplot.set_xticklabels(bplot.get_xticklabels(), rotation=45, fontsize=10);
    plt.show()
    plt.gcf().clear()
#%%    
# Plotting 2D PCA Plot only the one but First PCA bar plot w.r.t file number 
     #NumAttb=[x for x in range(FOCpca.shape[0])] Required if use plt.bar from matplotlib
     # here seaborn librray is used 
     
for n in range(NumFile):  
    
    plt.figure(figsize=(10,10))
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=14)
    plt.xlabel('Data Attributes',fontsize=20)
    plt.ylabel('Principal Component - 2',fontsize=20)
    MainTitle='PCA  -  ' 
    DynTitle=mylist[n][6:21]
    NowTtile=MainTitle+DynTitle
    bplot=sns.barplot(x=Attribute[:],y=-FOC_StatData_PCA3_df[n].loc[:, 'principal component 2'])
    plt.title(NowTtile,fontsize=20)
    bplot.set_xticklabels(bplot.get_xticklabels(), rotation=45, fontsize=10);
    plt.show()
    plt.gcf().clear()
#%%# Plotting 2D PCA Plot only the one but First PCA bar plot w.r.t file number 
     #NumAttb=[x for x in range(FOCpca.shape[0])] Required if use plt.bar from matplotlib
     # here seaborn librray is used 

MainTitle='PCA  -  ' 
PCA1=MainTitle+'1'
PCA2=MainTitle+'2'
pclabel=[PCA1,PCA2]
     
for n in range(NumFile):  
    fig, axs = plt.subplots(2)#, sharex=True, sharey=True, gridspec_kw={'hspace': 0})
    # plt.figure(figsize=(10,10))
    # plt.xticks(fontsize=8)
    # plt.yticks(fontsize=14)
    # plt.xlabel('Data Attributes',fontsize=20)
 
    DynTitle=mylist[n][6:21]
    NowTtile=MainTitle+DynTitle

    fig.suptitle(NowTtile,fontsize=20)
    
    axs[0]=sns.barplot(x=Attribute,y=-FOC_StatData_PCA3_df[n].loc[:, 'principal component 1'])
    axs[1]=sns.barplot(x=Attribute,y=-FOC_StatData_PCA3_df[n].loc[:, 'principal component 2'])
    
    axs[1].set_xticklabels(bplot.get_xticklabels(), rotation=60, fontsize=9);
   
    
    # for ax in axs.flat:
    #     ax.set_xticklabels(axs[ax].get_xticklabels(), rotation=45, fontsize=10);
    #     #ax.label_outer(xlabel=pclabel[ax], ylabel='y-label')
   
    
  
    
    plt.show()
    #plt.gcf().clear()

#%%  
# Plotting 2D Scatter Plot of PCA PCA-2 vs PCA-3 over the range of the FOC files

 for n in range(NumFile):  
     plt.figure()
     plt.figure(figsize=(10,10))
     plt.xticks(fontsize=12)
     plt.yticks(fontsize=14)
     plt.xlabel('Principal Component - 2',fontsize=20)
     plt.ylabel('Principal Component - 3',fontsize=20)
     MainTitle='PCA  -  ' 
     DynTitle=mylist[n][6:21]
     NowTtile=MainTitle+DynTitle
     plt.title(NowTtile,fontsize=20)
     plt.scatter(FOC_StatData_PCA3_df[n].loc[:, 'principal component 2']
                ,FOC_StatData_PCA3_df[n].loc[:, 'principal component 3'])
     plt.show()
     plt.gcf().clear()
     
#%%  
# Plotting 2D Scatter Plot of PCA PCA-1 vs PCA-3 over the range of the FOC files

 for n in range(NumFile):  
     plt.figure()
     plt.figure(figsize=(10,10))
     plt.xticks(fontsize=12)
     plt.yticks(fontsize=14)
     plt.xlabel('Principal Component - 1',fontsize=20)
     plt.ylabel('Principal Component - 3',fontsize=20)
     MainTitle='PCA  -  ' 
     DynTitle=mylist[n][6:21]
     NowTtile=MainTitle+DynTitle
     plt.title(NowTtile,fontsize=20)
     plt.scatter(FOCpca3_df[n].loc[:, 'principal component 1']
                ,FOCpca3_df[n].loc[:, 'principal component 3'])
     plt.show()
     plt.gcf().clear()



    
