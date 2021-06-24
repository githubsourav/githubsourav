# -*- coding: utf-8 -*-
"""
First Created on Tue Sep 15 09:45:04 2020

@author: BANERJES
"""
# -*- coding: utf-8 -*-
"""
Originally Created on Mon Jul  6 13:07:21 2020
This program is developed to analyze the Spindle Data from AFOSR
Project - STTR EFFEX - Funded to Advent Innovations and University of South Carolina
Machine Learning Program - DataFrame
Analysis of FOC data *.csv
@author: Banerjee

This Program Particularly Analyze T4 Data from T4 #1 and #2 Machine  


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

#from kneed import KneeLocator 
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from numpy import gradient
from scipy import signal
from scipy.fft import fftshift
from statistics import median
from sklearn import metrics 
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

#%% 
# Loading all the data files in the FOC Folder  
#fileExt="C:\Sourav Working Directory\OneDrive - University of South Carolina\2_Research_2017-2021\ProjectFund_2017-2021\ProjectFunded 2020\AFOSR STTR EFFEX\T4 Data\aaf10f69-1ecf-430d-b5b3-142d72e65a46\FOCAS";

# Getting the list of files in the FOCAS Directory
mylist = [f for f in glob.glob("T4 Data AFOSR\T4#2\Data\FOCAS\*.csv")]
#mylist = [f for f in glob.glob("T4 Data AFOSR\T4\T4\T4#2\Data\FOCAS\*.csv")]
# mylist = [f for f in glob.glob("T4 Data AFOSR\T4#1\Data\FOCAS\*.csv")]
#mylist = [f for f in glob.glob("T4 Data AFOSR\T4#1\Data\FOCAS\*.csv")]

#T4#2 Codes
ALARM_CODES = np.array([70008, 360004, 360006, 590021, 590025, 590040, 590041])
#T4#1 Codes
#ALARM_CODES = np.array([70008, 70273, 360002, 360004, 590039, 590041])

# Collecting Number of Files
NumFile=len(mylist)



#%% Here we Load all the Data files and remove the unnecessary columns Then calculate the STATS of each file
#%%  

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
       'FRONT OUT', 'REAR OUT', 'SPINDLE LOAD',
       'SPINDLE ACTUAL SPEED', 'FEED SPEED', 'FEED ACTUAL SPEED', 'DI_CUT',
       'FEED AXIS STATUS','DISTANCE3', 'DISTANCE4', 'DISTANCE5', 'DISTANCE6', 'DISTANCE7',
       'DISTANCE8','SEQ NUMBER','MACHINE WARNING','BLOCK COUNTER', 
       'EMERGENCY', 'ADDITIONAL INCYCLE STATUS', 'SPINDLE OVERRIDE'],axis=1)
    # Getting Statistics of the Data Columns 
    FOCFinalStat[n]=FOCNewData[n].describe().transpose();
    FOCFinalStat[n].fillna(0)
Attribute=FOCNewData[1].columns  # Getting the Remaining Columns 
AttributeGmm=Attribute.drop('MACHINE ALARM')
StatParamNames=FOCFinalStat[n].columns

for n in range(NumFile):
    FOCFinalStat[n].fillna(0)

#%% Test if MACHINE ALARM in the DataFrame has the Specific ALARM Codes 
#%%
# CODES [70008, 360004, 360006, 590021, 590025, 590040, 590041]

AlarmFileNo=[0]
for n in range(NumFile):
    IsAlarm=FOCNewData[n].loc[:,'MACHINE ALARM']
    for code in range(len(ALARM_CODES)):
        if (ALARM_CODES[code] in IsAlarm.values):
            AlarmFileNo=np.hstack((AlarmFileNo,n));
## Created a vector of file numbers where ALARMS were Recorded


#%% Create a DateFrame with StatMat Values of each attribute for all files 
#%%
#   with a Binary Index 0 or 1 - to Identify if MACHINE ALARM was positive or mathes with the CODES 

StatParam='mean'   # Select any Stat-attribute 'count' 'mean'  'std' 'min' '25%' '50%' '75' 'max'
StatMat=np.full([NumFile,len(Attribute)],0)

for n in range(NumFile):
    for a in range(len(Attribute)):
        FOCFinalStat[n].fillna(0)
        AttbOfInterest=Attribute[a]
        m=FOCFinalStat[n].loc[AttbOfInterest,StatParam]   

        # if np.isnan(m):
        #     m=0;
        if (AttbOfInterest=='MACHINE ALARM') & (m>0):   # ALL WARNINGS AND ALARMS 
            m=1
        StatMat[n,a]=m
      
        if ((AttbOfInterest=='MACHINE ALARM') & (n in AlarmFileNo)): # ONLY SEVERE SPINDLE ALARMS
           m=2
        StatMat[n,a]=m


minAtb=np.amin(StatMat,axis=0)
maxAtb=np.amax(StatMat,axis=0)

counts = np.unique(StatMat[:,30], return_counts=True)
howmanyAlarm=np.count_nonzero(StatMat[:,30] == 2)
howmanyWarning=np.count_nonzero(StatMat[:,30] == 1)


#%% This graph is to show how each ATTRIBUTE chnages over time and identify when the ALARM was Recorded 
#%%
#FilesToBePloted=NumFile
FilesToBePloted=500
FileNo=np.arange(1,FilesToBePloted+1)   # Creating an array of file no. for plotting 
AlarmOn=1
WarningOn=0

fileindexAlarm=np.full([howmanyAlarm,100],0)
fileindexWarning=np.full([howmanyWarning,100],0)

alarmline=np.full([howmanyAlarm,100],0)
alarm=-1
warning=-1

# Loop to create fileindex vector for each alarm state 
for n in range(FilesToBePloted):
        if (StatMat[n,30]==2):   # Attribute 30 is MACHINE ALARM
           alarm=alarm+1
           fileindexAlarm[alarm]=np.ones((1,100))*n
        if (StatMat[n,30]==1):   # Attribute 30 is MACHINE ALARM
           warning=warning+1
           fileindexWarning[warning]=np.ones((1,100))*n

# Loop for plotting 
for a in range(len(Attribute)):
    if (maxAtb[a]==minAtb[a]):
        maxAtb[a]=1
        minAtb[a]=-1
        
    plt.figure()
    plt.figure(figsize=(14,10))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.plot(FileNo,StatMat[0:FilesToBePloted,a],'bo')    
    alarmline=np.arange(start=minAtb[a],stop=maxAtb[a],step=((maxAtb[a]-minAtb[a])/100))
    
    if (AlarmOn==1):
        for al in range(howmanyAlarm):
    
            if (alarmline.size >100):
                alarmline=np.delete(alarmline,0) 
            plt.plot(fileindexAlarm[al],alarmline,color='red',linewidth=6) 
    
    if (WarningOn==1):
        for al in range(howmanyWarning):
    
            if (alarmline.size >100):
                alarmline=np.delete(alarmline,0) 
            plt.plot(fileindexWarning[al],alarmline,color='orange',linewidth=1)     
   
    plt.xlabel('File Sequence', fontsize=25)    
    plt.ylabel(Attribute[a],fontsize=25)
    plt.title('Attribute Distribution with Warning and Alarms')
    
    plt.show()
            

#%%   Dynamically Exploring the Box plot of each file attributes - ALL FILES - MANY PLOTS - AVOID running this cell IF Not necessary 
#%%
#     together with MACHINE ALARM to see which attribute changes the most before ALARM
  
for n in range(NumFile):
    myAtData=FOCNewData[n]
    # plt.figure()
    # plt.figure(figsize=(14,6))
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=14)
    
    MainTitle='BoxPlot  -  ' 
    DynTitle=mylist[n][34:45]
    NowTitle=MainTitle+DynTitle
    
    fig, axs = plt.subplots(figsize = (10,10))
    plot=sns.boxplot(data = myAtData);
    plot.set_xticklabels(plot.get_xticklabels(), rotation=40);
    
    
#%% Dynamically Plotting the Distribution Histogram of each attribute from the FOCNewData dataframe over time  - ALL FILES - MANY PLOTS - AVOID running this cell IF Not necessary 
#%%
# To Check the Distribution and its changes over time - for each defined atttribute
# ColmOfInterest=Attribute[6]
Attribute_Num=13
Interval = 100
Min={}
Max={}
gMinMat=[Attribute]
gMaxMat=[Attribute]
gMin={}
gMax={}
gIntv={}
bin_values={}

for n in range(NumFile):
    Min[n]=FOCFinalStat[n].loc[:,'min'] # getting min for all the attributes
    Max[n]=FOCFinalStat[n].loc[:,'max'] # getting max for all the attributes 
    
    gMinMat=np.vstack((gMinMat,Min[n].values))
    gMaxMat=np.vstack((gMaxMat,Max[n].values))
    

# This is to create bin_values to plot the dynamics of histogram for each attribute over time 
for a in range(len(Attribute)):
        gMin[a]=min(gMinMat[1:NumFile+1,a])
        gMax[a]=max(gMaxMat[1:NumFile+1,a])            
        circlePatchCenterX=gMin[a]+((gMax[a]-gMin[a])*0.75) # This is create a patch on the Plot to show Alarm
    
        if gMin[a]<gMax[a]:
            gIntv[a]=(gMax[a]-gMin[a])/Interval
            bin_values[a]= np.arange(start=gMin[a], stop=gMax[a], step=gIntv[a])
        else:
            bin_values[a]=0


AttbOfInterest=Attribute[Attribute_Num]

for n in range(NumFile):
    myAtData=FOCNewData[n].loc[:,AttbOfInterest]
    #circlePatchCenterY=max(myAtData)*0.75  # This is create a patch on the Plot to show Alarm
    
    plt.figure()
    plt.figure(figsize=(14,6))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    
    MainTitle='Dist  -  ' 
    DynTitle=mylist[n][34:45]
    NowTitle=MainTitle+DynTitle
    plt.title(NowTitle,fontsize=20)
    plt.hist(myAtData, bins=bin_values[Attribute_Num], align='left', color='r', edgecolor='y',
              linewidth=1)
    
    # Wanted to add a patch on the histogram to show the alarm visually
    # if (n in AlarmFileNo):
    #     circle1 = plt.Circle((circlePatchCenterY, circlePatchCenterY), 5, color='r')
    #     #circle2 = plt.Circle((0.5, 0.5), 0.2, color='blue')
    #     #circle3 = plt.Circle((1, 1), 0.2, color='g', clip_on=False)
    # plt.gcf().add_patch(circle1)

    # n, bins, patches = plt.hist(myAtData, bins=bin_values[Attribute_Num], facecolor='#2ab0ff', edgecolor='#e0e0e0', linewidth=0.5, alpha=0.7)
    # n = n.astype('int') # it MUST be an integer
    # for i in range(len(patches)):
    #     patches[i].set_facecolor(plt.cm.viridis(n[i]/max(n)))
    # # important patch    
    # patches[6].set_fc('red') # Set color
    # patches[6].set_alpha(1) # Set opacity    
    # Add annotation
    #plt.annotate('Important Bar!', xy=(0.57, 175), xytext=(2, 130), fontsize=15, arrowprops={'width':0.4,'headwidth':7,'color':'#333333'})
    # Add title and labels with custom font sizes
    
    plt.title('Distribution', fontsize=12)
    plt.xlabel('Stretch between Max and Min Bin Values', fontsize=10)
    plt.ylabel(AttbOfInterest, fontsize=10)
    
    plt.show()
    
#%% MACHINE LEARNING STARTS HERE     

#%% KMean Clustering of the DATA set - Single File

#%% This cell is to test how many clusters are there in the data set


#myTestData=FOCNewData[160]  # Test to check each files for no. of clusters in Main Data
myTestData=FOCFinalStat[160]  # Test to check each files for no. of clusters in Stat Data

# Finding KMeans Cluster from each data file based on STAT data FOCFinalStat
cluster_range=range(2,9) # look for 2 to 9 clusters
cluster_errors=[]


for num_clusters in cluster_range:
    clusters = KMeans(n_clusters=num_clusters,max_iter=300,) # Main KMeans calling 
    clusters.fit(myTestData) # Main KMeans fittign the data  
    labels=clusters.labels_  # Get Cluster labels
    centroids=clusters.cluster_centers_ # Finding Cluster Centers in multidimensional space
    cluster_errors.append(clusters.inertia_)
clusters_df=pd.DataFrame({"num_clusters":cluster_range,"Cluster_errors":cluster_errors})
#clusters_df[0:15]

# Elbow plot
plt.plot(clusters_df.num_clusters,clusters_df.Cluster_errors,marker= 'o')
# The elbow plot confirms that we have 6 distinguish clusters in almost all files. 

y_hat = clusters.predict(myTestData) # Predict which cluster each column of FOCAS belongs to 
clusterLonging=pd.DataFrame({"Attributes":Attribute,"Cluster it Belongs to":y_hat}) # Store them in a data frame to identify 
S_Score=metrics.silhouette_score(myTestData,labels,metric='euclidean') # Calculate Silhouette score for this data file 


#%%
# The elbow plot confirms that we have 6 distinguish clusters in almost all files. Proceed with 6 clusters
# Just to distinguish from the previous cell - variable 'clusters' is chnaged to 'kmeanCluster' and subsequently other variables are changed

kmeanCluster=KMeans(n_clusters=6,max_iter=300) # Main KMeans calling
kmeanCluster.fit(myTestData) # Main KMeans fittign the data
kmean_centroids=kmeanCluster.cluster_centers_ # Finding Cluster Centers in multidimensional space 
y_hat = kmeanCluster.predict(myTestData) # Predict which cluster each Attribute of FOCAS belongs to 
kmean_clusterLonging=pd.DataFrame({"Attributes":Attribute,"Cluster it Belongs to":y_hat}) # Store them in a data frame to identify 
kmean_labels = kmeanCluster.labels_ # Get Cluster labels
kmean_Score=metrics.silhouette_score(myTestData,kmean_labels,metric='euclidean') # Calculate Silhouette score for this data file 


#%% RUNNING KMeans ALGORITHM for ALL DATA files
#%%

# Now based on this understanding from the previous cell lets categorize the files and score them with kmean_Score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
FOC_StatData_KMeans_df={}
kmean_clusterLonging={}
NumOptCluster=6   # This is obtained from Previous Cell 
NumKmeanCluster=NumOptCluster
kmeanCluster=KMeans(n_clusters=NumOptCluster,max_iter=300)
kmean_Score=np.zeros(NumFile)
kmean_silhouette=np.zeros(NumFile)


ClusterCol=[]
for c in range(NumOptCluster):
    ClusString='Cluster '+ str(c+1)
    ClusterCol.append(ClusString) 

for n in range(NumFile):
    myFileData = FOCFinalStat[n].loc[:, :].values
    kmeanCluster.fit(myFileData)
    kmean_centroids=kmeanCluster.cluster_centers_
    
    y_hat= kmeanCluster.predict(myFileData)
    kmean_clusterLonging[n]=pd.DataFrame({"Attributes":Attribute,"Cluster it Belongs to":y_hat}) # Store them in a data frame to identify 
    kmean_labels = kmeanCluster.labels_
    kmean_silhouette[n]=metrics.silhouette_score(myFileData,kmean_labels,metric='euclidean') # Calculate Silhouette score for this data file 
    kmean_Score[n]=1/kmean_silhouette[n]
    FOC_StatData_KMeans_df[n] = pd.DataFrame(data = kmean_centroids.transpose(),
                                          columns = ClusterCol)
    
    
    
#%%  Plotting the Inverse of Silhouette Score for ALARM Indexing 
#%%   
kMin=min(kmean_Score)
kMax=max(kmean_Score) 
IntervalkmeanHist=100           
    #circlePatchCenterX=gMin[a]+((gMax[a]-gMin[a])*0.75) # This is create a patch on the Plot to show Alarm

if kMin<kMax:
    kIntv=(kMax-kMin)/IntervalkmeanHist
    k_bin_values= np.arange(start=kMin, stop=kMax, step=kIntv)
else:
    k_bin_values=0

# Seaborn distplot to visualize Kernel Density Function is there is any kink 
sns.histplot(kmean_Score, kde=True, color='red', bins=k_bin_values).set(title='Computed Alarm Index and its Density | Threshold Prediction', xlabel='Alarm Index', ylabel='Silhouette Density')
# Plotting using Matplotlib to identify the Alarm Threshold 
plt.figure(figsize=(20,10))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Alarm Index',fontsize=30)
plt.ylabel('Silhouette Density',fontsize=30)
plt.title('Computed Alarm Index and its Density | Threshold Prediction', fontsize=20)
# Plotting the Histogram - Distribution of kmean_Score
#plt.hist(kmean_Score, bins=k_bin_values, align='left', color='r', edgecolor='y',linewidth=1)

n, bins, patches = plt.hist(kmean_Score, bins=k_bin_values, facecolor='#2ab0ff', edgecolor='#e0e0e0', linewidth=0.5, alpha=0.7)
n = n.astype('int') # it MUST be an integer
for i in range(len(patches)):
     patches[i].set_facecolor(plt.cm.viridis(n[i]/max(n)))
# important patch    
patches[9].set_fc('red') # Set color
patches[9].set_alpha(1) # Set opacity 

patches[12].set_fc('blue') # Set color
patches[12].set_alpha(1) # Set opacity      
# Add annotation
plt.annotate('Threshold for ALARM Conservative 1.3', xy=(1.3, 130), xytext=(1.7, 230), fontsize=25, arrowprops={'width':0.6,'headwidth':7,'color':'#333333'})
plt.annotate('Threshold for ALARM TRUE 1.4', xy=(1.45, 80), xytext=(2, 120), fontsize=25, arrowprops={'width':0.6,'headwidth':7,'color':'#333333'})
    # Add title and labels with custom font sizes
plt.show()    


#%% Plotting Kmean_Score  THIS IS FINAL PLOT with Thresold Setting
#%%
# Specify what you wish to see in the plot 
AlarmOn=1
WarningOn=0
ThresholdSet=1
threshold=1.4

FilesToBePloted=NumFile  # i.e. All files in the dictionary of data set
#FilesToBePloted=300     # Specify specific number of files to be plotted
axes=plt.gca()
axes.set_xlim(0,FilesToBePloted)
maxScore=max(kmean_Score)
axes.set_ylim(0,maxScore)
# create alarn lines to plot
alarmlineKmean=np.arange(start=0,stop=maxScore,step=(maxScore-0)/100)
# create threshold line to plot 
thresholdlineX=np.arange(start=0,stop=FilesToBePloted,step=1)
thresholdlineY=np.ones((FilesToBePloted,1))*threshold


fileindexAlarm=np.full([howmanyAlarm,100],0)
fileindexWarning=np.full([howmanyWarning,100],0)

alarm=-1
warning=-1
# Loop to create fileindex vector for each alarm state 
for n in range(FilesToBePloted):
        if (StatMat[n,30]==2):   # Attribute 30 is MACHINE ALARM
           alarm=alarm+1
           fileindexAlarm[alarm]=np.ones((1,100))*n
        if (StatMat[n,30]==1):   # Attribute 30 is MACHINE ALARM
           warning=warning+1
           fileindexWarning[warning]=np.ones((1,100))*n


for n in range(FilesToBePloted):
    
    plt.bar(n, kmean_Score[n], align='center', alpha=0.5)
    
    if (ThresholdSet==1):

        plt.plot(thresholdlineX,thresholdlineY,color='green',linewidth=4) 
    
    
    if (AlarmOn==1):
        for al in range(howmanyAlarm):
    
            if (alarmlineKmean.size >100):
                alarmlineKmean=np.delete(alarmlineKmean,0) 
            plt.plot(fileindexAlarm[al],alarmlineKmean,color='red',linewidth=3) 
    
    if (WarningOn==1):
        for al in range(howmanyWarning):
    
            if (alarmlineKmean.size >100):
                alarmlineKmean=np.delete(alarmlineKmean,0) 
            plt.plot(fileindexWarning[al],alarmlineKmean,color='orange',linewidth=1)    
    
plt.xlabel('File number in Sequence', fontsize=14)
strTitle='ARARM INDEX [AU] Threshold ' + str(threshold)
plt.ylabel('ARARM INDEX [AU]', fontsize=14)
plt.title(strTitle,fontsize=15)

plt.annotate('RED bars | SPINDLE Failure', xy=(450, 2.0), xytext=(1000, 2.5), fontsize=10, arrowprops={'width':0.4,'headwidth':7,'color':'#333333'})
plt.annotate('GREEN Line | Detection Threshold', xy=(1400, 1.4), xytext=(1500, 0.5), fontsize=10, arrowprops={'width':0.4,'headwidth':7,'color':'#333333'})

plt.show()    

#%% Change in GRADIENT of Kmean Score
#%%
AlarmOn=1
WarningOn=0
# Objevtive is to find if that fluctuation of gradient is high (occured with higher frequency) that should be ALARMING
# So lets explore the SPECTOGRAM of Gradient of Silhoette Score of Kmean obtained from MAIN DATA - YEAH its Complicated 

from numpy import gradient
Kmean_grad=gradient(kmean_Score) # Gradient of the Kmean_ScorePCA

# This results a fluctuating Kmean_gradPCA between negative and positive number 

# Analyzing Spectogram
freq, FileNum, S_Alarm = signal.spectrogram(Kmean_grad)
#Plotting Spectogram
plt.pcolormesh(FileNum, freq, S_Alarm, shading='gouraud') # another option shading='gouraud'

plt.ylabel('Frequency [AU]', fontsize=14)
plt.xlabel('Fiel Number', fontsize=14)
plt.show() 

#%% Here we dynamically plot the cluster bearing of each attribute 
#%%                                       
for n in range(NumFile):  
    
    plt.figure(figsize=(10,10))
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=14)
    plt.xlabel('Clusters 6',fontsize=20)
    plt.ylabel('Kmeans',fontsize=20)
    MainTitle='KMeans  -  ' 
    DynTitle=mylist[n][6:21]
    NowTtile=MainTitle+DynTitle
    bplot=sns.barplot(x=Attribute[:],y=kmean_clusterLonging[n].loc[:,"Cluster it Belongs to"])
    plt.title(NowTtile,fontsize=20)
    bplot.set_xticklabels(bplot.get_xticklabels(), rotation=45, fontsize=10);
    plt.show()
    plt.gcf().clear()


#%% Kmeans ENDS Here 

#%% MACHINE LEARNING CONTINUES     

#%% Principal Component Analysis (PCA) of the Main DATA set - Single File

#%% Chronological PCA Analysis 3 Principal Components Of MAIN DATA 

# Data Normalization
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

FOC_MainData_PCA3_df={}
for n in range(NumFile):
    xm = FOCNewData[n].loc[:, :].values     # Main Data
    xm = StandardScaler().fit_transform(xm) # normalizing the features
    where_are_NaNs = np.isnan(xm)
    xm[where_are_NaNs] = 0
    pca_FOC = PCA(n_components=3)
    FOCpca = pca_FOC.fit_transform(xm)
    FOC_MainData_PCA3_df[n] = pd.DataFrame(data = FOCpca,
                                columns = ['principal component 1', 
                                           'principal component 2',
                                           'principal component 3']) 


#%% Dynamic Chnage of PCA in Main Data 

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



#%% Chronological PCA Analysis 3 Principal Components Of STATISTICS of DATA COLUMNS 
#%%
# Data Normalization

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
FOC_StatData_PCA3_df={}
for n in range(NumFile):
    xs3 = FOCFinalStat[n].loc[:, :].values    # Stat Data
    xs3 = StandardScaler().fit_transform(xs3) # normalizing the features
    pca_FOC = PCA(n_components=3)
    FOCpca = pca_FOC.fit_transform(xs3)
    FOC_StatData_PCA3_df[n] = pd.DataFrame(data = FOCpca,
                                columns = ['principal component 1', 
                                           'principal component 2',
                                           'principal component 3']) 
    
#%%  3D PCA Scatter PLOTS of Stat Data 

from mpl_toolkits.mplot3d import Axes3D


for n in range(NumFile): 
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    MainTitle='PCA  -  ' 
    DynTitle=mylist[n][34:45]
    NowTitle=MainTitle+DynTitle
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
# Plotting 2D PCA Plot only the one PCA (defined in PCAtoPlot) bar plot w.r.t file number  
     #NumAttb=[x for x in range(FOCpca.shape[0])] Required if use plt.bar from matplotlib
     # here seaborn librray is used 

PCAtoPlot='principal component 1'

     
for n in range(NumFile):  
    
    plt.figure(figsize=(10,10))
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=14)
    plt.xlabel('Data Attributes',fontsize=20)
    plt.ylabel('Principal Component - 1',fontsize=20)
    MainTitle='PCA  -  ' 
    DynTitle=mylist[n][34:45]
    NowTitle=MainTitle+DynTitle
    bplot=sns.barplot(x=Attribute[:],y=-FOC_StatData_PCA3_df[n].loc[:, PCAtoPlot])
    plt.title(NowTtile,fontsize=20)
    bplot.set_xticklabels(bplot.get_xticklabels(), rotation=45, fontsize=10);
    plt.show()
    plt.gcf().clear()

#%%
#%%
#%% MACHINE LEARNING CONTINUES     

#%% Kmeans Clustering of PCA of the DATA set 

#%% Chronological Kmeans of PCA Analysis 3 Principal Components Of MAIN DATA 

# Data Normalization
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
FOC_MainData_KMeansPCA_df={}
kmean_clusterLongingPCA={}
NumOptCluster=2
kmeanClusterPCA=KMeans(n_clusters=NumOptCluster,max_iter=300)
kmean_ScorePCA=np.zeros(NumFile)
kmean_silhouettePCA=np.zeros(NumFile)

ClusterCol=[]
for c in range(NumOptCluster):
    ClusString='Cluster '+ str(c+1)
    ClusterCol.append(ClusString) 
#%%

for n in range(NumFile):
    print(n)
    xm = FOCNewData[n].loc[:, :].values     # Loading Main Data by file number 
    xm = StandardScaler().fit_transform(xm) # normalizing the features
    where_are_NaNs = np.isnan(xm)
    xm[where_are_NaNs] = 0
    pca_FOC = PCA(n_components=3)
    FOCpca = pca_FOC.fit_transform(xm)  # PCA are found 
    # KMeans of PCA here
    kmeanClusterPCA.fit(FOCpca)
    kmean_centroidsPCA=kmeanClusterPCA.cluster_centers_
    
    #y_hatPCA= kmeanClusterPCA.predict(FOCpca)
    #kmean_clusterLongingPCA[n]=pd.DataFrame({"Attributes":Attribute,"Cluster it Belongs to":y_hatPCA}) # Store them in a data frame to identify 
    kmean_labelsPCA = kmeanClusterPCA.labels_
    if (np.sum(kmean_labelsPCA != 0)): # No cluster found
        kmean_silhouettePCA[n]=metrics.silhouette_score(FOCpca,kmean_labelsPCA,metric='euclidean') # Calculate Silhouette score for this data file 
    else:
        kmean_silhouettePCA[n]=1
    if (kmean_silhouettePCA[n]!=0):
        kmean_ScorePCA[n]=1/kmean_silhouettePCA[n]
    else:
        kmean_ScorePCA[n]=1
    FOC_MainData_KMeansPCA_df[n] = pd.DataFrame(data = kmean_centroidsPCA.transpose(),columns = ClusterCol)
#%%
    
#%%  Plotting the Inverse of Silhouette Score for ALARM Indexing from KMeans-PCA
#%% 
  
kMin=min(kmean_ScorePCA)
kMax=max(kmean_ScorePCA) 
# if (kMin < 0):
#     kmean_ScorePCA=kmean_ScorePCA-kMin
    
IntervalkmeanHist=100           
    #circlePatchCenterX=gMin[a]+((gMax[a]-gMin[a])*0.75) # This is create a patch on the Plot to show Alarm

if kMin<kMax:
    kIntv=(kMax-kMin)/IntervalkmeanHist
    k_bin_values= np.arange(start=kMin, stop=kMax, step=kIntv)
else:
    k_bin_values=0

# Seaborn distplot to visualize Kernel Density Function is there is any kink 
sns.histplot(kmean_ScorePCA, kde=True, color='red', bins=k_bin_values).set(title='Computed Alarm Index and its Density | Threshold Prediction', xlabel='Alarm Index', ylabel='Silhouette Density')
# Plotting using Matplotlib to identify the Alarm Threshold 
plt.figure(figsize=(10,10))
plt.xticks(fontsize=8)
plt.yticks(fontsize=14)
plt.xlabel('Alarm Index',fontsize=20)
plt.ylabel('Silhouette Density',fontsize=20)
plt.title('Computed Alarm Index and its Density | Threshold Prediction', fontsize=20)
# Plotting the Histogram - Distribution of kmean_Score
#plt.hist(kmean_Score, bins=k_bin_values, align='left', color='r', edgecolor='y',linewidth=1)

n, bins, patches = plt.hist(kmean_ScorePCA, bins=k_bin_values, facecolor='#2ab0ff', edgecolor='#e0e0e0', linewidth=0.5, alpha=0.7)
n = n.astype('int') # it MUST be an integer
for i in range(len(patches)):
     patches[i].set_facecolor(plt.cm.viridis(n[i]/max(n)))
# important patch    
patches[12].set_fc('red') # Set color
patches[12].set_alpha(1) # Set opacity 

patches[14].set_fc('blue') # Set color
patches[14].set_alpha(1) # Set opacity      
# Add annotation
plt.annotate('Threshold for ALARM Conservative 1.3', xy=(1.3, 130), xytext=(1.7, 130), fontsize=15, arrowprops={'width':0.4,'headwidth':7,'color':'#333333'})
plt.annotate('Threshold for ALARM TRUE 1.4', xy=(1.32, 80), xytext=(2, 80), fontsize=15, arrowprops={'width':0.4,'headwidth':7,'color':'#333333'})
    # Add title and labels with custom font sizes
plt.show()    

#%% Plotting Kmean_ScorePCA  THIS IS FINAL PLOT with Thresold Setting
#%%
# Specify what you wish to see in the plot 
AlarmOn=1
WarningOn=0
ThresholdSet=1
threshold=2.5

FilesToBePloted=NumFile
#FilesToBePloted=300
axes=plt.gca()
axes.set_xlim(0,FilesToBePloted)
maxScore=max(kmean_ScorePCA)
axes.set_ylim(0,maxScore)
# create alarn lines to plot
alarmlineKmean=np.arange(start=0,stop=maxScore,step=(maxScore-0)/100)
# create threshold line to plot 
thresholdlineX=np.arange(start=0,stop=FilesToBePloted,step=1)
thresholdlineY=np.ones((FilesToBePloted,1))*threshold

alarm=-1
warning=-1


fileindexAlarm=np.full([howmanyAlarm,100],0)
fileindexWarning=np.full([howmanyWarning,100],0)


# Loop to create fileindex vector for each alarm state 
####### NOT REQUIRED DONE PREVIOUSLY
for n in range(FilesToBePloted):
        if (StatMat[n,30]==2):   # Attribute 30 is MACHINE ALARM
            alarm=alarm+1
            fileindexAlarm[alarm]=np.ones((1,100))*n
        if (StatMat[n,30]==1):   # Attribute 30 is MACHINE ALARM
            warning=warning+1
            fileindexWarning[warning]=np.ones((1,100))*n
######

for n in range(FilesToBePloted):
    
    plt.bar(n, kmean_ScorePCA[n], align='center', alpha=0.5)
    
    if (ThresholdSet==1):

        plt.plot(thresholdlineX,thresholdlineY,color='green',linewidth=4) 
    
    
    if (AlarmOn==1):
        for al in range(howmanyAlarm):
    
            if (alarmlineKmean.size >100):
                alarmlineKmean=np.delete(alarmlineKmean,0) 
            plt.plot(fileindexAlarm[al],alarmlineKmean,color='red',linewidth=3) 
    
    if (WarningOn==1):
        for al in range(howmanyWarning):
    
            if (alarmlineKmean.size >100):
                alarmlineKmean=np.delete(alarmlineKmean,0) 
            plt.plot(fileindexWarning[al],alarmlineKmean,color='blue',linewidth=1)    
    
plt.xlabel('File number in Sequence', fontsize=14)
strTitle='ARARM INDEX [AU] Threshold ' + str(threshold)
plt.ylabel('ARARM INDEX [AU]', fontsize=14)
plt.title(strTitle,fontsize=15)

plt.annotate('RED bars | SPINDLE Failure', xy=(400,3.0), xytext=(500, 3.5), fontsize=10, arrowprops={'width':0.4,'headwidth':7,'color':'#333333'})
plt.annotate('GREEN Line | Detection Threshold', xy=(1500, 2.5), xytext=(1560, 1.5), fontsize=10, arrowprops={'width':0.4,'headwidth':7,'color':'#333333'})

plt.show()    

#%% Change in GRADIENT of Kmean Score
#%%
AlarmOn=1
WarningOn=0
# Objevtive is to find if that fluctuation of gradient is high (occured with higher frequency) that should be ALARMING
# So lets explore the SPECTOGRAM of Gradient of Silhoette Score of Kmean obtained from PCA clusters - YEAH its Complicated 

from numpy import gradient
Kmean_gradPCA=gradient(kmean_ScorePCA) # Gradient of the Kmean_ScorePCA

# This results a fluctuating Kmean_gradPCA between negative and positive number 

# Analyzing Spectogram
freq, FileNum, S_Alarm = signal.spectrogram(Kmean_gradPCA)
#Plotting Spectogram
plt.pcolormesh(FileNum, freq, S_Alarm, shading='gouraud') # another option shading='gouraud'

plt.ylabel('Frequency [AU]', fontsize=14)
plt.xlabel('Fiel Number', fontsize=14)
plt.show() 

#%%
#%%  K-Means Clustering of Stat Data and K-Mean Clustering of PCA of the Stat and Main Data are Performed above 

#    Please note the Name of the Variables generated for such each cases 

     # Var Name                                 -       Description
#    ________________                                 __________________________________________________________________________
    
#     FOCFinalStat                                   Dictionary of all Files contains Stats of All Attributes
#                                                    (3214 Data Frame, contains 31 x 8 Data Matrix)     
     
#     kmean_silhouette[n]                            Silhouette Score - of the nth file - time   
#     kmean_Score[n]                                 Inverse of Silhouette Score - of the nth file - time 
#     FOC_StatData_KMeans_df[n]                      K-mean cluster of Stat Data (StatParameter (8 here) x NumofClusters) of nth file - A Data Frame 
#                                                    (3214 Data Frame, contains 8 x 6 Data Matrix)


#     FOC_MainData_PCA3_df[n]                        PCA with 3 Principal Components obtained from MAIN Data - Data Frame 
   

#     FOC_StatData_PCA3_df[n]                        PCA with 3 Principal Components obtained from STAT Data - Data Frame 


#     FOC_MainData_KMeansPCA_df[n]                   K-mean cluster of PCA of MAIN Data - A Data Frame 
#                                                    (3214 Data Frame, contains Num of PCA x Num of Kmean cluster Data Matrix)

#     kmean_ScorePCA[n]                              Inverse of Silhouette Score from Kemans of PCA analysis - of the nth file - time 

#     kmean_silhouettePCA[n]                         Silhouette Score from PCA analysis - of the nth file - time  


#%%

#%%  Here we perfrom Gaussian Mixture Model (GMM) for One file and estimate the Weighted Mahalanobis Distace of each Data- Attribute  

from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal as mvn

NumOptCluster=6 # from Kmenas clustering 
OrderGMM=NumOptCluster
 
#for n in range(NumFile):
myFileData = FOCFinalStat[160].loc[:, :].values
DataShape=np.shape(myFileData)
NumDataPnts=DataShape[0]
NumFeatures=DataShape[1]

gmm = GaussianMixture(n_components=OrderGMM,covariance_type='full')
GmmFit=gmm.fit(myFileData)
GmmPredict=gmm.predict(myFileData)
GmmProb=gmm.predict_proba(myFileData)

GmmCenters = np.zeros((OrderGMM,NumFeatures))
GmmPriori=np.zeros((OrderGMM))
GmmSigma =np.zeros((OrderGMM,NumFeatures,NumFeatures))
MahalDataDis=np.zeros((NumDataPnts,OrderGMM))
MahalDis=np.zeros((NumDataPnts))

for j in range(OrderGMM):
    GmmDensity = mvn(cov=gmm.covariances_[j],mean=gmm.means_[j]).logpdf(myFileData)
                 # (NumFeatures x NumFeatures mat)  (Array of NumFeatures)
    GmmCenters[j,:]=myFileData[np.argmax(GmmDensity)]
    GmmPriori[j]=gmm.weights_[j]
    GmmSigma[j,:,:]=gmm.covariances_[j]
GmmMean=np.transpose(gmm.means_.T)

# Finding Mahalanobis Distance 

for i in range(NumDataPnts):
    MahalDis[i]=0
    for j in range(OrderGMM):
        u=myFileData[i,:]-GmmMean[j,:]                                         # u=(X - Mu)
        CovIn=np.linalg.inv(GmmSigma[j,:,:])                                   # inv(Cov)
        MahalDataDis[i,j]=np.dot(np.transpose(u),np.dot(CovIn,u))              # (X - Mu)*inv(Cov)*(X-Mu)' for each data and each order
        MahalDis[i]=MahalDis[i]+(GmmPriori[j]*np.sqrt(MahalDataDis[i,j]))      # Priori(order)*MahalaDataDis(order)  for each data (here attributes) 
    
    
#%% GMM Analysis of Each File - Time Evolution of Mahalanobis Distance of each attributes


from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal as mvn
import seaborn as sns

NumOptCluster=6 # from Kmenas clustering 
OrderGMM=NumOptCluster
myFileData = FOCFinalStat[2].iloc[:, :].values
DataShape=np.shape(myFileData)
NumDataPnts=DataShape[0]
NumFeatures=DataShape[1]

MahalFileDis=np.zeros((NumDataPnts-1,NumFile))  # Excluding the Alarm Data
Mahal_df={}

for n in range(NumFile):
    myFileData = FOCFinalStat[n].loc[:, :].values
    DataWOAlarm=myFileData[0:NumDataPnts-1,:]
    
    gmm   = GaussianMixture(n_components=OrderGMM,covariance_type='full')
    GmmFit=gmm.fit(DataWOAlarm)
    GmmPredict=gmm.predict(DataWOAlarm)
    GmmProb=gmm.predict_proba(DataWOAlarm)
    
    GmmCenters = np.zeros((OrderGMM,NumFeatures))
    GmmPriori=np.zeros((OrderGMM))
    GmmSigma =np.zeros((OrderGMM,NumFeatures,NumFeatures))
    MahalDataDis=np.zeros((NumDataPnts-1,OrderGMM))
    MahalDis=np.zeros((NumDataPnts-1))
    
    for j in range(OrderGMM):
        #GmmDensity = mvn(cov=gmm.covariances_[j],mean=gmm.means_[j]).logpdf(myFileData)
                     # (NumFeatures x NumFeatures mat)  (Array of NumFeatures)
        #GmmCenters[j,:]=myFileData[np.argmax(GmmDensity)]
        GmmPriori[j]=gmm.weights_[j]
        GmmSigma[j,:,:]=gmm.covariances_[j]
    GmmMean=np.transpose(gmm.means_.T)
    
    # Finding Mahalanobis Distance 
    
    for i in range(NumDataPnts-1):
        for j in range(OrderGMM):
            u=myFileData[i,:]-GmmMean[j,:]                                         # u=(X - Mu)
            CovIn=np.linalg.inv(GmmSigma[j,:,:])                                   # inv(Cov)
            MahalDataDis[i,j]=np.dot(np.transpose(u),np.dot(CovIn,u))              # (X - Mu)*inv(Cov)*(X-Mu)' for each data and each order
            MahalDis[i]=MahalDis[i]+(GmmPriori[j]*np.sqrt(MahalDataDis[i,j]))      # Priori(order)*MahalaDataDis(order)  for each data (here attributes) 
    
    MahalFileDis[:,n]=MahalDis
    
Mahal_df = pd.DataFrame(data = MahalFileDis.transpose(),columns = AttributeGmm) 

#for n in range(NumFile):
    
#%% Plotting the Distribution of Mahalanobis Distance of each Data Attribute from the GMM means 
plt.figure(figsize=(10,10))
plt.xticks(fontsize=8)
plt.yticks(fontsize=14)
plt.xlabel('Data Attributes',fontsize=20)
plt.ylabel('Priori Weighted Mahalabobis Distance',fontsize=20)
boxplot=sns.boxplot(data = Mahal_df);
boxplot.set_xticklabels(boxplot.get_xticklabels(), rotation=45, fontsize=10);

#%% Mahalanobis File to File Distance 
NumOptCluster=6 # from Kmenas clustering 
OrderGMM=NumOptCluster
myFileData = FOCFinalStat[2].iloc[:, :].values
DataShape=np.shape(myFileData)
NumDataPnts=DataShape[0]
NumFeatures=DataShape[1]

#GmmCenters = np.zeros((OrderGMM,NumFeatures))
GmmPriori=np.zeros((NumFile,OrderGMM))
GmmSigma =np.zeros((NumFile,OrderGMM,NumFeatures,NumFeatures))
GmmMean=np.zeros((NumFile,OrderGMM,NumFeatures))
MahalFileDis=np.zeros((NumFile,OrderGMM))
MahalDis=np.zeros((NumFile))
    
for n in range(NumFile):
    myFileData = FOCFinalStat[n].loc[:, :].values
    DataWOAlarm=myFileData[0:NumDataPnts-1,:]
    
    gmm   = GaussianMixture(n_components=OrderGMM,covariance_type='full')
    GmmFit=gmm.fit(DataWOAlarm)
    GmmPredict=gmm.predict(DataWOAlarm)
    GmmProb=gmm.predict_proba(DataWOAlarm)
    

    
    for j in range(OrderGMM):
        #GmmDensity = mvn(cov=gmm.covariances_[j],mean=gmm.means_[j]).logpdf(myFileData)
                     # (NumFeatures x NumFeatures mat)  (Array of NumFeatures)
        #GmmCenters[j,:]=myFileData[np.argmax(GmmDensity)]
        GmmPriori[n,j]=gmm.weights_[j]
        GmmSigma[n,j,:,:]=gmm.covariances_[j]
        GmmMean[n,j,:]=np.transpose(gmm.means_.T[:,j])
    
    # Finding Mahalanobis Distance 
    
for n in range(NumFile-1):
    
    for j in range(OrderGMM):
        u=GmmMean[n,j,:] -GmmMean[n+1,j,:]                                         # u=(X - Mu)
        CovInBackCov=np.linalg.inv(GmmSigma[n,j,:,:])                              # inv(Cov)
        MahalFileDis[n,j]=np.dot(np.transpose(u),np.dot(CovInBackCov,u))           # (X - Mu)*inv(Cov)*(X-Mu)' for each data and each order
    
        MahalDis[n]=MahalDis[n]+(GmmPriori[n,j]*np.sqrt(MahalFileDis[n,j]))          # Priori(order)*MahalaDataDis(order)  for each data (here attributes) 

#%% Final GMM Alarm Plot with GMM Score 
# Specify what you wish to see in the plot 
AlarmOn=1
WarningOn=0
ThresholdSet=1
threshold=250000

#FilesToBePloted=NumFile  # i.e. All files in the dictionary of data set
FilesToBePloted=600     # Specify specific number of files to be plotted
axes=plt.gca()
axes.set_xlim(0,FilesToBePloted)
maxScore=max(MahalDis)
axes.set_ylim(0,maxScore)
# create alarn lines to plot
alarmlineKmean=np.arange(start=0,stop=maxScore,step=(maxScore-0)/100)
# create threshold line to plot 
thresholdlineX=np.arange(start=0,stop=FilesToBePloted,step=1)
thresholdlineY=np.ones((FilesToBePloted,1))*threshold


fileindexAlarm=np.full([howmanyAlarm,100],0)
fileindexWarning=np.full([howmanyWarning,100],0)

alarm=-1
warning=-1
# Loop to create fileindex vector for each alarm state 
for n in range(FilesToBePloted):
        if (StatMat[n,30]==2):   # Attribute 30 is MACHINE ALARM
           alarm=alarm+1
           fileindexAlarm[alarm]=np.ones((1,100))*n
        if (StatMat[n,30]==1):   # Attribute 30 is MACHINE ALARM
           warning=warning+1
           fileindexWarning[warning]=np.ones((1,100))*n


for n in range(FilesToBePloted):
    
    plt.bar(n, MahalDis[n], align='center', alpha=0.5)
    
    if (ThresholdSet==1):

        plt.plot(thresholdlineX,thresholdlineY,color='green',linewidth=4) 
    
    
    if (AlarmOn==1):
        for al in range(howmanyAlarm):
    
            if (alarmlineKmean.size >100):
                alarmlineKmean=np.delete(alarmlineKmean,0) 
            plt.plot(fileindexAlarm[al],alarmlineKmean,color='red',linewidth=3) 
    
    if (WarningOn==1):
        for al in range(howmanyWarning):
    
            if (alarmlineKmean.size >100):
                alarmlineKmean=np.delete(alarmlineKmean,0) 
            plt.plot(fileindexWarning[al],alarmlineKmean,color='orange',linewidth=1)    
    
plt.xlabel('File number in Sequence', fontsize=14)
strTitle='GMM ARARM INDEX [AU] Threshold ' + str(threshold)
plt.ylabel('GMM ARARM INDEX [AU]', fontsize=14)
plt.title(strTitle,fontsize=15)
plt.ylim(0,900000)

if (FilesToBePloted==NumFile):
    plt.annotate('RED bars | SPINDLE Failure', xy=(450, 400000.0), xytext=(1000, 500000), fontsize=10, arrowprops={'width':0.4,'headwidth':7,'color':'#333333'})
    plt.annotate('GREEN Line | Detection Threshold', xy=(1400, threshold), xytext=(1500, 150000), fontsize=10, arrowprops={'width':0.4,'headwidth':7,'color':'#333333'})

if (FilesToBePloted==600):
    plt.annotate('RED bars | SPINDLE Failure', xy=(45, 400000.0), xytext=(100, 500000), fontsize=10, arrowprops={'width':0.4,'headwidth':7,'color':'#333333'})
    plt.annotate('GREEN Line | Detection Threshold', xy=(200, threshold), xytext=(250, 150000), fontsize=10, arrowprops={'width':0.4,'headwidth':7,'color':'#333333'})

plt.show()   
    
#%%    Lets TRY AgglomerativeClustering 

from sklearn.cluster import AgglomerativeClustering 
model = AgglomerativeClustering(n_clusters=6,sffinity='euclideab',linkage='average')
myStatData=FOCFinalStat[160].loc[:, :].values

model.fit(myStatData)

from scipy.cluster.hierarchy import cophenet, dendogram, linkage 
from scipy.spatial.distance import pdist # pairwise distribution between data points 

# cophenet index is a measure of the correlation between the distance of points in feature space and distance on dendogram 

Z=linkage('data variable here','average')
c,coph_dists=cophenet(Z,pdist('datahere'))

## Plottign Dendogram

plt.figure(figsize=(10,10))
plt.title('Agglomerative Hierarchical Clustering Dendogram')

plt.xlabel('sample index')
plt.ylabel('distance')

dendogram(Z,loafrotation=90.,color_threshold=40,leaf_font_size=8.)
plt.tight_layout()



#                                                         PROGRAM ENDS
#
#%%

