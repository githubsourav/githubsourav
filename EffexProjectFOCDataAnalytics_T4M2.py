# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 09:45:04 2020

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

This Program Particularly Analyze T4 Data from T4 #2 Machine  


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
# mylist = [f for f in glob.glob("T4 Data AFOSR\T4\T4\T4#2\Data\FOCAS\*.csv")]
mylist = [f for f in glob.glob("T4 Data AFOSR\T4#2\Data\FOCAS\*.csv")]
#mylist = [f for f in glob.glob("T4 Data AFOSR\T4#1\Data\FOCAS\*.csv")]

# Collecting Number of Files
NumFile=len(mylist)
#%%  Here we Load all the Data files and remove the unnecessary columns Then calculate the STATS of each file
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
Attribute=FOCNewData[1].columns  # Getting the Remaining Columns 

#%%
for n in range(NumFile):
    myAtData=FOCNewData[n]
    # plt.figure()
    # plt.figure(figsize=(14,6))
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=14)
    
    MainTitle='BoxPlot  -  ' 
    DynTitle=mylist[n][34:45]
    NowTtile=MainTitle+DynTitle
    
    fig, axs = plt.subplots(figsize = (10,10))
    plot=sns.boxplot(data = myAtData);
    plot.set_xticklabels(plot.get_xticklabels(), rotation=40);
    
    
#%% Plotting a Distribution from Describe data 

# To Check the Distribution and its changes over time - for each defined atttribute
# ColmOfInterest=Attribute[6]
Interval = 100
Min={}
Max={}
gMinMat=[Attribute]
gMaxMat=[Attribute]
gMin={}
gMax={}
gIntv={}
bin_values={}
#%%
for n in range(NumFile):
    Min[n]=FOCFinalStat[n].loc[:,'min'] # getting min for all the attributes
    Max[n]=FOCFinalStat[n].loc[:,'max'] # getting max for all the attributes 
    
    gMinMat=np.vstack((gMinMat,Min[n].values))
    gMaxMat=np.vstack((gMaxMat,Max[n].values))
    

#%%

# This is to create bin_values to plot the dynamics of histogram for each attribute over time 
for a in range(len(Attribute)):
        gMin[a]=min(gMinMat[1:NumFile+1,a])
        gMax[a]=max(gMaxMat[1:NumFile+1,a])            

    
        if gMin[a]<gMax[a]:
            gIntv[a]=(gMax[a]-gMin[a])/Interval
            bin_values[a]= np.arange(start=gMin[a], stop=gMax[a], step=gIntv[a])
        else:
            bin_values[a]=0

#%%
Attribute_Num=5
AttbOfInterest=Attribute[Attribute_Num]

for n in range(NumFile):
    myAtData=FOCNewData[n].loc[:,AttbOfInterest]
    plt.figure()
    plt.figure(figsize=(14,6))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    
    MainTitle='Dist  -  ' 
    DynTitle=mylist[n][34:45]
    NowTtile=MainTitle+DynTitle
    plt.title(NowTtile,fontsize=20)
    plt.hist(myAtData, bins=bin_values[Attribute_Num], align='left', color='r', edgecolor='y',
              linewidth=1)
    plt.xlabel('Stretch between Max and Min Bin Values')
    plt.ylabel(AttbOfInterest)
    
    plt.show()

#%%
myTestData=FOCFinalStat[160]  # Test to check each files 

# Finding KMeans Cluster from each data file based on STAT data FOCFinalStat
cluster_range=range(2,9) # look for 2 to 6 clusters
cluster_errors=[]


for num_clusters in cluster_range:
    clusters = KMeans(n_clusters=num_clusters,max_iter=300,)
    clusters.fit(myTestData)
    labels=clusters.labels_
    centroids=clusters.cluster_centers_
    cluster_errors.append(clusters.inertia_)
clusters_df=pd.DataFrame({"num_clusters":cluster_range,"Cluster_errors":cluster_errors})
clusters_df[0:15]

plt.plot(clusters_df.num_clusters,clusters_df.Cluster_errors,marker= 'o')

y_hat = clusters.predict(myTestData) # Predict which cluster each column of FOCAS belongs to 
clusterLonging=pd.DataFrame({"Attributes":Attribute,"Cluster it Belongs to":y_hat}) # Store them in a data frame to identify 
S_Score=metrics.silhouette_score(myTestData,labels,metric='euclidean') # Calculate Silhouette score for this data file 
# The elbow plot confirms that we have 6 distinguish clusters in almost all files. 

#%%
# The elbow plot confirms that we have 6 distinguish clusters in almost all files. Proceed with 6 clusters
# Just to distinguish from the previous cell - variable 'clusters' is chnaged to 'kmeanCluster' and subsequently other variables are changed

kmeanCluster=KMeans(n_clusters=6,max_iter=300)
kmeanCluster.fit(myTestData)
kmean_centroids=kmeanCluster.cluster_centers_
y_hat = kmeanCluster.predict(myTestData) # Predict which cluster each column of FOCAS belongs to 
kmean_clusterLonging=pd.DataFrame({"Attributes":Attribute,"Cluster it Belongs to":y_hat}) # Store them in a data frame to identify 
kmean_labels = kmeanCluster.labels_
kmean_Score=metrics.silhouette_score(myTestData,kmean_labels,metric='euclidean') # Calculate Silhouette score for this data file 


#%%

# Now based on this understanding from the previous cell lets categorize the files and score them with kmean_Score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
FOC_StatData_KMeans_df={}
kmean_clusterLonging={}
NumOptCluster=6
kmeanCluster=KMeans(n_clusters=NumOptCluster,max_iter=300)
kmean_Score=np.zeros(NumFile)

ClusterCol=[]
for c in range(NumOptCluster):
    ClusString='Cluster '+ str(c+1)
    ClusterCol.append(ClusString)
    
#%%   
plt.show()
axes=plt.gca()
axes.set_xlim(0,NumFile)
axes.set_ylim(0,1)

for n in range(NumFile):
    myFileData = FOCFinalStat[n].loc[:, :].values
    kmeanCluster.fit(myFileData)
    kmean_centroids=kmeanCluster.cluster_centers_
    
    y_hat= kmeanCluster.predict(myFileData)
    kmean_clusterLonging[n]=pd.DataFrame({"Attributes":Attribute,"Cluster it Belongs to":y_hat}) # Store them in a data frame to identify 
    kmean_labels = kmeanCluster.labels_
    kmean_Score[n]=metrics.silhouette_score(myFileData,kmean_labels,metric='euclidean') # Calculate Silhouette score for this data file 
    FOC_StatData_KMeans_df[n] = pd.DataFrame(data = kmean_centroids.transpose(),
                                          columns = ClusterCol)
    
    
    plt.bar(n, kmean_Score[n], align='center', alpha=0.5)
    
#%%    Here we dynamically plot the cluster bearing of each attribute 
                                       
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


#%%