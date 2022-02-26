- 👋 Hi, I’m @mohabwael
- 👀 I’m interested in ...
- 🌱 I’m currently learning ...
- 💞️ I’m looking to collaborate on ...
- 📫 How to reach me ...

<!---
mohabwael/mohabwael is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 17:05:50 2022

@author: XTREME
"""

import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans

k=4
Kmeans = pd.read_csv("Customer data.csv")
[row1,col1]=np.shape(Kmeans)
NewKmeans=Kmeans.iloc[:,1:col1]
Cust_Data=pd.DataFrame(NewKmeans).to_numpy()
[row,col]=np.shape(Cust_Data)
Centroids=Kmeans.iloc[0:k,1:col1] #FI HAGA GHALAT
New_Centroids=pd.DataFrame(Centroids).to_numpy()
Matrix_Centroids=np.zeros([row,col])
Matrix_Centroids[0:k,0:col1]=New_Centroids
DistanceEc=np.zeros([row,k])
Distance_Type='Eucledian distance'
Cluster_Distance=np.zeros((row,k))
Centroids_random=Cust_Data[np.random.choice(row,k,replace=False)]

def GUC_Distance ( Centroids_random, Cust_Data, Distance_Type ):
    if(Distance_Type == 'Eucledian distance'):
        for i in range(k):    
            DistanceEc[:,i] = ((Cust_Data-Centroids_random[i])**2).sum(axis=1)
            Cluster_Distance=DistanceEc           
    else:
        Cluster_Distance=np.corrcoef(Centroids_random[0:k,:],Cust_Data) 
    return Cluster_Distance

def GUC_Kmean (Cust_Data,k, Distance_Type ):
    Centroids_random=Cust_Data[np.random.choice(row,k,replace=False)]
    while True:
        closest=np.zeros(row).astype(int)
        old_closest=closest.copy()
        if(Distance_Type=='Eucledian distance'):
            distance=GUC_Distance(Centroids_random,Cust_Data,'Eucledian distance')
        else:
            distance=GUC_Distance(Centroids_random,Cust_Data,'Pearson correlation distance')
        closest=np.argmin(distance,axis=1)
        for i in range(k):
            Centroids_random[i,:]=Cust_Data[closest==i].mean(axis=0)
            Distortion_Fn=(1/row)*np.sum(Cust_Data-np.resize(Centroids_random,(row,col)))*2
            plt.plot(Distortion_Fn,k)
            if(Distance_Type=='Eucledian distance'):
                Final_Cluster_distance=GUC_Distance(Centroids_random,Cust_Data,'Eucledian distance')
            else:
                Final_Cluster_distance=GUC_Distance(Centroids_random,Cust_Data,'Pearson correlation distance')
        if any(closest==old_closest):
            break
    return  Final_Cluster_distance,Distortion_Fn
Result=GUC_Kmean (Cust_Data,k,'Eucledian distance')
kmeans = KMeans(n_clusters=4)
kmeans.fit(Cust_Data)

def display_cluster(X,km=[],num_clusters=0):
    color = 'brgcmyk' #List colors
    alpha = 0.5 #color obaque
    s = 20
    if num_clusters == 0:
        plt.scatter(X[:,0],X[:,1],c = color[0],alpha = alpha,s = s)
    else:
        for i in range(num_clusters):
            plt.scatter(X[kmeans.labels_==i,0],X[kmeans.labels_==i,1],c = color[i],alpha = alpha,s=s)
            plt.scatter(kmeans.cluster_centers_[i][0],kmeans.cluster_centers_[i][1],c = color[i], marker = 'x', s = 100)
# Warini=display_cluster(Cust_Data,km=[],num_clusters=0)
# prepare the figure sise and background
# this part can be replaced by a number of subplots
plt.rcParams['figure.figsize'] = [8,8]
sns.set_style("whitegrid")
sns.set_context("talk")
# Produce a data set that represent the x and y o coordinates of a circle
# this part can be replaced by data that you import froma file
angle = np.linspace(0,2*np.pi,20, endpoint = False)
X = np.append([np.cos(angle)],[np.sin(angle)],0).transpose()
# Data is displayed
# to display the data only it is assumed that the number of clusters is zero which is the
l=display_cluster(X)
k_means_customer_data=GUC_Kmean(Cust_Data,k,'Eucledian distance' )
kmeans = KMeans(n_clusters=4)
kmeans.fit(Cust_Data)

def display_cluster(X,km=[],num_clusters=0):
    color = 'brgcmyk' #List colors
    alpha = 0.5 #color obaque
    s = 20
    if num_clusters == 0:
        plt.scatter(X[:,0],X[:,1],c = color[0],alpha = alpha,s = s)
    else:
        for i in range(num_clusters):
            
            plt.scatter(X[kmeans.labels_==i,0],X[kmeans.labels_==i,1],c = color[i],alpha = alpha,s=s)
            plt.scatter(kmeans.cluster_centers_[i][0],kmeans.cluster_centers_[i][1],c = color[i]
, marker = 'x', s = 100)

o=display_cluster(Cust_Data,km=[],num_clusters=4)
plt.rcParams['figure.figsize'] = [8,8]
sns.set_style("whitegrid")
sns.set_context("talk")
# Produce a data set that represent the x and y o coordinates of a circle
# this part can be replaced by data that you import froma file
angle = np.linspace(0,2*np.pi,20, endpoint = False)
X = np.append([np.cos(angle)],[np.sin(angle)],0).transpose()
# Data is displayed
# to display the data only it is assumed that the number of clusters is zero which is the
l=display_cluster(X)
n_samples = 1000
n_bins = 4
centers = [(-3, -3), (0, 0), (3, 3), (6, 6), (9,9)]
X, y = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.0,
centers=centers, shuffle=False, random_state=42)
n_samples = 1000
X, y = noisy_moons = make_moons(n_samples=n_samples, noise= .1)
a=display_cluster(X)
k_means_Examples= GUC_Kmean (np.resize(X,(row,col)),k,'Eucledian distance' )
    
# def GUC_Kmean (Cust_Data,k, Distance_Type ):
# while True:
# Min_index=np.zeros(row)
# Min_index_Temp=Min_index.copy()
# Centroids_random=Cust_Data[np.random.choice(row,k,replace=False)]
# distance=GUC_Distance(Centroids_random,Cust_Data,'Eucledian distance')
# Min_index=distance.argmin(axis=1)
# Min_Dis=distance.min(axis=1)
# if(Min_index.all()!=np.zeros(row)):
#     if (Min_index.all()==Min_index_Temp.all()):
            # break
# for x in range(k):
# Centroids_random[x,:]=Cust_Data[Min_index==x].mean(axis=0)
# return [ Final_Cluster_Distance , Cluster_Metric ]
