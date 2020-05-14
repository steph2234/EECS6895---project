import numpy as np
import pandas as pd



## LOF Scratch to generate anomaly score for each patient
## Input is the n by n numpy array matrix; every row, standing for a unique patient, starting with 0
##       and his/her distance with subsequent patient

def lof(df,k_neighbour = k):
    def get_lrd(patient_id,df):
        temp = dict()
        for k,v in enumerate(df[patient_id][1:]):
        
            if k >= patient_id:
                temp[k+1] = v  
            else: temp[k] = v
    
        k_neighbours = {k: v for k, v in sorted(temp.items(), key=lambda item: item[1])[0:k]}
    
        reach_distance = []
        for i in k_neighbours:
            distance_ab = k_neighbours[i]
            current = dict()
            for k,v in enumerate(df[i][1:]):
                current[k] = v
            k_distance_b = sorted(current.items(), key=lambda item: item[1])[k-1][1]
            reach_distance.append(max(k_distance_b,distance_ab))
    
        return 1/(sum(reach_distance)/100)

    lrd_list = []
    for i in range(len(df)):
        center = get_lrd(i,df)
        k_neigh = dict()
        for k,v in enumerate(df[i][1:]):
            if k >= i:
                k_neigh[k+1] = v 
            else: k_neigh[k] = v
    
        k_neigh = {k: v for k, v in sorted(k_neigh.items(), key=lambda item: item[1])[0:k]}
    
        k_lrd = []
        for j in k_neigh:
            k_lrd.append(get_lrd(j,df))
        lrd_list.append(np.average(center/np.asarray(k_lrd)))
    
    return lrd_list


