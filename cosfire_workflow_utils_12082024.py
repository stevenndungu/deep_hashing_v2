# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 10:25:28 2023

@author: P307791
"""
import glob
import ast
import pandas as pd
import numpy as np
import os, random, string
from scipy import stats
import torch #torch==1.13.1
import torch.nn as nn
from torch.linalg import vector_norm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
#pytorch-metric-learning==2.1.0
# scikit-image==0.20.0
# scikit-learn==1.2.2
#seaborn==0.12.2
#tqdm==4.65.0
import argparse
import pandas as pd #pandas==2.0.0
from scipy.io import loadmat #scipy==1.10.1
import matplotlib.pyplot as plt
import numpy as np #numpy==1.26.4
import seaborn as sns
os.environ['PYTHONHASHSEED'] = 'python'
import re
from sklearn.preprocessing import label_binarize


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import random

import torch
from torch.linalg import vector_norm
from sklearn.manifold import TSNE
from sklearn import preprocessing
from PIL import Image

dic_labels = { 'Bent':2,
  'Compact':3,
    'FRI':0,
    'FRII':1
}

dic_labels_rev = { 2:'Bent',
                3:'Compact',
                  0:'FRI',
                  1: 'FRII'
              }

#For Reproducibility
def reproducibility_requirements(seed=100):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #torch.use_deterministic_algorithms(True)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #print("Set seed of", str(seed),"is done for Reproducibility")

reproducibility_requirements()


def generate_unique_identifier(length):
    characters = string.ascii_letters + string.digits
    identifier = ''.join(random.choice(characters) for _ in range(length))
    return identifier


#%%
def SimplifiedTopMap(rB, qB, retrievalL, queryL, topk):
  '''
    rB - binary codes of the training set - reference set,
    qB - binary codes of the query set,
    retrievalL - labels of the training set - reference set, 
    queryL - labels of the query set, and 
    topk - the number of top retrieved results to consider.

    rB = r_binary
    qB = q_binary
    retrievalL = train_label
    queryL = valid_label
    topk = 100
  '''
  num_query = queryL.shape[0]
  mAP = [0] * num_query
 
  for i, query in enumerate(queryL):

    rel = (np.dot(query, retrievalL.transpose()) > 0)*1 # relevant train label refs.
    hamm = np.count_nonzero(qB[i] != rB, axis=1) #hamming distance
    ind = np.argsort(hamm) #  Hamming distances in ascending order.
    rel = rel[ind] #rel is reordered based on the sorted indices ind, so that it corresponds to the sorted Hamming distances.

    top_rel = rel[:topk] #contains the relevance values for the top-k retrieved results
    tot_relevant_sum = np.sum(top_rel) #total number of relevant images in the top-k retrieved results

    #skips the iteration if there are no relevant results.
    if tot_relevant_sum == 0:
        continue

    pr_num = np.linspace(1, tot_relevant_sum, tot_relevant_sum) 
    pr_denom = np.asarray(np.where(top_rel == 1)) + 1.0 #is the indices where top_rel is equal to 1 (i.e., the positions of relevant images)
    pr = pr_num / pr_denom # precision
    # dic_labels_rev = { 2:'Bent',
    #             3:'Compact',
    #               0:'FRI',
    #               1: 'FRII'
    #           }
    if (query == np.array([0, 0, 1, 0])).sum()==4:#Bent
         mAP_sub = np.sum(pr) /np.min(np.array([305,topk]))
    elif (query == np.array([0, 1, 0, 0])).sum()==4:#FRII
         mAP_sub = np.sum(pr) / np.min(np.array([434,topk]))
    elif (query == np.array([1, 0, 0, 0])).sum()==4:#FRI
         mAP_sub = np.sum(pr) /  np.min(np.array([215,topk]))
    else:# (query == np.array([0, 0, 0, 1])).sum()==4:#Compact
         mAP_sub = np.sum(pr) / np.min(np.array([226,topk]))
   
    #mAP_sub = np.sum(pr) / topk

    mAP[i] = mAP_sub 
    
      
  #return np.mean(mAP)*100, stats.median_abs_deviation(mAP, scale=1)*100, mAP_values
  return np.mean(mAP)*100, np.std(mAP)*100, mAP


def mAP_values(r_database,q_database, thresh = 0.5, percentile = True, topk = 100):
    if percentile:
        r_binary = np.array([((out >= np.percentile(out,thresh))*1).tolist()  for _, out in enumerate(r_database.predictions)])
        q_binary = np.array([((out >= np.percentile(out,thresh))*1).tolist()  for _, out in enumerate(q_database.predictions)])
    else:
        r_binary = np.array([((out >= thresh) * 1).tolist() for _, out in enumerate(r_database.predictions)])
        q_binary = np.array([((out >= thresh) * 1).tolist() for _, out in enumerate(q_database.predictions)])

    train_label = label_binarize(r_database.label_code, classes=[0, 1, 2,3])
    valid_label = label_binarize(q_database.label_code, classes=[0,1, 2,3])

    rB = r_binary
    qB = q_binary
    retrievalL = train_label
    queryL = valid_label
    topk = topk
    mAP, mAP_std, mAP_values1 = SimplifiedTopMap(rB, qB, retrievalL, queryL, topk)
  
    return mAP,mAP_std,mAP_values1, r_binary, train_label, q_binary, valid_label

#%%

def CalcTopMapWithPR123(referenceBinaryCodes, queryBinaryCodes, referenceLabels, queryLabels, topk):
    '''
    Parameters:
    - referenceBinaryCodes: binary codes of the training set (reference set),
    - queryBinaryCodes: binary codes of the query set,
    - referenceLabels: labels of the training set (reference set),
    - queryLabels: labels of the query set,
    - topk: the number of top retrieved results to consider.

    Returns:
    - topkmap: mean Average Precision (mAP) for the top k retrieved items,
    - cum_prec: cumulative precision over all queries,
    - cum_recall: cumulative recall over all queries,
    - cumulative_mAP: cumulative mean Average Precision across all queries.
    '''
    
    numQueries = queryLabels.shape[0]  # Number of queries
    numReferences = referenceLabels.shape[0]  # Number of items in the reference set
    topkmap = 0  # Initialize the mAP counter
    precisionMatrix = np.zeros((numQueries, numReferences))  # Precision matrix
    recallMatrix = np.zeros((numQueries, numReferences))  # Recall matrix
    mAPs = np.zeros(numQueries)  # Array to store mAPs for each query
    
    for i in tqdm(range(numQueries)):
        # Determine the relevance of each reference item to the current query
        relevant = (np.dot(queryLabels[i, :], referenceLabels.transpose()) > 0) * 1  # Binary relevance (1 if relevant, else 0)
        
        # Calculate Hamming distances between the query and all reference items
        hammingDistances = np.count_nonzero(queryBinaryCodes[i] != referenceBinaryCodes, axis=1)
        
        # Sort the indices of the reference items by ascending Hamming distance
        sortedIndices = np.argsort(hammingDistances)
        relevant = relevant[sortedIndices]  # Reorder relevance according to sorted distances

        # Extract the top-k relevant items based on sorted Hamming distances
        topRelevant = relevant[0:topk]
        numRelevantInTopK = np.sum(topRelevant).astype(int)  # Count the number of relevant items in the top-k
        
        if numRelevantInTopK == 0:
            continue  # Skip if there are no relevant items in the top-k results
        
        # Create a sequential array up to the number of relevant items found
        relevantCount = np.linspace(1, numRelevantInTopK, numRelevantInTopK)
        totalRelevant = np.sum(relevant)  # Total number of relevant items in the reference set

        # Cumulative sum of relevant items retrieved up to each point
        cumulativeRelevantSum = np.cumsum(relevant)
        retrievedItems = np.arange(1, numReferences + 1)  # Index array for all retrieved items

        # Precision and recall calculations for the current query
        precisionMatrix[i, :] = cumulativeRelevantSum / retrievedItems
        recallMatrix[i, :] = cumulativeRelevantSum / totalRelevant

        # Assertions to ensure correctness
        assert recallMatrix[i, -1] == 1.0  # Ensure recall reaches 1.0 at the end
        assert totalRelevant == cumulativeRelevantSum[-1]  # Ensure all relevant items are accounted for

        # Calculate the average precision for the top-k retrieved items
        relevantIndices = np.asarray(np.where(topRelevant == 1)) + 1.0  # Indices of relevant items in top-k
        topkAveragePrecision = np.mean(relevantCount / relevantIndices)  # Average precision for the top-k
        mAPs[i] = topkAveragePrecision  # Store mAP for this query
        topkmap += topkAveragePrecision  # Accumulate the top-k mAP

    topkmap /= numQueries  # Normalize by the number of queries

    # Filter the precision and recall matrices to only include valid queries
    validIndices = np.argwhere(recallMatrix[:, -1] == 1.0).squeeze()
    precisionMatrix = precisionMatrix[validIndices]
    recallMatrix = recallMatrix[validIndices]
    
    # Compute cumulative precision and recall across all valid queries
    cumulativePrecision = np.mean(precisionMatrix, 0)
    cumulativeRecall = np.mean(recallMatrix, 0)
    
    # Compute cumulative mAP across all queries
    cumulative_mAP = np.cumsum(mAPs) / (np.arange(1, numQueries + 1))  # Cumulative mean of mAPs

    return np.round(topkmap,4) * 100, cumulativePrecision, cumulativeRecall, cumulative_mAP  # Return the mAP, cumulative precision, cumulative recall, and cumulative mAP

def CalcTopMapWithPR(referenceBinaryCodes, queryBinaryCodes, referenceLabels, queryLabels, topk):
    '''
    Parameters:
    - referenceBinaryCodes: binary codes of the training set (reference set),
    - queryBinaryCodes: binary codes of the query set,
    - referenceLabels: labels of the training set (reference set),
    - queryLabels: labels of the query set,
    - topk: the number of top retrieved results to consider.

    Returns:
    - topkmap: mean Average Precision (mAP) for the top k retrieved items,
    - cum_prec: cumulative precision over all queries,
    - cum_recall: cumulative recall over all queries,
    - cumulative_mAP: cumulative mean Average Precision across all queries.
    '''
    
    numQueries = queryLabels.shape[0]  # Number of queries
    numReferences = referenceLabels.shape[0]  # Number of items in the reference set
    topkmap = 0  # Initialize the mAP counter
    precisionMatrix = np.zeros((numQueries, numReferences))  # Precision matrix
    recallMatrix = np.zeros((numQueries, numReferences))  # Recall matrix
    mAPs = np.zeros(numQueries)  # Array to store mAPs for each query
    
    for i in tqdm(range(numQueries)):
        # Determine the relevance of each reference item to the current query
        relevant = (np.dot(queryLabels[i, :], referenceLabels.transpose()) > 0) * 1  # Binary relevance (1 if relevant, else 0)
        
        # Calculate Hamming distances between the query and all reference items
        hammingDistances = np.count_nonzero(queryBinaryCodes[i] != referenceBinaryCodes, axis=1)
        
        # Sort the indices of the reference items by ascending Hamming distance
        sortedIndices = np.argsort(hammingDistances)
        relevant = relevant[sortedIndices]  # Reorder relevance according to sorted distances

        # Extract the top-k relevant items based on sorted Hamming distances
        topRelevant = relevant[0:topk]
        numRelevantInTopK = np.sum(topRelevant).astype(int)  # Count the number of relevant items in the top-k
        
        if numRelevantInTopK == 0:
            continue  # Skip if there are no relevant items in the top-k results
        
        # Create a sequential array up to the number of relevant items found
        relevantCount = np.linspace(1, numRelevantInTopK, numRelevantInTopK)
        totalRelevant = np.sum(relevant)  # Total number of relevant items in the reference set

        # Cumulative sum of relevant items retrieved up to each point
        cumulativeRelevantSum = np.cumsum(relevant)
        retrievedItems = np.arange(1, numReferences + 1)  # Index array for all retrieved items

        # Precision and recall calculations for the current query
        precisionMatrix[i, :] = cumulativeRelevantSum / retrievedItems
        recallMatrix[i, :] = cumulativeRelevantSum / totalRelevant

        # Assertions to ensure correctness
        assert recallMatrix[i, -1] == 1.0  # Ensure recall reaches 1.0 at the end
        assert totalRelevant == cumulativeRelevantSum[-1]  # Ensure all relevant items are accounted for

        # Calculate the average precision for the top-k retrieved items
        relevantIndices = np.asarray(np.where(topRelevant == 1)) + 1.0  # Indices of relevant items in top-k
        topkAveragePrecision = np.mean(relevantCount / relevantIndices)  # Average precision for the top-k
        mAPs[i] = topkAveragePrecision  # Store mAP for this query
        topkmap += topkAveragePrecision  # Accumulate the top-k mAP

    topkmap /= numQueries  # Normalize by the number of queries

    # Filter the precision and recall matrices to only include valid queries
    validIndices = np.argwhere(recallMatrix[:, -1] == 1.0).squeeze()
    precisionMatrix = precisionMatrix[validIndices]
    recallMatrix = recallMatrix[validIndices]
    
    # Compute cumulative precision and recall across all valid queries
    cumulativePrecision = np.mean(precisionMatrix, 0)
    cumulativeRecall = np.mean(recallMatrix, 0)
    
    # Compute cumulative mAP across all queries
    cumulative_mAP = np.cumsum(mAPs) / (np.arange(1, numQueries + 1))  # Cumulative mean of mAPs

    return np.round(topkmap,4) * 100, cumulativePrecision, cumulativeRecall, cumulative_mAP  # Return the mAP, cumulative precision, cumulative recall, and cumulative mAP


def CalcTopMapWithPR2(referenceBinaryCodes, queryBinaryCodes, referenceLabels, queryLabels, topk):
    '''
    Parameters:
    - referenceBinaryCodes: binary codes of the training set (reference set),
    - queryBinaryCodes: binary codes of the query set,
    - referenceLabels: labels of the training set (reference set),
    - queryLabels: labels of the query set,
    - topk: the number of top retrieved results to consider.

    Returns:
    - topkmap: mean Average Precision (mAP) for the top k retrieved items,
    - cum_prec: cumulative precision over all queries,
    - cum_recall: cumulative recall over all queries.
    '''
    
    numQueries = queryLabels.shape[0]  # Number of queries
    numReferences = referenceLabels.shape[0]  # Number of items in the reference set
    topkmap = 0  # Initialize the mAP counter
    precisionMatrix = np.zeros((numQueries, numReferences))  # Precision matrix
    recallMatrix = np.zeros((numQueries, numReferences))  # Recall matrix
    
    for i in tqdm(range(numQueries)):
        # Determine the relevance of each reference item to the current query
        relevant = (np.dot(queryLabels[i, :], referenceLabels.transpose()) > 0) * 1  # Binary relevance (1 if relevant, else 0)
        
        # Calculate Hamming distances between the query and all reference items
        hammingDistances = np.count_nonzero(queryBinaryCodes[i] != referenceBinaryCodes, axis=1)
        
        # Sort the indices of the reference items by ascending Hamming distance
        sortedIndices = np.argsort(hammingDistances)
        relevant = relevant[sortedIndices]  # Reorder relevance according to sorted distances

        # Extract the top-k relevant items based on sorted Hamming distances
        topRelevant = relevant[0:topk]
        numRelevantInTopK = np.sum(topRelevant).astype(int)  # Count the number of relevant items in the top-k
        
        if numRelevantInTopK == 0:
            continue  # Skip if there are no relevant items in the top-k results
        
        # Create a sequential array up to the number of relevant items found
        relevantCount = np.linspace(1, numRelevantInTopK, numRelevantInTopK)
        totalRelevant = np.sum(relevant)  # Total number of relevant items in the reference set

        # Cumulative sum of relevant items retrieved up to each point
        cumulativeRelevantSum = np.cumsum(relevant)
        retrievedItems = np.arange(1, numReferences + 1)  # Index array for all retrieved items

        # Precision and recall calculations for the current query
        precisionMatrix[i, :] = cumulativeRelevantSum / retrievedItems
        recallMatrix[i, :] = cumulativeRelevantSum / totalRelevant

        # Assertions to ensure correctness
        assert recallMatrix[i, -1] == 1.0  # Ensure recall reaches 1.0 at the end
        assert totalRelevant == cumulativeRelevantSum[-1]  # Ensure all relevant items are accounted for

        # Calculate the average precision for the top-k retrieved items
        relevantIndices = np.asarray(np.where(topRelevant == 1)) + 1.0  # Indices of relevant items in top-k
        topkAveragePrecision = np.mean(relevantCount / relevantIndices)  # Average precision for the top-k
        topkmap += topkAveragePrecision  # Accumulate the top-k mAP

    topkmap /= numQueries  # Normalize by the number of queries

    # Filter the precision and recall matrices to only include valid queries
    validIndices = np.argwhere(recallMatrix[:, -1] == 1.0).squeeze()
    precisionMatrix = precisionMatrix[validIndices]
    recallMatrix = recallMatrix[validIndices]
    
    # Compute cumulative precision and recall across all valid queries
    cumulativePrecision = np.mean(precisionMatrix, 0)
    cumulativeRecall = np.mean(recallMatrix, 0)

    return np.round(topkmap,4) *100, cumulativePrecision, cumulativeRecall  # Return the mAP, cumulative precision, and cumulative recall


# def CalcTopMapWithPR(rB, qB, retrievalL, queryL, topk):
   
#     num_query = queryL.shape[0]
#     num_gallery = retrievalL.shape[0]
#     topkmap = 0
#     prec = np.zeros((num_query, num_gallery))
#     recall = np.zeros((num_query, num_gallery))
#     for i in tqdm(range(num_query)):
#         gnd = (np.dot(queryL[i, :], retrievalL.transpose()) > 0)*1#.astype(np.float32)
#         hamm = np.count_nonzero(qB[i] != rB, axis=1) 
#         ind = np.argsort(hamm)
#         gnd = gnd[ind]

#         tgnd = gnd[0:topk]
#         tsum = np.sum(tgnd).astype(int)
#         if tsum == 0:
#             continue
#         count = np.linspace(1, tsum, tsum)
#         all_sim_num = np.sum(gnd)

#         prec_sum = np.cumsum(gnd)
#         return_images = np.arange(1, num_gallery + 1)

#         prec[i, :] = prec_sum / return_images
#         recall[i, :] = prec_sum / all_sim_num

#         assert recall[i, -1] == 1.0
#         assert all_sim_num == prec_sum[-1]

#         tindex = np.asarray(np.where(tgnd == 1)) + 1.0
#         topkmap_ = np.mean(count / (tindex))
#         topkmap = topkmap + topkmap_
#     topkmap = topkmap / num_query
#     index = np.argwhere(recall[:, -1] == 1.0)
#     index = index.squeeze()
#     prec = prec[index]
#     recall = recall[index]
#     cum_prec = np.mean(prec, 0)
#     cum_recall = np.mean(recall, 0)

#     return np.round(topkmap,4) *100, cum_prec, cum_recall

    



def MapWithPR_values(r_database,q_database, thresh = 0.5, percentile = True, topk = 100):
    if percentile:
        r_binary = np.array([((out >= np.percentile(out,thresh))*1).tolist()  for _, out in enumerate(r_database.predictions)])
        q_binary = np.array([((out >= np.percentile(out,thresh))*1).tolist()  for _, out in enumerate(q_database.predictions)])
    else:
        r_binary = np.array([((out >= thresh) * 1).tolist() for _, out in enumerate(r_database.predictions)])
        q_binary = np.array([((out >= thresh) * 1).tolist() for _, out in enumerate(q_database.predictions)])

    train_label = label_binarize(r_database.label_code, classes=[0, 1, 2,3])
    valid_label = label_binarize(q_database.label_code, classes=[0,1, 2,3])

    rB = r_binary
    qB = q_binary
    retrievalL = train_label
    queryL = valid_label
    topk = topk
    topkmap, cum_prec, cum_recall, cum_map = CalcTopMapWithPR(rB, qB, retrievalL, queryL, topk)
  
    return topkmap, cum_prec, cum_recall,cum_map

def MapWithPR_values123(r_database,q_database, thresh = 0.5, percentile = True, topk = 100):
    if percentile:
        r_binary = np.array([((out >= np.percentile(out,thresh))*1).tolist()  for _, out in enumerate(r_database.predictions)])
        q_binary = np.array([((out >= np.percentile(out,thresh))*1).tolist()  for _, out in enumerate(q_database.predictions)])
    else:
        r_binary = np.array([((out >= thresh) * 1).tolist() for _, out in enumerate(r_database.predictions)])
        q_binary = np.array([((out >= thresh) * 1).tolist() for _, out in enumerate(q_database.predictions)])

    train_label = label_binarize(r_database.label_code, classes=[0, 1, 2,3])
    valid_label = label_binarize(q_database.label_code, classes=[0,1, 2,3])

    rB = r_binary
    qB = q_binary
    retrievalL = train_label
    queryL = valid_label
    topk = topk
    topkmap, cum_prec, cum_recall, cum_map = CalcTopMapWithPR123(rB, qB, retrievalL, queryL, topk)
  
    return topkmap, cum_prec, cum_recall,cum_map

def get_data(path, dic_labels):
       
   # Load the MATLAB file
   data = loadmat(path)
   df0 = pd.DataFrame(data['COSFIREdescriptor']['training'][0][0][0][0][0])
   df0['label'] = 'FRI'
   df1 = pd.DataFrame(data['COSFIREdescriptor']['training'][0][0][0][0][1])
   df1['label'] = 'FRII'
   df2 = pd.DataFrame(data['COSFIREdescriptor']['training'][0][0][0][0][2])
   df2['label'] = 'Bent'
   df3 = pd.DataFrame(data['COSFIREdescriptor']['training'][0][0][0][0][3])
   df3['label'] = 'Compact'
   df_train = pd.concat([df0, df1, df2, df3], ignore_index=True)

   df0 = pd.DataFrame(data['COSFIREdescriptor']['testing'][0][0][0][0][0])
   df0['label'] = 'FRI'
   df1 = pd.DataFrame(data['COSFIREdescriptor']['testing'][0][0][0][0][1])
   df1['label'] = 'FRII'
   df2 = pd.DataFrame(data['COSFIREdescriptor']['testing'][0][0][0][0][2])
   df2['label'] = 'Bent'
   df3 = pd.DataFrame(data['COSFIREdescriptor']['testing'][0][0][0][0][3])
   df3['label'] = 'Compact'
   df_test = pd.concat([df0, df1, df2, df3], ignore_index=True)
  
  
   # Rename the columns:
   column_names = ["descrip_" + str(i) for i in range(1, 401)] + ["label_code"]
   df_train.columns = column_names
   df_test.columns = column_names

   #select the optimal number of columns from the classification paper.#Get the optimal 372 descriptors only
   column_list = [f'descrip_{i}' for i in range(1, 373)] + ['label_code']
   df_train = df_train[column_list]
   df_test = df_test[column_list]

     
   df_train['label_code'] = df_train['label_code'].map(dic_labels)
   df_test['label_code'] = df_test['label_code'].map(dic_labels)


   return df_train, df_test


def sanity_check(test_df,train_df, valid_df, dic_labels):
   df_test = pd.DataFrame(test_df.label_code.value_counts())
   tt1 = df_test.loc[dic_labels['Bent']]['label_code'] == 103
   tt2 = df_test.loc[dic_labels['Compact']]['label_code'] == 100
   tt3 = df_test.loc[dic_labels['FRI']]['label_code'] == 100
   tt4 = df_test.loc[dic_labels['FRII']]['label_code'] == 101
   if tt1 and tt2 and tt3 and tt4:
       pass
      #print(f'Test folder is great')
   else:
      raise Exception(f'Test folder is incomplete!!')

   df_train = pd.DataFrame(train_df.label_code.value_counts())

   tt1 = df_train.loc[dic_labels['Bent']]['label_code'] == 305
   tt2 = df_train.loc[dic_labels['Compact']]['label_code'] == 226
   tt3 = df_train.loc[dic_labels['FRI']]['label_code'] == 215
   tt4 = df_train.loc[dic_labels['FRII']]['label_code'] == 434

   if tt1 and tt2 and tt3 and tt4:
       pass
      #print(f'Train folder  is great')
   else:
      raise Exception(f'Test folder  is incomplete!!')
   df_valid = pd.DataFrame(valid_df.label_code.value_counts()) 
   tt1 = df_valid.loc[dic_labels['Bent']]['label_code'] == 100
   tt2 = df_valid.loc[dic_labels['Compact']]['label_code'] == 80
   tt3 = df_valid.loc[dic_labels['FRI']]['label_code'] == 74
   tt4 = df_valid.loc[dic_labels['FRII']]['label_code'] == 144

   if tt1 and tt2 and tt3 and tt4:
       pass
      #print(f'Valid folder  is great')
   else:
      raise Exception(f'Test folder  is incomplete!!')
   #print('##################################################')
   

def get_and_check_data(data_path,data_path_valid,dic_labels):


    train_df, valid_test_df = get_data(data_path,dic_labels)
    _, valid_prev = get_data(data_path_valid,dic_labels)
    print('valid_prev data shape: ', valid_prev.shape)
    
    cols = list(train_df.columns[:10])
    valid_test_df['id'] = range(valid_test_df.shape[0])
            
    valid_df = pd.merge(valid_prev[cols], valid_test_df, on=cols)
    diff_set = set(np.array(valid_test_df.id)) - set(np.array(valid_df.id))
    test_df = valid_test_df[valid_test_df['id'].isin(diff_set)]

    # diff = valid_df[cols]-valid_prev[cols]
    # valid_df = valid_df[~np.isnan(np.array(diff.descrip_1))]
    diff_set_valid = set(np.array(valid_test_df.id)) - set(np.array(test_df.id))
    valid_df = valid_test_df[valid_test_df['id'].isin(diff_set_valid)]
    print(valid_df.label_code.value_counts())
    
    
    valid_df.drop(columns=['id'], inplace=True)
    test_df.drop(columns=['id'], inplace=True)

    #print('##################################################')
    # Verify the data set sizes based on Table 1 of the paper. 
    # print('Train data shape: ', train_df.shape)
    # print('Valid data shape: ', valid_df.shape)                        
    # print('Test data shape: ', test_df.shape)
    
    sanity_check(test_df,train_df, valid_df,dic_labels)

    return train_df,valid_df,test_df



def get_and_check_data_prev(data_path,data_path_valid,data_path_test,dic_labels):
   
    train_df, valid_test_df = get_data(data_path,dic_labels)
    _, valid_prev = get_data(data_path_valid,dic_labels)
    _, test_prev = get_data(data_path_test,dic_labels)
    
    cols = list(train_df.columns[:10])
    valid_test_df['id'] = range(valid_test_df.shape[0])
    test_df = pd.merge(test_prev[cols], valid_test_df, on=cols)

    diff_set = set(np.array(valid_test_df.id)) - set(np.array(test_df.id))
    valid_df = valid_test_df[valid_test_df['id'].isin(diff_set)]
    valid_df.drop(columns=['id'], inplace=True)
    test_df.drop(columns=['id'], inplace=True)

    # Rename label_name column:   
    train_df.rename(columns = {'label_name': 'label_code'}, inplace = True)
    valid_df.rename(columns = {'label_name': 'label_code'}, inplace = True)
    test_df.rename(columns = {'label_name': 'label_code'}, inplace = True)

    #print('##################################################')
    # Verify the data set sizes based on Table 1 of the paper. 
    # print('Train data shape: ', train_df.shape)
    # print('Valid data shape: ', valid_df.shape)                        
    # print('Test data shape: ', test_df.shape)
    
    sanity_check(test_df,train_df, valid_df,dic_labels)

    return train_df,valid_df,test_df

    


def CalcTopMapWithPR1234(referenceBinaryCodes, queryBinaryCodes, referenceLabels, queryLabels, topk):
    '''
    Parameters:
    - referenceBinaryCodes: binary codes of the training set (reference set),
    - queryBinaryCodes: binary codes of the query set,
    - referenceLabels: labels of the training set (reference set),
    - queryLabels: labels of the query set,
    - topk: the number of top retrieved results to consider.

    Returns:
    - topkmap: mean Average Precision (mAP) for the top k retrieved items,
    - cumulativePrecision: cumulative precision over all queries,
    - cumulativeRecall: cumulative recall over all queries,
    - cumulative_mAP: cumulative mean Average Precision over all queries at each retrieval step.
    '''
    
    numQueries = queryLabels.shape[0]  # Number of queries
    numReferences = referenceLabels.shape[0]  # Number of items in the reference set
    precisionMatrix = np.zeros((numQueries, numReferences))  # Precision matrix
    recallMatrix = np.zeros((numQueries, numReferences))  # Recall matrix
    mAPMatrix = np.zeros((numQueries, numReferences))  # mAP matrix for each retrieval step
    
    for i in tqdm(range(numQueries)):
        # Determine the relevance of each reference item to the current query
        relevant = (np.dot(queryLabels[i, :], referenceLabels.transpose()) > 0) * 1  # Binary relevance (1 if relevant, else 0)
        
        # Calculate Hamming distances between the query and all reference items
        hammingDistances = np.count_nonzero(queryBinaryCodes[i] != referenceBinaryCodes, axis=1)
        
        # Sort the indices of the reference items by ascending Hamming distance
        sortedIndices = np.argsort(hammingDistances)
        relevant = relevant[sortedIndices]  # Reorder relevance according to sorted distances

        # Cumulative sum of relevant items retrieved up to each point
        cumulativeRelevantSum = np.cumsum(relevant)
        retrievedItems = np.arange(1, numReferences + 1)  # Index array for all retrieved items
        
        # Precision and recall calculations for the current query
        precisionMatrix[i, :] = cumulativeRelevantSum / retrievedItems
        recallMatrix[i, :] = cumulativeRelevantSum / np.sum(relevant)

        # Calculate mAP at each retrieval step
        precisionAtK = precisionMatrix[i, :] * relevant  # Precision at each retrieved item
        cumulativePrecisionSum = np.cumsum(precisionAtK)  # Cumulative sum of precision values
        relevantIndices = np.cumsum(relevant)  # Number of relevant items retrieved
        with np.errstate(divide='ignore', invalid='ignore'):
            mAPMatrix[i, :] = np.where(relevantIndices > 0, cumulativePrecisionSum / relevantIndices, 0)

    # Filter the precision, recall, and mAP matrices to only include valid queries
    validIndices = np.argwhere(recallMatrix[:, -1] == 1.0).squeeze()
    precisionMatrix = precisionMatrix[validIndices]
    recallMatrix = recallMatrix[validIndices]
    mAPMatrix = mAPMatrix[validIndices]
    
    # Compute cumulative precision, recall, and mAP across all valid queries
    cumulativePrecision = np.mean(precisionMatrix, axis=0)
    cumulativeRecall = np.mean(recallMatrix, axis=0)
    cumulative_mAP = np.mean(mAPMatrix, axis=0)  # Cumulative mAP at each retrieval step

    # Calculate overall top-k mAP
    topkmap = np.mean([mAPMatrix[i, topk-1] for i in range(len(validIndices))])

    return np.round(topkmap, 4) * 100, cumulativePrecision, cumulativeRecall, cumulative_mAP


def MapWithPR_values1234(r_database,q_database, thresh = 0.5, percentile = True, topk = 100):
    if percentile:
        r_binary = np.array([((out >= np.percentile(out,thresh))*1).tolist()  for _, out in enumerate(r_database.predictions)])
        q_binary = np.array([((out >= np.percentile(out,thresh))*1).tolist()  for _, out in enumerate(q_database.predictions)])
    else:
        r_binary = np.array([((out >= thresh) * 1).tolist() for _, out in enumerate(r_database.predictions)])
        q_binary = np.array([((out >= thresh) * 1).tolist() for _, out in enumerate(q_database.predictions)])

    train_label = label_binarize(r_database.label_code, classes=[0, 1, 2,3])
    valid_label = label_binarize(q_database.label_code, classes=[0,1, 2,3])

    rB = r_binary
    qB = q_binary
    retrievalL = train_label
    queryL = valid_label
    topk = topk
    topkmap, cum_prec, cum_recall, cum_map = CalcTopMapWithPR1234(rB, qB, retrievalL, queryL, topk)
  
    return topkmap, cum_prec, cum_recall,cum_map


def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(item) for item in obj]
    else:
        return obj