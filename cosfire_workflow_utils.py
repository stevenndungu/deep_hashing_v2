# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 10:25:28 2023

@author: P307791
"""

import glob
import os, random, string
import torch
import torch.nn as nn
import argparse
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
os.environ['PYTHONHASHSEED'] = 'python'

from sklearn.preprocessing import label_binarize

from torch.linalg import vector_norm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import random
from PIL import Image

import torch
from torch.linalg import vector_norm
from sklearn.manifold import TSNE
from sklearn import preprocessing


# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 10:25:28 2023

@author: P307791
"""
import glob
import os, random, string

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

dic_labels = { 'Bent':2,
  'Compact':3,
    'FRI':0,
    'FRII':1
}

# dic_labels = { 2:'Bent',
#                 3:'Compact',
#                   0:'FRI',
#                   1: 'FRII'
#               }


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
    tsum = np.sum(top_rel) 

    #skips the iteration if there are no relevant results.
    if tsum == 0:
        continue

    pr_count = np.linspace(1, tsum, tsum) 
    tindex = np.asarray(np.where(top_rel == 1)) + 1.0 #is the indices where top_rel is equal to 1 (i.e., the positions of relevant images)
    pr = pr_count / tindex # precision
    mAP_sub = np.mean(pr) # AP
    mAP[i] = mAP_sub 
      


  return np.round(np.mean(mAP),4) *100 #mAP


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
    mAP = SimplifiedTopMap(rB, qB, retrievalL, queryL, topk)
  
    return np.round(mAP,4), r_binary, train_label, q_binary, valid_label


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
   column_names = ["descrip_" + str(i) for i in range(1, 401)] + ["label_name"]
   df_train.columns = column_names
   df_test.columns = column_names


   #select the optimal number of columns from the classification paper.#Get the optimal 372 descriptors only
   column_list = [f'descrip_{i}' for i in range(1, 373)] + ['label_name']
   df_train = df_train[column_list]
   df_test = df_test[column_list]

     


   #select the optimal number of columns from the classification paper.#Get the optimal 372 descriptors only
   column_list = [f'descrip_{i}' for i in range(1, 373)] + ['label_name']
   df_train = df_train[column_list]
   df_test = df_test[column_list]

   # dic_labels = { 'Bent':2,
   #                'Compact':3,
   #                   'FRI':0,
   #                   'FRII':1
   #             }

  

   # df_train['label_name'] = df_train['label_code'].map(dic_labels)
   # df_test['label_name'] = df_test['label_code'].map(dic_labels)


   return df_train, df_test


def sanity_check(test_df,train_df, valid_df):
   df_test = pd.DataFrame(test_df.label_name.value_counts())
   tt1 = df_test.loc['Bent']['label_name'] == 103
   tt2 = df_test.loc['Compact']['label_name'] == 100
   tt3 = df_test.loc['FRI']['label_name'] == 100
   tt4 = df_test.loc['FRII']['label_name'] == 101
   if tt1 and tt2 and tt3 and tt4:
      #print(f'Test folder is great')
      pass
   else:
      raise Exception(f'Test folder is incomplete!!')

   df_train = pd.DataFrame(train_df.label_name.value_counts())

   tt1 = df_train.loc['Bent']['label_name'] == 305
   tt2 = df_train.loc['Compact']['label_name'] == 226
   tt3 = df_train.loc['FRI']['label_name'] == 215
   tt4 = df_train.loc['FRII']['label_name'] == 434

   if tt1 and tt2 and tt3 and tt4:
      pass
      #print(f'Train folder  is great')
   else:
      raise Exception(f'Test folder  is incomplete!!')
   df_valid = pd.DataFrame(valid_df.label_name.value_counts()) 
   tt1 = df_valid.loc['Bent']['label_name'] == 100
   tt2 = df_valid.loc['Compact']['label_name'] == 80
   tt3 = df_valid.loc['FRI']['label_name'] == 74
   tt4 = df_valid.loc['FRII']['label_name'] == 144

   if tt1 and tt2 and tt3 and tt4:
      pass
      #print(f'Valid folder  is great')
   else:
      raise Exception(f'Test folder  is incomplete!!')
   print('##################################################')
   

def get_and_check_data(data_path,data_path_valid,dic_labels):


    train_df, valid_test_df = get_data(data_path,dic_labels)
    _, valid_prev = get_data(data_path_valid,dic_labels)
    #print('valid_prev data shape: ', valid_prev.shape)
    
    cols = list(train_df.columns[:15])
    valid_test_df['id'] = range(valid_test_df.shape[0])
            
    valid_df = pd.merge(valid_prev[cols], valid_test_df, on=cols)
    diff_set = set(np.array(valid_test_df.id)) - set(np.array(valid_df.id))
    test_df = valid_test_df[valid_test_df['id'].isin(diff_set)]
    #print(valid_df.label_code.value_counts())
    
    
    valid_df.drop(columns=['id'], inplace=True)
    test_df.drop(columns=['id'], inplace=True)

    #print('##################################################')
    # Verify the data set sizes based on Table 1 of the paper. 
   #  print('Train data shape: ', train_df.shape)
   #  print('Valid data shape: ', valid_df.shape)                        
   #  print('Test data shape: ', test_df.shape)
    
    sanity_check(test_df,train_df, valid_df)

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

   #  train_df['label_name'] = train_df['label_code'].map(dic_labels)
   #  test_df['label_name'] = test_df['label_code'].map(dic_labels)
   #  valid_df['label_name'] = valid_df['label_code'].map(dic_labels)

   #  # Rename label_name column:   
   #  train_df.rename(columns = {'label_name': 'label_code'}, inplace = True)
   #  valid_df.rename(columns = {'label_name': 'label_code'}, inplace = True)
   #  test_df.rename(columns = {'label_name': 'label_code'}, inplace = True)

   #  print('##################################################')
   #  # Verify the data set sizes based on Table 1 of the paper. 
   #  print('Train data shape: ', train_df.shape)
   #  print('Valid data shape: ', valid_df.shape)                        
   #  print('Test data shape: ', test_df.shape)
    
    sanity_check(test_df,train_df, valid_df)

    return train_df,valid_df,test_df

    


    