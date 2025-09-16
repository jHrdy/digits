# find out the prediction of the model
# fit DBSCAN for the specific class (if 3 is predicted fit dbscan with tensors labeled as 3)
# for experimental purposes lets visualize outlier score (using knn or OCSVM)

from sklearn.cluster import DBSCAN
from load_data import data
import pandas as pd
import torch
from model import Net

from model import Net

def get_dataframe():
    X = data.data.numpy()
    y = data.targets.numpy()
    X = X.reshape(X.shape[0], -1)  

    df = pd.DataFrame(X)
    df['label'] = y
    print(df.columns)    
    return df

get_dataframe()

# debug this

def is_outlier(prediction_tensor, prediction_label):
    df = get_dataframe()
    df_subset = df[df['label'] == prediction_label]
    
    temp_df = {i : prediction_tensor[i] for i in range(len(prediction_tensor))}
    temp_df['label'] = prediction_label

    temp_df = pd.DataFrame(temp_df)

    print(temp_df.head())
    exit()
    df_subset_with_new_value = pd.concat([df_subset, temp_df], axis=1)
    dbscan = DBSCAN()
    return dbscan.fit_predict(df_subset_with_new_value)
from test_script import trainset

x, label = trainset.__getitem__(0)
print(is_outlier(x, label))