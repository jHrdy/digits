# find out the prediction of the model
# fit DBSCAN for the specific class (if 3 is predicted fit dbscan with tensors labeled as 3)
# for experimental purposes lets visualize outlier score (using knn or OCSVM)

from sklearn.svm import OneClassSVM
from load_data import data
import pandas as pd
import torch
from plot_tensor import plot_tensor
import matplotlib.pyplot as plt

def get_dataframe():
    X = data.data.numpy()
    y = data.targets.numpy()
    X = X.reshape(X.shape[0], -1)  

    df = pd.DataFrame(X)
    df['label'] = y
    print(df.columns)    
    return df

#get_dataframe()

# continue working on this -> add feature to append user input to dataset and preditc its outlier measure

def is_outlier(input_tensor, predicted_label=3):
    
    from test_script import trainset
    from model import Net

    # load data into numpy
    X = data.data.numpy()
    y = data.targets.numpy()
    X = X.reshape(X.shape[0], -1)
    
    # create dataframe
    df = pd.DataFrame(X)
    df['label'] = y   

    # filter for predicted label
    df_N = df[df['label'] == predicted_label]

    # compute outlier measure
    out_detector = OneClassSVM()
    out_detector.fit(df_N[:-1].values)
    out_detector.predict(df_N[:-1].values)
    out_measure = out_detector.score_samples(df_N[:-1].values)
    
    # plot outlier measure
    plt.scatter(range(len(out_measure)), out_measure)
    plt.show()
    
    # print index on lowest scoring point (OCSVM assigns low vals for likely outliers)
    print(out_measure.argmin())

# rand img for testing
random_image = torch.rand(1, 28, 28).flatten()
from plot_tensor import plot_tensor
plot_tensor(random_image)
#is_outlier()