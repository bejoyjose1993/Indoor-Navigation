# Just Visualization
import pandas as pd                     #to read file
import numpy as np                      #to cal
import matplotlib.pyplot as plt         #plot 
from mpl_toolkits.mplot3d import Axes3D #for 3D plot
from sklearn.cluster import KMeans      #use kmeans
from sklearn.metrics import silhouette_samples, silhouette_score #to cal the silhouette_score
from numpy import *
method = 'km'
path_to_kmeans_data = 'D:\\Project\\Dissertation\\Master\\dump'
test_raw_dir ="D:\\Project\\Dissertation\\raw_example.csv"


def load_kmeans_data(path_to_data):
    # read test data
    FILE_NAME = path_to_data + '/new_cluster_df.csv'
    data = pd.read_csv(FILE_NAME,header=0)
    return data

def load_cluster_centers(path_to_data):
    FILE_NAME = path_to_data + '/new_cluster.csv'
    clust = pd.read_csv(FILE_NAME,header=0)
    cs_temp = pd.read_csv(path_to_data + '/cs_temp.csv',header=0)
    ss_temp = pd.read_csv(path_to_data + '/ss_temp.csv',header=0)
    return clust, cs_temp, ss_temp

def get_all_cluster_data(data):
    df =  pd.DataFrame(data)
    rss_df = df[['0x0001','0x0002','0x0003','0x0004','0x0005','0x0006','0x0012','0x0013','0x0014','0x0015','0x0016']]
    loc_df = df[['x_axis','y_axis','z_axis']]
    loc_df = df[['x_axis','y_axis','z_axis']]
    cluster_labels = df[['cluster_labels']]
    return loc_df, rss_df, cluster_labels

def load_test_data(path_to_data):
    data = pd.read_csv(path_to_data, names=['date','time','type','id_0','id_1','id_2','sensor_data','temperature','humidity','raw_data'])
    return (data)   

def format_data(data):
    #Empty data Frame
    initdata =[]
    columns = ['date','time','0x0001','0x0002', '0x0003','0x0004','0x0005', '0x0006','0x0012','0x0013', '0x0014','0x0015','0x0016']
    #columns = ['date','time']
    dataframe1 = pd.DataFrame(initdata,columns=columns)
    df = data.iloc[1:]
    df = df.set_index("time", drop = False)
    time = df.time.unique()
    for tym in time:
        combinedList = []
        list1 = ["",""]
        date = ""
        dataf = None
        for i in range(len(df)):
            dataList = []
            if(df.time[i] == tym):
                dataList.append(df.id_2[i].strip().replace(" ",""))    
                rssi_val = compute_RSSI(df.raw_data[i].strip().replace(" ",""))
                dataList.append(rssi_val)
                #dataList.append(df.raw_data[i].strip().replace(" ",""))
                date = df.date[i].strip()
                combinedList.append(dataList)   
        dataf = pd.DataFrame(combinedList).T
        index = np.array(dataf[0:1]).squeeze()
        if type(index.tolist()) == str:
            list1.append(index.tolist()) 
            list1 = list1[2:]
        else:
            list1 = index.tolist()

        dataf.columns = list1
        dataf = dataf[1:]
        dataf.insert(loc=0, column='date', value=date)
        dataf.insert(loc=1, column='time', value=tym)
        dataframe1 = pd.concat([dataframe1,dataf])
    dataframe1 = dataframe1.fillna(method='pad')
    dataframe1 = dataframe1.fillna(method='bfill')
    #print(dataframe1)
    return dataframe1

def get_req_test_data(data):
    df =  pd.DataFrame(data)
    df = df[['0x0001','0x0002','0x0003','0x0004','0x0005','0x0006','0x0012','0x0013','0x0014','0x0015','0x0016']]
    return df

def compute_RSSI(raw_data):
    hex_val = raw_data[-2:]
    int_val = int(hex_val, 16)
    rssi_val = int_val - 256
    return rssi_val

def distance(a, b):
    return np.sqrt(np.sum(np.power(a-b, 2)))


def cluster_subset_kmeans(clusters, labels, pos, X_test):
    d = []
    for i,c in enumerate(clust):
        d.append(distance(pos[:2], c[:2]))

    center = np.argmin(d)

    return (ss[center], cs[center])

def bayes_position(X_train, y_train, X_test, N, sigma, eps, th, lth, div):
    diff = X_train - X_test

    proba = 1/(np.sqrt(2*np.pi)*sigma)*np.exp( \
        -np.power(diff, 2)/(2.0*sigma**2))

    proba[np.isnan(proba)] = eps
    proba[proba < th] = eps
    proba = np.log(proba)
    cost = np.sum(proba, axis=1)

    inv = np.zeros(X_train.shape[0])
    X_train = X_train.values
    for i in range(X_train.shape[0]):
        a = np.logical_and(~np.isnan(X_train[i]), np.isnan(X_test))
        b = np.logical_and(np.isnan(X_train[i]), ~np.isnan(X_test))

        nfound = np.concatenate((X_train[i,a], X_test[b]))
        for v in nfound[nfound > lth]:
            inv[i] += v - lth

    inv /= div
    cost -= inv

    idx = np.argsort(cost)[::-1]

    bias = 3
    position = np.zeros(3)
    N = min(N, y_train.shape[0])
    idx = idx.to_numpy()
    for i in range(N):
        weight = N-i
        if i == 0:
            weight += bias    
        position += weight*y_train[idx[i]]

    position /= N*(N+1)/2+bias

    return (np.array(position), np.mean(inv[idx[:20]]))


def position_route(method, X_train, y_train, X_test, clusters, labels,
                   N=5, sigma=5, eps=3e-4, th=1e-25, lth=-85, div=10):

    error = []
    error2D = []
    fdetect = 0
    y_pred = []
    cused = []
    for i in range(X_test.shape[0]):
        if i > 1:
            if method=='km':
                subset, c = cluster_subset_kmeans(clusters, labels, pos, X_test[i])
                cused.append(c)
        else:
            subset = np.ones(X_train.shape[0]).astype(np.bool)

        if method=='km':
            pos, q = bayes_position(X_train[subset], y_train[subset], X_test[i], N, sigma,
                                    eps, th, lth, div)

            if q > 50:
                pos, _ = bayes_position(X_train, y_train, X_test[i], N, sigma,
                                        eps, th, lth, div)

        if i > 1:
            y_pred.append(pos)
#            error.append(distance(y_test[i], y_pred[-1]))
#            fdetect += y_pred[-1][2] == y_test[i][2]
#            # 2D error only if floor was detected correctly
#            if y_pred[-1][2] == y_test[i][2]:
#                error2D.append(distance(y_test[i,0:2], np.array(y_pred[-1])[0:2]))
    print("y_pred")
    print(y_pred[-1])

    return (np.array(y_pred), np.array(error), np.array(error2D), fdetect, np.array(cused))

#Get Train Data 
train = load_kmeans_data(path_to_kmeans_data)
try1, rss_df, cluster_labels  = get_all_cluster_data(train)
matrix = try1.values

#Get Test Data 
data  = load_test_data(test_raw_dir)
my_data = format_data(data)
x_test = get_req_test_data(my_data)

#Get Saved Cluster Data 
clust_ld, cs_temp, ss_temp = load_cluster_centers(path_to_kmeans_data)
clust = clust_ld.to_numpy()
cs = cs_temp.to_numpy()
ss = ss_temp.to_numpy()
test_matrix = x_test.values

#Predict Code
y, error3D, error2D, fdetect, cused = position_route(method, rss_df,
            matrix, test_matrix, clust, cluster_labels, N=5, eps=1e-3)