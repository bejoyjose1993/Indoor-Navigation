# Just Visualization
import pandas as pd                     #to read file
import numpy as np                      #to cal
import matplotlib.pyplot as plt         #plot 
from mpl_toolkits.mplot3d import Axes3D #for 3D plot
from sklearn.cluster import AffinityPropagation      #use kmeans
from sklearn.metrics import silhouette_samples, silhouette_score #to cal the silhouette_score
from numpy import *
from pandas import DataFrame
import mysql.connector
import time
from statistics import mode, mean 

method = 'ap'
path_to_database = 'D:\\Project\\Dissertation\\Master\\'
test_raw_dir ="D:\\Project\\Dissertation\\raw_example.csv"

mydb = mysql.connector.connect(
    host = "localhost",
    user = "root",
    password = "root",
    database = "indoor_nav_using_affinity_prop"
)
def load_data():
    # read test data
    my_cusor = mydb.cursor()
    block = []
    select_stmt = "SELECT * FROM indoor_nav_using_affinity_prop.affinity_prop_master"
    my_cusor.execute(select_stmt)
    data = DataFrame(my_cusor.fetchall())
    field_names = [i[0] for i in my_cusor.description]
    data.columns = field_names  
    my_cusor.close()    
    return data

def get_RSS(data):
    df =  pd.DataFrame(data)
    rss_df = df[['0x0001','0x0002','0x0003','0x0004','0x0005','0x0006','0x0012','0x0013','0x0014','0x0015','0x0016']]
    loc_df = df[['x_axis','y_axis','z_axis']]
    #loc_df = df[['reference_points','messurement_points']]
    return rss_df, loc_df  

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

def bdist(a, b, sigma, eps, th, lth=-85, div=10):
    diff = a - b

    #Let us compare the sampled distribution to the analytical distribution. 
    #We generate a large set of samples, and calculate the probability of getting 
    # each value using the matplotlib.pyplot.hist command. 
    #https://kitchingroup.cheme.cmu.edu/pycse/pycse.html
    proba = 1/(np.sqrt(2*np.pi)*sigma)*np.exp( \
        -np.power(diff, 2)/(2.0*sigma**2))

    proba[np.isnan(proba)] = eps
    proba[proba < th] = eps
    proba = np.log(proba)
    if a.ndim == 2:
        cost = np.sum(proba, axis=1)
    else:
        cost = np.sum(proba)

    inv = np.zeros(a.shape[0])
    for i in range(a.shape[0]):
        aa = np.logical_and(~np.isnan(a[i]), np.isnan(b))
        bb = np.logical_and(np.isnan(a[i]), ~np.isnan(b))

        nfound = np.concatenate((a[i,aa], b[bb]))
        for v in nfound[nfound > lth]:
            inv[i] += v - lth

    inv /= div
    cost -= inv

    return cost

def cluster_subset_affinityprop(clusters, labels, X_test):
    subset = np.zeros(labels.shape[0]).astype(np.bool)

    d = bdist(clusters, X_test, 5, 1e-3, 1e-25)
    idx = np.argsort(d)[::-1]

    cused = 0
    for c in idx[:5]:
        subset = np.logical_or(subset, c == labels)
        cused += 1

    return (subset, cused)

def bayes_position(X_train, y_train, X_test, N, sigma, eps, th, lth, div):
    diff = X_train - X_test

    proba = 1/(np.sqrt(2*np.pi)*sigma)*np.exp( \
        -np.power(diff, 2)/(2.0*sigma**2))

    proba[np.isnan(proba)] = eps
    proba[proba < th] = eps
    proba = np.log(proba)
    cost = np.sum(proba, axis=1)

    inv = np.zeros(X_train.shape[0])
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
    pos_pred = []
    cused = []
    x_pred = []
    y_pred = []
    for i in range(X_test.shape[0]):
        if i > 1:
            if method=='ap':
                subset, c = cluster_subset_affinityprop(clusters, labels, X_test[i])
                cused.append(c)    
        else:
            subset = np.ones(X_train.shape[0]).astype(np.bool)

        if method=='ap':
            pos, _ = bayes_position(X_train[subset], y_train[subset], X_test[i], N, sigma,
                                    eps, th, lth, div)
        #pos[2] = floors[np.argmin(np.abs(floors-pos[2]))]

        if i > 1:
            pos_pred.append(pos)

    print("y_pred")
    print(pos_pred)
    for pred in pos_pred :
        x_pred.append(pred[0])
        y_pred.append(pred[1])

    #x = Counter(x_pred)
    x_max_count = mode(x_pred)
    y_max_count = mode(y_pred)
    pred = [round(x_max_count,1),round(y_max_count,1)]
    print(pred)
    return (np.array(pos_pred), np.array(error), np.array(error2D), fdetect, np.array(cused), pred)
################ END KMEANS PREDICTION MODEL SESSION  ################  

def get_absolute_loc_for_nav(pred):
    my_cusor = mydb.cursor()
    block = []
    select_stmt = "SELECT block FROM indoor_nav_using_affinity_prop.geo_location_cord where x_axis = %(x_axis)s and y_axis =  %(y_axis)s"
    my_cusor.execute(select_stmt,{ 'x_axis': pred[0] , 'y_axis': pred[1] })
    rows = my_cusor.fetchall()
    for r in rows:
        block.append(r[0])  

    if(len(block)<1):
        select_stmt = "select block, block_area from indoor_nav_using_affinity_prop.geo_location_cord order by abs(x_axis - %(x_axis)s) + abs(y_axis - %(y_axis)s) limit 1"
        my_cusor.execute(select_stmt,{ 'x_axis': pred[0] , 'y_axis': pred[1] })
        rows = my_cusor.fetchall()
        for r in rows:
            block = (r[1].split())    
    my_cusor.close()    
    return block

train = load_data()
X_train, y_train  = get_RSS(train)

data  = load_test_data(test_raw_dir)
my_data = format_data(data)
x_test = get_req_test_data(my_data)
#test_matrix = x_test.values
test_mean = x_test.mean(axis = 0).to_frame().transpose()
test_matrix_new = test_mean.values
test_matrix_new = np.vstack((test_matrix_new, test_mean.values))
test_matrix = np.vstack((test_matrix_new, test_mean.values))

print(test_matrix)



#X_train = X_train.fillna(0)
X_ktrain = X_train.values
y_ktrain = y_train.values
#print(X_train.head())

N = X_ktrain.shape[0]
affinity = np.zeros((N,N))
for i in range(N):
    affinity[i,:] = bdist(X_ktrain, X_ktrain[i], 5, 1e-3, 1e-25)

cluster = AffinityPropagation(damping=0.5, affinity='precomputed')
labels = cluster.fit_predict(affinity)
C = np.unique(labels).size
clusters = X_ktrain[cluster.cluster_centers_indices_]




tsum = 0
t = time.process_time()
# estimate positions for test data
y, error3D, error2D, fdetect, cused, pred = position_route(method, X_ktrain,
            y_ktrain, test_matrix, clusters, labels, N=5, eps=1e-3)

print("Init_pred")
print(pred)
pred_block = get_absolute_loc_for_nav(pred)

print("get_absolute_loc_for_nav")
print(pred_block)

tsum += time.process_time() - t