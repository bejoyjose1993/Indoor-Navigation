# Just Visualization
import pandas as pd                     #to read file
import numpy as np                      #to cal
import matplotlib.pyplot as plt         #plot 
from mpl_toolkits.mplot3d import Axes3D #for 3D plot
from sklearn.cluster import KMeans      #use kmeans
from sklearn.metrics import silhouette_samples, silhouette_score #to cal the silhouette_score
from numpy import *
method = 'km'
path_to_database = 'D:\\Project\\Dissertation\\Master\\'
test_raw_dir ="D:\\Project\\Dissertation\\raw_example.csv"
def load_data(path_to_data):
    # read test data
    FILE_NAME = path_to_data + '/master_data.csv'
    data = pd.read_csv(FILE_NAME,header=0)
    #df =  pd.DataFrame(data)
    #train= df.sample(frac=1,random_state=200) #random state is a seed value
    return data

def get_RSS(data):
    df =  pd.DataFrame(data)
    rss_df = df[['0x0001','0x0002','0x0003','0x0004','0x0005','0x0006','0x0012','0x0013','0x0014','0x0015','0x0016']]
    loc_df = df[['x_axis','y_axis','z_axis']]
    return loc_df, rss_df

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
    for i,c in enumerate(kmeans.cluster_centers_):
        d.append(distance(pos[:2], c[:2]))

    center = np.argmin(d)

    return (ss[center], cs[center])

def bayes_position(X_train, y_train, X_test, N, sigma, eps, th, lth, div, y_test):
    diff = X_train - X_test

    proba = 1/(np.sqrt(2*np.pi)*sigma)*np.exp( \
        -np.power(diff, 2)/(2.0*sigma**2))

    proba[np.isnan(proba)] = eps
    proba[proba < th] = eps
    proba = np.log(proba)
    cost = np.sum(proba, axis=1)

    inv = np.zeros(X_train.shape[0])
    #X_train_e = pd.DataFrame(X_train)
    #print(X_train_e.iloc[1])
    X_train = X_train.values
    #print(X_train.shape[0])
    #print(X_train_temp[1])
    #print("My")
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


def position_route(method, X_train, y_train, X_test, y_test, clusters, labels,
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
            #print("subset")
            #print(X_train)
            #print(X_train[subset])
            #print(X_train[subset].shape)
            #print(y_train[subset].shape)
            #print(len(cused)) 
            pos, q = bayes_position(X_train[subset], y_train[subset], X_test[i], N, sigma,
                                    eps, th, lth, div, y_test)
            #print(pos)
            #print(q) 
            if q > 50:
                pos, _ = bayes_position(X_train, y_train, X_test[i], N, sigma,
                                        eps, th, lth, div, y_test)

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





train = load_data(path_to_database)
try1, rss_df  = get_RSS(train)
#Stry1[isnan(try1)] = -79

data  = load_test_data(test_raw_dir)
my_data = format_data(data)
x_test = get_req_test_data(my_data)


matrix = try1.values
print(matrix)
print(type(matrix))

n_clusters = 11                                              # number of clusters
kmeans = KMeans(n_clusters=n_clusters)   # use kmeans 
cluster_labels = kmeans.fit_predict(matrix)               # get the labels
print("cluster_labels")
#print(cluster_labels)
clust = kmeans.cluster_centers_
#Test Matrix
#x_test = x_test.fillna(0)
test_mean = x_test.mean(axis = 0).to_frame().transpose()
test_matrix = test_mean.values
print(test_matrix)

#Saving Cluster Details
new_cluster_df = train
new_cluster_df['cluster_labels'] = cluster_labels
new_cluster_df.to_csv ("D:\\Project\\Dissertation\\Master\\master_with_kmean_cluster.csv", index = False, header=True)

#Saving Cluster Centers
new_cluster = pd.DataFrame(clust)
new_cluster.to_csv ("D:\\Project\\Dissertation\\Master\\kmeans_cluster_centers.csv", index = False, header=True)




#print(clust)
C = n_clusters
N = rss_df.shape[0]
aux = np.zeros((C,C))

for i in range(N):
    dist = np.zeros(N)
    for j in range(N):
        dist[j] = distance(matrix[i], matrix[j])
    idx = np.argsort(dist)
    print(cluster_labels[idx])
    for p in np.where(cluster_labels[idx] != cluster_labels[i])[0]:
        if dist[idx[p]] < 10:
            aux[cluster_labels[i],cluster_labels[idx[p]]] += 1
ss = np.zeros((C,cluster_labels.size)).astype(np.bool)
cs = np.zeros(C)

rssl = []
rssc = []
for c in range(C):
    aux[c,c] = 1
    for i in np.where(aux[c] != 0)[0]:
        ss[c] = np.logical_or(ss[c], cluster_labels == i)
        cs[c] += 1

cs_temp = pd.DataFrame(cs)
ss_temp = pd.DataFrame(ss)

cs_temp.to_csv ("D:\\Project\\Dissertation\\Master\\dump\\cs_kmean.csv", index = False)
ss_temp.to_csv ("D:\\Project\\Dissertation\\Master\\dump\\ss_kmean.csv", index = False)


y_test = [0,1,1,1]

y, error3D, error2D, fdetect, cused = position_route(method, rss_df,
            matrix, test_matrix, y_test, clust, cluster_labels, N=5, eps=1e-3)



plt.figure(figsize=(20, 10), dpi=80)  # set fig
plt.subplot(1, 1, 1)                 # set the plot size
N = 6


number=[np.sum(cluster_labels == 0), np.sum(cluster_labels == 1),np.sum(cluster_labels == 2),
         np.sum(cluster_labels == 3),np.sum(cluster_labels == 4),np.sum(cluster_labels == 5),np.sum(cluster_labels == 6),
         np.sum(cluster_labels == 7),np.sum(cluster_labels == 8),np.sum(cluster_labels == 9),np.sum(cluster_labels == 10)]

number.sort()

values = (number[0],number[1],number[2],number[3],number[4],number[5]) # set values for dirrerent bars

index = np.arange(N)                                                # num of each bar
width = 0.3                                                         # width of the bar
p2 = plt.bar(index, values, width, label="--", color="#87CEFA")     # plot set the color

plt.xlabel('Cluster')                               # set xlabel
plt.ylabel('Number')                                # set ylabel
plt.title('K-Means Result')                         # set title

plt.xticks(index, ('cluster-1', 'cluster-2','cluster-3','cluster-4','cluster-5','cluster-6','cluster-7', 'cluster-8','cluster-9','cluster-10','cluster-11'))       # set the name of each bar
plt.yticks(np.arange(0, 2000, 100))                   # from 0~35000(50000 for each separate)
plt.legend(loc="upper right")
plt.show()    