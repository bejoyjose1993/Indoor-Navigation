import pandas as pd
import numpy as np
import time
from sklearn.cluster import KMeans, AffinityPropagation, SpectralClustering
from scipy.interpolate import griddata
from matplotlib import pyplot as plt
import os
import errno

path_to_database = 'D:\\Project\\Dissertation\\Master\\'
method = 'km'

def load_data(path_to_data):
    # read test data
    FILE_NAME = path_to_data + '/master_data.csv'
    data = pd.read_csv(FILE_NAME,header=0)
    df =  pd.DataFrame(data)
    #train= df.sample(frac=1,random_state=200) #random state is a seed value
    test= []
    #print(len(train))
    #print(len(test))
    return (df,test)
    #return (train,test)

def get_RSS(data):
    df =  pd.DataFrame(data)
    df = df[['0x0001','0x0002','0x0003','0x0004','0x0005','0x0006','0x0012','0x0013','0x0014','0x0015','0x0016']]
    return df
 
def get_geo_cord(data):
    df =  pd.DataFrame(data)
    #print(df[['orientation','Messurement_points']])
    om_df = df[['orientation','Messurement_points']]
    df = df[['x_axis','y_axis','z_axis']]
    #print(om_df)
    return df,om_df   

def distance(a, b):
    return np.sqrt(np.sum(np.power(a-b, 2)))


def cluster_subset_kmeans(clusters, labels, pos, X_test):
    d = []
    for i,c in enumerate(kmeans.cluster_centers_):
        d.append(distance(pos[:2], c[:2]))

    center = np.argmin(d)

    return (ss[center], cs[center])

def cluster_subset_affinityprop(clusters, labels, X_test):
    subset = np.zeros(labels.shape[0]).astype(np.bool)

    d = bdist(clusters, X_test, 5, 1e-3, 1e-25)
    idx = np.argsort(d)[::-1]

    cused = 0
    for c in idx[:5]:
        subset = np.logical_or(subset, c == labels)
        cused += 1

    return (subset, cused)

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

def bayes_position(X_train, y_train, X_test, N, sigma, eps, th, lth, div, y_test):
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


def position_route(method, X_train, y_train, X_test, y_test, clusters, labels,
                   N=5, sigma=5, eps=3e-4, th=1e-25, lth=-85, div=10):

    error = []
    error2D = []
    fdetect = 0
    y_pred = []
    cused = []
    print("clusters")
    print(clusters)
    print(labels.shape)
    tp = pd.DataFrame(X_train)
    cord = pd.DataFrame(y_train)
    tp["x_axis"] = y_train[:,0]
    tp["y_axis"] = y_train[:,1]
    tp["z_axis"] = y_train[:,2]

    tp["orientation"] = om_train['orientation']
    tp["Messurement_points"] = om_train["Messurement_points"]
    tp["labels"] = labels.tolist()
    #tp["cluster_labels"] = "RP-"+labels.tolist()

    export_file_path = 'D:\\Project\\Dissertation\\Master\\new_kmeaan_labels.csv'
    if not os.path.exists(os.path.dirname(export_file_path)):
        try:
            os.makedirs(os.path.dirname(export_file_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    tp.to_csv (export_file_path, index = False, header=True)


    #print(pos)
    for i in range(X_test.shape[0]):
        if i > 1:
            if method=='km':
                subset, c = cluster_subset_kmeans(clusters, labels, pos, X_test[i])
                cused.append(c)
                
            elif method=='ap':
                subset, c = cluster_subset_affinityprop(clusters, labels, X_test[i])
                cused.append(c)
        else:
            subset = np.ones(X_train.shape[0]).astype(np.bool)

        if method=='km':
            #print("subset")
            #print(X_train[subset].shape)
            #print(y_train[subset].shape)
            #print(len(cused)) 
            pos, q = bayes_position(X_train[subset], y_train[subset], X_test[i], N, sigma,
                                    eps, th, lth, div, y_test[i])
            #print(pos)
            #print(q) 
            if q > 50:
                pos, _ = bayes_position(X_train, y_train, X_test[i], N, sigma,
                                        eps, th, lth, div, y_test[i])
        elif method=='ap':
            pos, _ = bayes_position(X_train[subset], y_train[subset], X_test[i], N, sigma,
                                    eps, th, lth, div, y_test[i])

        pos[2] = floors[np.argmin(np.abs(floors-pos[2]))]

        if i > 1:
            y_pred.append(pos)
            error.append(distance(y_test[i], y_pred[-1]))
            fdetect += y_pred[-1][2] == y_test[i][2]
            # 2D error only if floor was detected correctly
            if y_pred[-1][2] == y_test[i][2]:
                error2D.append(distance(y_test[i,0:2], np.array(y_pred[-1])[0:2]))

    return (np.array(y_pred), np.array(error), np.array(error2D), fdetect, np.array(cused))


tsum = 0
#Load Data
train, test = load_data(path_to_database)

#Get Train RSSI And Geo Cordinates
x_train = get_RSS(train)
x_train[x_train==-79] = np.nan

y_train, om_train = get_geo_cord(train)

#Get Test RSSI And Geo Cordinates
#x_test = get_RSS(test)
#x_test[x_test==-79] = np.nan

#y_test, om_test = get_geo_cord(test)

x_train = x_train.to_numpy()
y_train = y_train.to_numpy()   
#x_test = x_test.to_numpy()
#y_test = y_test.to_numpy()   

# prepare data for processing
ap_count = x_train.shape[1]
floors = np.unique(y_train[:,2])

x_ktrain = x_train.copy()
y_ktrain = y_train.copy()

#X_aux = x_ktrain.copy()
#X_aux[np.isnan(X_aux)] = 0

#M = x_ktrain.shape[1]

#corr = np.zeros((M,M))
#cth = 500
#keep = np.ones(M).astype(np.bool)
#print(np.sum(corr[1,:] < cth))
#print(np.abs(X_aux[:,1] - X_aux[:,2]).shape)
#for i in range(M):
#    for j in range(i,M):
#        if i != j:
#            diff = np.abs(X_aux[:,i] - X_aux[:,j])
#            corr[i,j] = corr[j,i] = np.sum(diff)
#        else:
#            corr[i,j] = cth

#    if keep[i] and np.sum(corr[i,:] < cth) > 0:
#        for p in np.where(corr[i,:] < cth)[0]:
#            keep[p] = False
#print(corr)               
#print(keep)           
#print(x_ktrain.shape)
#x_ktrain = x_ktrain[:,keep]
#x_test = x_test[:,keep]

print(x_ktrain,om_train)

if method=='km':
    C = 6

    kmeans = KMeans(n_clusters=C, n_init=500, n_jobs=2, tol=1e-9)
    labels = kmeans.fit_predict(y_ktrain)
    clusters = kmeans.cluster_centers_
    print("clusters")
    print(np.unique(labels))
    N = x_ktrain.shape[0]
    aux = np.zeros((C,C))
    #print(labels.shape)
    for i in range(N):
        dist = np.zeros(N)
        for j in range(N):
            dist[j] = distance(y_ktrain[i], y_ktrain[j])

        idx = np.argsort(dist)
        #print(labels[idx])
        for p in np.where(labels[idx] != labels[i])[0]:
            if dist[idx[p]] < 10:
                #if dist[idx[p]] < 10:
                aux[labels[i],labels[idx[p]]] += 1

    ss = np.zeros((C,labels.size)).astype(np.bool)
    cs = np.zeros(C)
    rssl = []
    rssc = []
    for c in range(C):
        aux[c,c] = 1

        for i in np.where(aux[c] != 0)[0]:
            ss[c] = np.logical_or(ss[c], labels == i)
            cs[c] += 1

elif method=='ap':
    N = x_ktrain.shape[0]
    affinity = np.zeros((N,N))
    for i in range(N):
        affinity[i,:] = bdist(x_ktrain, x_ktrain[i], 5, 1e-3, 1e-25)
    print("Start AffinityPropagation clusters")
    cluster = AffinityPropagation(damping=0.5, affinity='precomputed')
    print("affinity")
    print(affinity.shape)
    #print(affinity)
    labels = cluster.fit_predict(affinity)
    print(labels)
    C = np.unique(labels).size
    clusters = x_ktrain[cluster.cluster_centers_indices_]
    print(clusters)

else:
    print('Unknown method. Please choose either "km" or "ap".')
    quit()

print(x_ktrain)
print(om_train,labels)
print(labels)
print(head())

#t = time.process_time()
# estimate positions for test data
#y, error3D, error2D, fdetect, cused = position_route(method, x_ktrain,
#            y_ktrain, x_test, y_test, clusters, labels, N=5, eps=1e-3)
#tsum += time.process_time() - t

print('Mean positioning error 2D: \t%.3lf m' % np.mean(error2D))
print('Mean positioning error 3D: \t%.3lf m' % np.mean(error3D))
print('Floor detection rate: \t\t%2.2lf %%' % ((float(fdetect) / error3D.shape[0])*100))



