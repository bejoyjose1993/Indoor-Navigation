# Just Visualization
import pandas as pd                     #to read file
import numpy as np                      #to cal
import matplotlib.pyplot as plt         #plot 
from mpl_toolkits.mplot3d import Axes3D #for 3D plot
from sklearn.cluster import AffinityPropagation      #use kmeans
from sklearn.metrics import silhouette_samples, silhouette_score #to cal the silhouette_score
from sklearn.model_selection import train_test_split
from numpy import *
from pandas import DataFrame
import mysql.connector
import time
from statistics import mode, mean 
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, ConfusionMatrixDisplay


method = 'ap'
path_to_database = 'D:\\Project\\Dissertation\\Master\\'
test_raw_dir ="D:\\Project\\Dissertation\\raw_example.csv"

mydb = mysql.connector.connect(
    host = "localhost",
    user = "root",
    password = "root",
    database = "indoor_nav_using_affinity_prop"
)

#Using Test 21 vehicle density data 

#path_to_test_data = "D:\\Project\\Dissertation\\Test Data\\Test_with_21_vehicles\\Test Master\\total_test_master.csv"

#Using Test 13 vehicle density data 
#path_to_test_data = "D:\\Project\\Dissertation\\Test Data\\Test_with_13_vehicles\\Test Master\\total_test_master.csv"

def load_data():
    # read test data
    my_cusor = mydb.cursor()
    block = []

    #Normal affinity propagation
    select_stmt = "SELECT * FROM indoor_nav_using_affinity_prop.affinity_prop_master"

    #Affinity Propagation using 21 vehicle density
    #select_stmt = "SELECT * FROM indoor_nav_using_affinity_prop.affinity_evaluation_master"


    my_cusor.execute(select_stmt)
    data = DataFrame(my_cusor.fetchall())
    field_names = [i[0] for i in my_cusor.description]
    data.columns = field_names  
    my_cusor.close()    
    return data

def load_test_data(path_to_data):
    data = pd.read_csv(path_to_data, names=['date','time','0x0001','0x0002','0x0003','0x0004','0x0005','0x0006','0x0012','0x0013','0x0014','0x0015','0x0016','x_axis','y_axis','z_axis','orientation','reference_points','messurement_points'])
    data = data.iloc[1:,:]
    return data

def get_RSS(data):
    df =  pd.DataFrame(data)
    rss_df = df[['0x0001','0x0002','0x0003','0x0004','0x0005','0x0006','0x0012','0x0013','0x0014','0x0015','0x0016']]
    loc_df = df[['x_axis','y_axis','z_axis']]
    #loc_df = df[['reference_points','messurement_points']]
    return rss_df, loc_df     


def distance(a, b):
    return np.sqrt(np.sum(np.power(a-b, 2)))

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

    proba = 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-np.power(diff, 2)/(2.0*sigma**2))

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
    pos_pred = []
    cused = []
    x_pred = []
    y_pred = []
    acc_pred = []
    true_labels =[]
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
            y_pred.append(pos)
            y_temp = y_test.iloc[[i]].values
            Y_test = y_temp.ravel()
            acc_pred.append(get_absolute_loc_for_nav(y_pred[-1]))
            true_labels.append(Y_test)
            #Y_test = np.array(Y_test,dtype=list)
            error.append(distance(Y_test, acc_pred[-1]))
            fdetect += acc_pred[-1][2] == Y_test[2]
            # 2D error only if floor was detected correctly
            if acc_pred[-1][2] == Y_test[2]:
                error2D.append(distance(Y_test[0:2], np.array(acc_pred[-1])[0:2]))
    return (np.array(y_pred), np.array(error), np.array(error2D), fdetect, np.array(cused), true_labels, acc_pred)
################ END KMEANS PREDICTION MODEL SESSION  ################  
def get_true_labels(true_loc, acc_pred):
    my_cusor = mydb.cursor()
    true_label = []
    pred_label = []
    #select_stmt = "SELECT ref_point, mess_point FROM indoor_nav_using_affinity_prop.geo_location_cord where x_axis = %(x_axis)s and y_axis =  %(y_axis)s"
    select_stmt = "SELECT ref_point FROM indoor_nav_using_affinity_prop.geo_location_cord where x_axis = %(x_axis)s and y_axis =  %(y_axis)s"
    for i in range(len(true_loc)):
        my_cusor.execute(select_stmt,{ 'x_axis': true_loc[i][0] , 'y_axis': true_loc[i][1] })
        rows = my_cusor.fetchall()
        for r in rows:
            #true_temp = [r[0],r[1]]
            true_temp = r[0]
        true_label.append(true_temp)
        #true_label.append(true_temp)         
    print(true_label)

    #select_stmt = "SELECT ref_point, mess_point FROM indoor_nav_using_affinity_prop.geo_location_cord where x_axis = %(x_axis)s and y_axis =  %(y_axis)s"
    select_stmt = "SELECT ref_point FROM indoor_nav_using_affinity_prop.geo_location_cord where x_axis = %(x_axis)s and y_axis =  %(y_axis)s"
    for i in range(len(acc_pred)):
        my_cusor.execute(select_stmt,{ 'x_axis': acc_pred[i][0] , 'y_axis': acc_pred[i][1] })
        rows = my_cusor.fetchall()
        for r in rows:
            #pred_temp = [r[0],r[1]]
            pred_temp = r[0]
        pred_label.append(pred_temp) 
        #pred_label.append(pred_temp)
    print(pred_label)
    my_cusor.close() 
    return(true_label, pred_label)


def get_absolute_loc_for_nav(pred):
    my_cusor = mydb.cursor()
    loc = []
    select_stmt = "SELECT x_axis, y_axis, z_axis FROM indoor_nav_using_affinity_prop.geo_location_cord where x_axis = %(x_axis)s and y_axis =  %(y_axis)s"
    my_cusor.execute(select_stmt,{ 'x_axis': pred[0] , 'y_axis': pred[1] })
    rows = my_cusor.fetchall()
    for r in rows:
        loc = [r[0],r[1],0]  

    if(len(loc)<1):
        select_stmt = "select x_axis, y_axis, z_axis from indoor_nav_using_affinity_prop.geo_location_cord order by abs(x_axis - %(x_axis)s) + abs(y_axis - %(y_axis)s) limit 1"
        my_cusor.execute(select_stmt,{ 'x_axis': pred[0] , 'y_axis': pred[1] })
        rows = my_cusor.fetchall()
        for r in rows:
            loc = [r[0],r[1],0]           
    my_cusor.close()    
    return loc

train = load_data()
X_train, y_train  = get_RSS(train)


#Splitting Single data into 80% train and 20% test data 

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2,random_state=78)

#Loading External test data 
#test = load_test_data(path_to_test_data)
#X_test, y_test  = get_RSS(test)

X_test = X_test.astype(float)
y_test = y_test.astype(float)


print(X_test)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

x_test = X_test.values
print(type(X_test.iloc[1,1]))
#X_train = X_train.fillna(0)
X_ktrain = X_train.values
y_ktrain = y_train.values
#print(X_train.head())

N = X_ktrain.shape[0]
affinity = np.zeros((N,N))
for i in range(N):
    affinity[i,:] = bdist(X_ktrain, X_ktrain[i], 5, 1e-3, 1e-25)

#Time
tsum = 0
t = time.process_time()

cluster = AffinityPropagation(damping=0.5, affinity='precomputed')
labels = cluster.fit_predict(affinity)
C = np.unique(labels).size
clusters = X_ktrain[cluster.cluster_centers_indices_]



# estimate positions for test data
pred, error3D, error2D, fdetect, cused, true_labels, acc_pred = position_route(method, X_ktrain,
            y_ktrain, x_test, y_test, clusters, labels, N=5, eps=1e-3)

#Time
tsum += time.process_time() - t

#print("pos_pred")
#print(pred.shape)
#print("true_labels")
#print(len(true_labels))

#true_label, pred_label = get_true_labels(true_labels, acc_pred)
#print(confusion_matrix(true_label, pred_label))
#display_labels = ["RP-1", "RP-2", "RP-3", "RP-4", "RP-5", "RP-6", "RP-7", "RP-8", "RP-9", "RP-10", "RP-11"]
#cm = confusion_matrix(pred_label, true_label,labels= display_labels)
#print("confusion_matrix")
#print(cm)
#print("")





#print("precision_score")
#print(precision_score(true_label, pred_label,average='macro'))
#print(precision_score(true_label, pred_label,average='micro'))
#print(precision_score(true_label, pred_label,average='weighted'))
#print(precision_score(true_label, pred_label,average=None))

#print("recall")
#print(recall_score(true_label, pred_label,average='macro'))
#print(recall_score(true_label, pred_label,average='micro'))
#print(recall_score(true_label, pred_label,average='weighted'))

#print("accuracy")
#print(accuracy_score(true_label, pred_label))


#tn, fp, fn, tp = perf_measure(true_label, pred_label)
#print("tn",tn)
#print("fp",fp)
#print("fn",fn)
#print("tp",tp)

#print()
#print("Time")
#print(tsum)
#disp = ConfusionMatrixDisplay(confusion_matrix=cm,
#                              display_labels=display_labels)

#disp = disp.plot()
#plt.show()

# To Remove Outliers
def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]


true_label_list = []
for i in range(len(true_labels)):
    mylist = []
    mylist = [true_labels[i][0] , true_labels[i][1], 0.0]
    true_label_list.append(mylist)

pred_list = []
for i in range(len(pred)):
    mylist = []
    mylist = [pred[i][0] , pred[i][1], 0.0]
    pred_list.append(mylist)

pred_df = pd.DataFrame(pred_list)
true_label_df = pd.DataFrame(true_label_list)

predicted_x_loc = pred_df.iloc[:,1]
true_label_x_loc = true_label_df.iloc[:,1]

position_error =  abs(predicted_x_loc - true_label_x_loc) 

refined_position_error =reject_outliers(position_error)

#print(len(true_label_x_loc))
number_of_records = len(pred_list) 
#print(len(predicted_x_loc))
#print(len(true_label_x_loc))
print("position_error")
print(np.mean(refined_position_error))
df=pd.DataFrame({'x': range(1,number_of_records+1), 'y1': predicted_x_loc, 'y2': true_label_x_loc})
 
# multiple line plot
plt.plot( 'x', 'y1', data=df, marker='.', color='red', linewidth=2, linestyle='dashed', label="predicted_y_loc")
plt.plot( 'x', 'y2', data=df, marker='.', color='blue', linewidth=2, label="true_y_loc", alpha=0.7)
#plt.plot( 'x', 'y3', data=df, marker='', color='olive', linewidth=2, linestyle='dashed', label="true_x_loc")
plt.legend()
plt.show()
