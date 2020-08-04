import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib import style
import matplotlib.image as mpimg
import os
import mysql.connector
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from playsound import playsound
from gtts import gTTS 
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D #for 3D plot
from sklearn.cluster import AffinityPropagation      #use kmeans
from sklearn.metrics import silhouette_samples, silhouette_score #to cal the silhouette_score
from numpy import *
from pandas import DataFrame
import statistics
import time 
from statistics import mode, mean 

data_set_dir ="D:\\Project\\Dissertation\\Master\\master_data.csv"
initial_test_dir = 'D:\\Project\\Dissertation\\raw_example.csv'
#initial_test_dir = "D:\\Project Softwares\\Cypress BLE-Beacon\\EXE\\2_Test_Dataset\\example.csv"
initial_loc_audio = 'D:\\Project\\Dissertation\\Audio\\init_position.mp3'

test_raw_dir ="D:\\Project\\Dissertation\\raw_example.csv"
#test_raw_dir ="D:\\Project Softwares\\Cypress BLE-Beacon\\EXE\\2_Test_Dataset\\example.csv"
init_block_audio ="D:\\Project\\Dissertation\\Audio\\init_block.mp3"
distance_audio ="D:\\Project\\Dissertation\\Audio\\distance.mp3"
input_dest ="D:\\Project\\Dissertation\\Audio\\dest_block.mp3" 
select_dest = "D:\\Project\\Dissertation\\Audio\\select_dest.mp3"

#Kmeans Pre-requsits
method = 'ap'
#path_to_database = 'D:\\Project\\Dissertation\\Master\\'
test_raw_dir ="D:\\Project\\Dissertation\\raw_example.csv"

#Connect DB using mysql.connector
mydb = mysql.connector.connect(
    host = "localhost",
    user = "root",
    password = "root",
    database = "indoor_nav_using_affinity_prop"
)

class Graph():
    def __init__(self):
        """
        self.edges is a dict of all possible next nodes
        e.g. {'X': ['A', 'B', 'C', 'E'], ...}
        self.weights has all the weights between two nodes,
        with the two nodes as a tuple as the key
        e.g. {('X', 'A'): 7, ('X', 'B'): 2, ...}
        """
        self.edges = defaultdict(list)
        self.weights = {}
    
    def add_edge(self, from_node, to_node, weight):
        # Note: assumes edges are bi-directional
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)
        self.weights[(from_node, to_node)] = weight
        self.weights[(to_node, from_node)] = weight

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

# START KMEANS PREDICTION MODEL SESSION    
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

    for pred in pos_pred :
        x_pred.append(pred[0])
        y_pred.append(pred[1])

    #x = Counter(x_pred)
    x_max_count = mode(x_pred)
    y_max_count = mode(y_pred)
    pred = [round(x_max_count,1),round(y_max_count,1)]
    print(pred)
    return (np.array(pos_pred), np.array(error), np.array(error2D), fdetect, np.array(cused), pred)
################ END AFFINITY PREDICTION MODEL SESSION  ################  



def initial_localization(initial_loc_path, audio_file, X_train, y_train, method, clusters, labels):
    playsound(audio_file)

    #Get Test Data 
    data  = load_test_data(test_raw_dir)
    my_data = format_data(data)
    x_test = get_req_test_data(my_data)
    #test_matrix = x_test.values
    test_mean = x_test.mean(axis = 0).to_frame().transpose()
    test_matrix_new = test_mean.values
    test_matrix_new = np.vstack((test_matrix_new, test_mean.values))
    test_matrix = np.vstack((test_matrix_new, test_mean.values))
    
    #Predict Code
    tsum = 0
    t = time.process_time()
    # estimate positions for test data
    y, error3D, error2D, fdetect, cused, pred = position_route(method, X_ktrain,
            y_ktrain, test_matrix, clusters, labels, N=5, eps=1e-3)
    return pred

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

def get_refpoint_from_pos(pred):
    my_cusor = mydb.cursor()
    block = []
    select_stmt = "SELECT ref_point FROM indoor_nav_using_kmeans.geo_location_cord where x_axis = %(x_axis)s and y_axis =  %(y_axis)s"
    my_cusor.execute(select_stmt,{ 'x_axis': pred[0] , 'y_axis': pred[1] })
    rows = my_cusor.fetchall()
    for r in rows:
        block.append(r[0])

    if(len(block)<1):
        select_stmt = "select ref_point from indoor_nav_using_affinity_prop.geo_location_cord order by abs(x_axis - %(x_axis)s) + abs(y_axis - %(y_axis)s) limit 1"
        my_cusor.execute(select_stmt,{ 'x_axis': pred[0] , 'y_axis': pred[1] })
        rows = my_cusor.fetchall()
        for r in rows:
            block = r[0] 

    my_cusor.close()    
    return block    

def text_to_speach(mytext, audio_file):
    language = 'en'
    myobj = gTTS(text=mytext, lang=language, slow=False) 
    os.remove(audio_file)
    myobj.save(audio_file) 
    playsound(audio_file)
    

def input_destination():
    mytext = "Enter your destination block"
    text_to_speach(mytext,select_dest)
    dest = input("Enter your destination block:")
    dest_text = "Your destination is block " + dest
    text_to_speach(dest_text,input_dest)
    print(dest_text)
    return dest

def get_graph_edge():
    edges = [
        ('RP-1', 'RP-2', 7.20),
        ('RP-2', 'RP-3', 7.20),
        ('RP-3', 'RP-4', 7.20),
        ('RP-4', 'RP-5', 7.20),
        ('RP-5', 'RP-6', 7.20),
        ('RP-5', 'RP-7', 7.20),
        ('RP-7', 'RP-8', 7.20),
        ('RP-8', 'RP-9', 7.20),
        ('RP-9', 'RP-10', 7.20),
        ('RP-10', 'RP-11', 7.20),
    ]
    return edges

def dijsktra(graph, initial, end):
    # shortest paths is a dict of nodes
    # whose value is a tuple of (previous node, weight)
    shortest_paths = {initial: (None, 0)}
    current_node = initial
    visited = set()
    
    while current_node != end:
        visited.add(current_node)
        destinations = graph.edges[current_node]
        weight_to_current_node = shortest_paths[current_node][1]

        for next_node in destinations:
            weight = graph.weights[(current_node, next_node)] + weight_to_current_node
            if next_node not in shortest_paths:
                shortest_paths[next_node] = (current_node, weight)
            else:
                current_shortest_weight = shortest_paths[next_node][1]
                if current_shortest_weight > weight:
                    shortest_paths[next_node] = (current_node, weight)
        
        next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}
        if not next_destinations:
            return "Route Not Possible"
        # next node is the destination with the lowest weight
        current_node = min(next_destinations, key=lambda k: next_destinations[k][1])
    
    # Work back through destinations in shortest path
    path = []
    distance = 0
    while current_node is not None:
        path.append(current_node)
        next_node = shortest_paths[current_node][0]

        #Calculate Distance
        if (next_node != None):
            my_cusor = mydb.cursor()
            select_stmt = "SELECT distance FROM indoor_nav_using_kmeans.edge_direction_graph where from_edge = %(current_node)s and to_edge =  %(next_node)s"
            my_cusor.execute(select_stmt,{ 'current_node': current_node , 'next_node': next_node })
            rows = my_cusor.fetchall()
            for r in rows:
                distance = distance + r[0]
            my_cusor.close()   

        current_node = next_node
    # Reverse path
    path = path[::-1]
    return path, distance

def get_nav_ref_points(cur_block, dest_block):
    my_cusor = mydb.cursor()
    dest_cusor = mydb.cursor()
    print("cur_block")
    print(cur_block)
    print("dest_block")
    print(dest_block)

    X = []
    Y = []
    loc = []
    if(len(cur_block) > 1):
        block_area = cur_block[0] + " " + cur_block[1]
        select_stmt = "SELECT ref_point FROM indoor_navigation.geo_location_cord where block = %(cur_block1)s or block =  %(cur_block2)s and block_area = %(block_area)s"
        my_cusor.execute(select_stmt,{ 'cur_block1': cur_block[0] , 'cur_block2': cur_block[1], 'block_area' : block_area})
    else:
        block_area = cur_block[0]
        select_stmt = "SELECT ref_point FROM indoor_navigation.geo_location_cord where block = %(cur_block1)s and block_area = %(block_area)s"
        my_cusor.execute(select_stmt,{ 'cur_block1': cur_block[0], 'block_area' : block_area})
    rows = my_cusor.fetchall()
    for r in rows:
        X =r[0]
    dest_stmt = "SELECT ref_point,x_axis, y_axis FROM indoor_navigation.geo_location_cord where block = %(dest_block)s"
    dest_cusor.execute(dest_stmt,{ 'dest_block': dest_block})
    rows = dest_cusor.fetchall()
    for r in rows:
        Y.append(r[0])
        #loc.append([r[1],r[2]])
    reff_point = list(dict.fromkeys(Y))
    index = 0
    if(len(reff_point)>1):
        ref_dist = []
        for r in reff_point:
            my_path, distance = dijsktra(graph, X, r)
            ref_dist.append(distance)
        index = ref_dist.index(min(ref_dist))
    #print(reff_point[index])

    dest_stmt = "SELECT ref_point,x_axis, y_axis FROM indoor_navigation.geo_location_cord where ref_point = %(ref_point)s"
    dest_cusor.execute(dest_stmt,{ 'ref_point':     reff_point[index]})
    loc_rows = dest_cusor.fetchall()
    for r in loc_rows:
        #Y.append(r[0])
        loc = [r[1],r[2]]
    #print(loc)
    my_cusor.close()
    dest_cusor.close()
    return (X,reff_point[index],loc)

def get_direction(cur_rp, dest_rp,path, distance_audio):

    next_rp = ""
    direction = "Destination"
    if(len(path) > 1):
        next_rp = path[path.index(cur_rp)+1]

    #dest_cusor = mydb.cursor()
    #dest_stmt = "SELECT ref_point FROM indoor_navigation.geo_location_cord where block = %(dest_block)s"
    #dest_cusor.execute(dest_stmt,{ 'dest_block': dest_block})
    #rows = dest_cusor.fetchall()
    #for r in rows:
    #    dest_rp =r[0]
    #dest_cusor.close()

    print(cur_rp,dest_rp)


    if(cur_rp == dest_rp):
        distance_text = 'Your Have Reached Your Destination'
        text_to_speach(distance_text,distance_audio)
        print(distance_text)
    else:
        my_cusor = mydb.cursor()
        select_stmt = "SELECT direction_comment,direction FROM indoor_nav_using_kmeans.edge_direction_graph where from_edge = %(cur_rp)s and to_edge =  %(next_rp)s"
        my_cusor.execute(select_stmt,{ 'cur_rp': cur_rp , 'next_rp' : next_rp})
        rows = my_cusor.fetchall()
        for r in rows:
            distance_text =r[0]
            direction = r[1]
        my_cusor.close()
        #distance_text = 'Your Have Reached Your Destination'
        text_to_speach(distance_text,distance_audio)
        print(distance_text)
    return  direction       

def affinity_train(X_train):
    #X_train = X_train.fillna(0)
    X_ktrain = X_train.values
    #print(X_train.head())

    N = X_ktrain.shape[0]
    affinity = np.zeros((N,N))
    for i in range(N):
        affinity[i,:] = bdist(X_ktrain, X_ktrain[i], 5, 1e-3, 1e-25)

    cluster = AffinityPropagation(damping=0.5, affinity='precomputed')
    labels = cluster.fit_predict(affinity)
    C = np.unique(labels).size
    clusters = X_ktrain[cluster.cluster_centers_indices_]
    return clusters, labels, C

#Get Train Data 
train = load_data()
X_train, y_train  = get_RSS(train)
X_ktrain = X_train.values
y_ktrain = y_train.values
clusters, labels, C = affinity_train(X_train)


init_pred = initial_localization(initial_test_dir, initial_loc_audio, X_ktrain, y_ktrain, method, clusters, labels)
init_block = get_absolute_loc_for_nav(init_pred)
print(init_block)
if(len(init_block) > 1):
    block_text = 'User is located near block! ' + init_block[0] +' and ' + init_block[1]
else:
    block_text = 'User is located near block! ' + init_block[0]


text_to_speach(block_text,init_block_audio)
#print(block_text)
dest_block = input_destination()


graph = Graph()
edges = get_graph_edge()
for edge in edges:
    graph.add_edge(*edge)
X, dest_rp , loc= get_nav_ref_points(init_block, dest_block)
path, distance = dijsktra(graph, X, dest_rp)
distance_text = 'Your Location is approximately ' + str(round(distance)) +' meeters away'
text_to_speach(distance_text,distance_audio)
#print(distance_text)
 
fig = plt.figure()
#ax1 = fig.add_subplot(1,1,1)

img = mpimg.imread("D:\\Project\\Dissertation\\floor_plan_samp.png")
#print(img)

ax=plt.gca()  
ax.yaxis.tick_right()        
plt.grid(False)  
xlabel = "DEST"
plt.scatter(loc[0],loc[1], marker="o",c="RED")

plt.xlabel("East - Direction", fontdict=None, labelpad=None)
plt.ylabel("South - Direction", fontdict=None, labelpad=None)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

def animate(i):

    from random import random
#   data  = load_data(test_set_dir)
    #Get Test Data 
    data  = load_test_data(test_raw_dir)
    my_data = format_data(data)
    x_test = get_req_test_data(my_data)
    test_mean = x_test.mean(axis = 0).to_frame().transpose()
    test_matrix_new = test_mean.values
    test_matrix_new = np.vstack((test_matrix_new, test_mean.values))
    test_matrix = np.vstack((test_matrix_new, test_mean.values))
    print(test_matrix)

    #Predict Code
    y, error3D, error2D, fdetect, cused, pred = position_route(method, X_ktrain,
            y_ktrain, test_matrix, clusters, labels, N=5, eps=1e-3)
    ref_pred = get_refpoint_from_pos(pred)
    print(ref_pred)
    X = ref_pred

    
    new_path, distance = dijsktra(graph, X, dest_rp)
    direction = get_direction(X, dest_rp,new_path,distance_audio)
    distance_text = 'Your Location is approximately ' + str(round(distance)) +' meeters away'
    text_to_speach(distance_text,distance_audio)
    print(distance_text)

    #print(ref_pred)
    #my_cusor = mydb.cursor()
    #select_stmt = "SELECT x_axis, y_axis FROM indoor_nav_using_kmeans.geo_location_cord where mess_point = %(mess_point)s and ref_point =  %(reff_point)s"
    #my_cusor.execute(select_stmt,{ 'mess_point': pred[0][1] , 'reff_point': pred[0][0] })
    #rows = my_cusor.fetchall()
    #for r in rows:
    #    xs = r[0]
    #    ys = r[1]
    #plt.clear()
    xs = pred[0]
    ys = pred[1]
    r = random()
    b = random()
    g = random()
    color = (r, g, b)
    if(direction == "Destination"):
        plt.scatter(xs, ys, c=color)
    elif(direction == "South"):
        plt.scatter(xs, ys,marker="^", c=color)
    elif(direction == "North"):  
        plt.scatter(xs, ys,marker="v", c=color)  
ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.imshow(img, extent=[36.15,0,-4.90,51.32])
plt.show()

mydb.close()