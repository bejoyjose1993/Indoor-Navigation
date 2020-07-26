import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib import style
import os
import mysql.connector
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from playsound import playsound
from gtts import gTTS 
from collections import defaultdict

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

#Connect DB using mysql.connector
mydb = mysql.connector.connect(
    host = "localhost",
    user = "root",
    password = "root",
    database = "indoor_navigation"
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



def load_data(path_to_data):
    data = pd.read_csv(path_to_data,header=0)
    return (data)

def load_test_data(path_to_data):
    data = pd.read_csv(path_to_data, names=['date','time','type','id_0','id_1','id_2','sensor_data','temperature','humidity','raw_data'])
    return (data)    

def get_req_data(data):
    df =  pd.DataFrame(data)
    df = df[['0x0001','0x0002','0x0003','0x0004','0x0005','0x0006','0x0012','0x0013','0x0014','0x0015','0x0016','reference_points','Messurement_points']]
    return df

def get_req_test_data(data):
    df =  pd.DataFrame(data)
    df = df[['0x0001','0x0002','0x0003','0x0004','0x0005','0x0006','0x0012','0x0013','0x0014','0x0015','0x0016']]
    return df

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


def compute_RSSI(raw_data):
    hex_val = raw_data[-2:]
    int_val = int(hex_val, 16)
    rssi_val = int_val - 256
    return rssi_val
    


def initial_localization(initial_loc_path, audio_file):
    playsound(audio_file)
    data  = load_test_data(initial_loc_path)
    my_data = format_data(data)
    x_test = get_req_test_data(my_data)
    x_test = x_test.fillna(0)
    test_mean = x_test.mean(axis = 0).to_frame().transpose()
    pred = knn.predict(test_mean)
    return pred

def compute_init_block(pred):
    my_cusor = mydb.cursor()
    block = []
    select_stmt = "SELECT block FROM indoor_navigation.geo_location_cord where mess_point = %(mess_point)s and ref_point =  %(reff_point)s"
    my_cusor.execute(select_stmt,{ 'mess_point': pred[0][1] , 'reff_point': pred[0][0] })
    rows = my_cusor.fetchall()
    for r in rows:
        block.append(r[0])  
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
    dest_text = "Your destination is block" + dest
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
            select_stmt = "SELECT distance FROM indoor_navigation.edge_direction_graph where from_edge = %(current_node)s and to_edge =  %(next_node)s"
            my_cusor.execute(select_stmt,{ 'current_node': current_node , 'next_node': next_node })
            rows = my_cusor.fetchall()
            for r in rows:
                distance = distance + r[0]
            my_cusor.close()   

        current_node = next_node
    # Reverse path
    path = path[::-1]
    return path, distance
def get_block_ref_point(block_name):
    my_cusor = mydb.cursor()
    Y = []
    stmt = "SELECT ref_point FROM indoor_navigation.geo_location_cord where block = %(block_name)s"
    my_cusor.execute(stmt,{ 'block_name': block_name})
    rows = my_cusor.fetchall()
    for r in rows:
        Y =r[0]
    my_cusor.close()
    return Y

def get_nav_ref_points(cur_block, dest_block):
    my_cusor = mydb.cursor()
    dest_cusor = mydb.cursor()
    X = []
    Y = []
    block_area = cur_block[0] + " " + cur_block[1]
    #format_strings = ','.join(['%s'] * len(cur_block))
    #my_cusor.execute("SELECT ref_point FROM indoor_navigation.geo_location_cord where block  IN (%s) " % format_strings, tuple(cur_block))
    select_stmt = "SELECT ref_point FROM indoor_navigation.geo_location_cord where block = %(cur_block1)s or block =  %(cur_block2)s and block_area = %(block_area)s"
    my_cusor.execute(select_stmt,{ 'cur_block1': cur_block[0] , 'cur_block2': cur_block[1], 'block_area' : block_area})
    rows = my_cusor.fetchall()
    for r in rows:
        X =r[0]

    dest_stmt = "SELECT ref_point FROM indoor_navigation.geo_location_cord where block = %(dest_block)s"
    dest_cusor.execute(dest_stmt,{ 'dest_block': dest_block})
    rows = dest_cusor.fetchall()
    for r in rows:
        Y =r[0]
    my_cusor.close()
    dest_cusor.close()
    return (X,Y)

def get_direction(cur_rp, dest_block,path, distance_audio):
    print(path)
    next_rp = ""
    if(len(path) > 1):
        next_rp = path[path.index(cur_rp)+1]

    dest_cusor = mydb.cursor()
    dest_stmt = "SELECT ref_point FROM indoor_navigation.geo_location_cord where block = %(dest_block)s"
    dest_cusor.execute(dest_stmt,{ 'dest_block': dest_block})
    rows = dest_cusor.fetchall()
    for r in rows:
        dest_rp =r[0]
    dest_cusor.close()

    print(cur_rp,dest_rp)


    if(cur_rp == dest_rp):
        distance_text = 'Your Have Reached Your Destination'
        text_to_speach(distance_text,distance_audio)
        print(distance_text)
    else:
        my_cusor = mydb.cursor()
        select_stmt = "SELECT direction_comment FROM indoor_navigation.edge_direction_graph where from_edge = %(cur_rp)s and to_edge =  %(next_rp)s"
        my_cusor.execute(select_stmt,{ 'cur_rp': cur_rp , 'next_rp' : next_rp})
        rows = my_cusor.fetchall()
        for r in rows:
            distance_text =r[0]
        my_cusor.close()
        #distance_text = 'Your Have Reached Your Destination'
        text_to_speach(distance_text,distance_audio)
        print(distance_text)
        
data  = load_data(data_set_dir)
df = get_req_data(data)


X_train = df.drop(['reference_points','Messurement_points'],axis=1)
X_train = X_train.fillna(0)
#y_train = df[['reference_points','Messurement_points']]
y_train = df['Messurement_points']

#X_train, X_test, y_train, y_test = train_test_split(X_train,df[['reference_points','Messurement_points']],
#                test_size=0.01)
X_train, X_test, y_train, y_test = train_test_split(X_train,df['Messurement_points'],
                test_size=0.01)                

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)

KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=1, p=2,
           weights='uniform')
pred = knn.predict(X_test)
print(pred)

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

error_rate = []


# Will take some time

for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10,6))

plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')
plt.show()


knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=1')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


knn = KNeighborsClassifier(n_neighbors=6)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=23')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))

#K = 1,3,4,5 giving good results 1-best
