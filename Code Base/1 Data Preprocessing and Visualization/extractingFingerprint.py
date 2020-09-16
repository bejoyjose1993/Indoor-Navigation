import numpy as np
from numpy import genfromtxt
import matplotlib as mpl
from matplotlib import pyplot as plt
import pylab
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from matplotlib.cbook import flatten
import os
import psycopg2
import errno
import mysql.connector
# enter path to directory where data is stored

#Connect DB using mysql.connector
mydb = mysql.connector.connect(
    host = "localhost",
    user = "root",
    password = "root",
    database = "indoor_navigation"
)

def load_data(path_to_data, f_name):

    # data
    
    FILE_NAME_RSS = path_to_data + f_name
    # read test data
    data = pd.read_csv(FILE_NAME_RSS, names=['date','time','type','id_0','id_1','id_2','sensor_data','temperature','humidity','raw_data'])
    #X = genfromtxt(FILE_NAME_RSS, delimiter=',')
    #X[X==100] = np.nan
    return (data)

def format_data(data,cord,rp,p_dir,file):
    #Empty data Frame
    initdata =[]
    columns = ['date','time','0x0001','0x0002', '0x0003','0x0004','0x0005', '0x0006','0x0012','0x0013', '0x0014','0x0015','0x0016']
    #columns = ['date','time']
    dataframe1 = pd.DataFrame(initdata,columns=columns)

    #Format File Orientation
    index = file.find('-')
    orientation = file[:index]

    df = data.iloc[1:]
    df = df.set_index("time", drop = False)
    time = df.time.unique()
    #print(time)
    #print(df.info())
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
        #print(type(list1))
        #print(list1)
        dataf.columns = list1
        dataf = dataf[1:]
        dataf.insert(loc=0, column='date', value=date)
        dataf.insert(loc=1, column='time', value=tym)
        dataf.insert(loc=0, column='x_axis', value=cord[0])
        dataf.insert(loc=1, column='y_axis', value=cord[1])
        dataf.insert(loc=2, column='z_axis', value= 0)
        dataf.insert(loc=3, column='orientation', value=orientation)
        dataf.insert(loc=4, column='reference_points', value=rp)
        dataf.insert(loc=5, column='Messurement_points', value=p_dir)
        #print(dataf)
        #print(dataf.info()) 
        dataframe1 = pd.concat([dataframe1,dataf])
    #print(dataframe1.head())
    dataframe1 = dataframe1.fillna(method='pad')
    dataframe1 = dataframe1.fillna(method='bfill')
    print(dataframe1)
    return dataframe1

def get_data():
#    data_set_dir = 'D:\\Project\\Dissertation\\Data Set\\Project Dataset\\'
    data_set_dir = 'D:\\Project\\Dissertation\\Test Data\\Dataset_Merge\\Project Dataset\\'
    root, Rp_dir, files = os.walk(data_set_dir).__next__()
    #temp = os.walk(data_set_dir)
    #os.path.dirname(data_set_dir)
    #Waliking Through directories
    #directory_list = list()
    #for root, dirs, files in os.walk(data_set_dir, topdown=True):
    #    for name in dirs:
    #        print(name)
            #directory_list.append(os.path.join(root, name))
    for rp in Rp_dir:
        rp_set_dir = data_set_dir + rp + "\\"
        rt, P_dir, files = os.walk(rp_set_dir).__next__()
        for p_dir in P_dir:
            p_set_dir = rp_set_dir + p_dir + "\\"
            rt, P_dir, files = os.walk(p_set_dir).__next__()
            cord = get_geo_cord(rp,p_dir)
            #print(cord)
            print("Cordinates for Reference Point: " +rp + "and messurement point: "+ p_dir+"is x_axis: "+str(cord[0])+"is y_axis: "+str(cord[1]))
            for f_name in files:
                rss_dB = load_data(p_set_dir,f_name)
                data =  pd.DataFrame(rss_dB)
                my_data = format_data(data,cord,rp,p_dir,f_name)
                #my_data = format_data(data,rp,P_dir,files)
                export_file_path = 'D:\\Project\\Dissertation\\Test Data\\Dataset_Merge\\Formated_Dataset\\' + rp +'\\' + p_dir + '\\' + f_name 
#                export_file_path = 'D:\\Project\\Dissertation\\Data Set\\Formated_Dataset\\' + rp +'\\' + p_dir + '\\' + f_name 
                if not os.path.exists(os.path.dirname(export_file_path)):
                    try:
                        os.makedirs(os.path.dirname(export_file_path))
                    except OSError as exc: # Guard against race condition
                        if exc.errno != errno.EEXIST:
                            raise
                my_data.to_csv (export_file_path, index = False, header=True)
                #print(my_data.info())
            #print(rp)
            #print(p_dir)
            #print(p_set_dir)
            #print(files)
            
        

    #print directory_list
    
    #path_to_database = 'D:\\Project\\Formated_Dataset\\RP-1\\P-1\\'       
    #rss_dB = load_data(path_to_database)
    #data =  pd.DataFrame(rss_dB)
    #print(data.head(5))
    #my_data = format_data(data)
    #export_file_path = 'D:\\Project\\My_Name.csv'
    #my_data.to_csv (export_file_path, index = False, header=True)

def get_geo_cord(rp, p_dir):
    cord = list()
    #con = psycopg2.connect(
    #host= "localhost",
    #database="indoor_nav_db",
    #user="postgres",
    #password="bejoy1993",
    #port= 5432)

    #cursor
    #cur = con.cursor()
    my_cusor = mydb.cursor()
    my_cusor.execute("select x_axis, y_axis from geo_location_cord where ref_point = '"+ rp + "' and mess_point = '"+ p_dir +"'")
    rows = my_cusor.fetchall()
    #print(rows)
    for r in rows:
        cord = [r[0],r[1]]

    #Close
    my_cusor.close()    
    #con.close()
    return cord


def compute_RSSI(raw_data):
    hex_val = raw_data[-2:]
    int_val = int(hex_val, 16)
    rssi_val = int_val - 256
    return rssi_val
    

get_data()
mydb.close()
#path_to_database = 'D:\\Project\\Project Dataset\\RP-1\\P-1\\'       
#rss_dB = load_data(path_to_database, "West 0.csv")
#data =  pd.DataFrame(rss_dB)
#print(data.head(5))
#my_data = format_data(data)
#export_file_path = 'D:\\Project\\My_Name.csv'
#my_data.to_csv (export_file_path, index = False, header=True)
#compute_RSSI("0201041AFF4C0002150005000100001000800000805F9B01310001B460BF")
