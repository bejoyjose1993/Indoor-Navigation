#Creating Coarse Figerprints
import mysql.connector
from pandas import DataFrame
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

mydb = mysql.connector.connect(
    host = "localhost",
    user = "root",
    password = "root",
    database = "indoor_navigation"
)

def create_coase_rf():
    #field_names = ['0x0001','0x0002','0x0003','0x0004','0x0005','0x0006','0x0012','0x0013','0x0014','0x0015','0x0016','x_axis','y_axis','z_axis','orientation','messurement_points','labels','cluster_labels']
    my_cusor = mydb.cursor()
    cluster_rp = []
    mess_points = []
    orientations = []
    select_stmt = "SELECT cluster_labels, messurement_points, orientation FROM indoor_navigation.master_with_kmean_cluster ORDER BY cluster_labels ASC;"
    my_cusor.execute(select_stmt)
    rows = my_cusor.fetchall()
    for r in rows:
        cluster_rp.append(r[0])  
        mess_points.append(r[1])
        orientations.append(r[2])
    
    cluster_labels = list(dict.fromkeys(cluster_rp))
    messssurement_points = list(dict.fromkeys(mess_points))
    ori_rp = list(dict.fromkeys(orientations))
    select_cluster = "SELECT * FROM indoor_navigation.master_with_kmean_cluster where cluster_labels = %(cluster_labels)s and messurement_points = %(messurement_points)s and orientation = %(orientation)s"
    engine = create_engine('mysql+mysqlconnector://root:root@localhost/indoor_navigation')
    for cl in cluster_labels:
        for mp in messssurement_points:
            for orp in ori_rp:
                my_cusor.execute(select_cluster,{ 'cluster_labels': cl, 'messurement_points': mp, 'orientation': orp})
                df = DataFrame(my_cusor.fetchall())
                field_names = [i[0] for i in my_cusor.description]
                df.columns = field_names  
                x_test, reference_point, x_axis, y_axis, z_axis = get_req_data(df)
                test_mean = x_test.mean(axis = 0).to_frame().transpose()
                test_mean['x_axis'] = x_axis
                test_mean['y_axis'] = y_axis
                test_mean['z_axis'] = 0
                test_mean['reference_points'] = reference_point
                test_mean['messurement_points'] = mp
                test_mean['orientation'] = orp
                test_mean['cluster_labels'] = cl                
                test_mean.to_sql('clustered_ref_point',con=engine,index=False,if_exists='append')
                #print(test_mean)
    my_cusor.close()   


def get_req_data(data):
    ref_points = []
    df =  pd.DataFrame(data)
    rssi_df = df[['0x0001','0x0002','0x0003','0x0004','0x0005','0x0006','0x0012','0x0013','0x0014','0x0015','0x0016']]
    
    #rf_poits Data
    rf_poits = df['reference_points']
    rf_poits = pd.DataFrame(rf_poits)
    ref_points = rf_poits.iloc[:,0].values.tolist()
    reference_point = list(dict.fromkeys(ref_points))

    #x_axis Data
    x = df['x_axis']
    x = pd.DataFrame(x)
    x_axis_loc = x.iloc[:,0].values.tolist()
    x_axis = list(dict.fromkeys(x_axis_loc))

    #y_axis Data
    y = df['y_axis']
    y = pd.DataFrame(y)
    y_axis_loc = y.iloc[:,0].values.tolist()
    y_axis = list(dict.fromkeys(y_axis_loc))

    #Z_axis Data
    z = df['z_axis']
    z = pd.DataFrame(z)
    z_axis_loc = z.iloc[:,0].values.tolist()
    z_axis = list(dict.fromkeys(z_axis_loc))
    return rssi_df, reference_point, x_axis, y_axis, z_axis
  
create_coase_rf()
