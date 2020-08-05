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
    database = "indoor_nav_using_affinity_prop"
)

def create_coase_rf():
    #field_names = ['0x0001','0x0002','0x0003','0x0004','0x0005','0x0006','0x0012','0x0013','0x0014','0x0015','0x0016','x_axis','y_axis','z_axis','orientation','messurement_points','labels','refference_labels']
    my_cusor = mydb.cursor()
    ref_points = []
    mess_points = []
    orientations = []
    select_stmt = "SELECT reference_points, messurement_points, orientation FROM indoor_nav_using_affinity_prop.master_data ORDER BY reference_points ASC;"
    my_cusor.execute(select_stmt)
    rows = my_cusor.fetchall()
    for r in rows:
        ref_points.append(r[0])  
        mess_points.append(r[1])
        orientations.append(r[2])
    
    refference_labels = list(dict.fromkeys(ref_points))
    messssurement_points = list(dict.fromkeys(mess_points))
    ori_rp = list(dict.fromkeys(orientations))
    #print(refference_labels)
    #print(messssurement_points)
    #print(ori_rp)
    select_cluster = "SELECT * FROM indoor_nav_using_affinity_prop.master_data where reference_points = %(reference_points)s and messurement_points = %(messurement_points)s and orientation = %(orientation)s"
    engine = create_engine('mysql+mysqlconnector://root:root@localhost/indoor_nav_using_affinity_prop')
    for rf in refference_labels:
        for mp in messssurement_points:
            for orp in ori_rp:
                my_cusor.execute(select_cluster,{'reference_points': rf, 'messurement_points': mp, 'orientation': orp})
                df = DataFrame(my_cusor.fetchall())
                field_names = [i[0] for i in my_cusor.description]
                df.columns = field_names  
                x_test, x_axis, y_axis, z_axis = get_req_data(df)
                test_mean = x_test.mean(axis = 0).to_frame().transpose()
                test_mean['x_axis'] = x_axis
                test_mean['y_axis'] = y_axis
                test_mean['z_axis'] = 0
                test_mean['reference_points'] = rf
                test_mean['messurement_points'] = mp
                test_mean['orientation'] = orp
                test_mean.to_sql('affinity_prop_master',con=engine,index=False,if_exists='append')
                #print(test_mean)
    my_cusor.close()   


def get_req_data(data):
    ref_points = []
    df =  pd.DataFrame(data)
    rssi_df = df[['0x0001','0x0002','0x0003','0x0004','0x0005','0x0006','0x0012','0x0013','0x0014','0x0015','0x0016']]
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

    #y_axis Data
    z = df['z_axis']
    z = pd.DataFrame(z)
    z_axis_loc = z.iloc[:,0].values.tolist()
    z_axis = list(dict.fromkeys(z_axis_loc))
    
    return rssi_df, x_axis, y_axis, z_axis
  
create_coase_rf()
