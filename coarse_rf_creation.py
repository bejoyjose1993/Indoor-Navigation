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
    orientations = []
    select_stmt = "SELECT cluster_labels, orientation FROM indoor_navigation.master_with_kmean_cluster ORDER BY cluster_labels ASC;"
    my_cusor.execute(select_stmt)
    rows = my_cusor.fetchall()
    for r in rows:
        cluster_rp.append(r[0])  
        orientations.append(r[1])
    
    cluster_labels = list(dict.fromkeys(cluster_rp))
    ori_rp = list(dict.fromkeys(orientations))
    select_cluster = "SELECT * FROM indoor_navigation.master_with_kmean_cluster where cluster_labels = %(cluster_labels)s and orientation = %(orientation)s"
    engine = create_engine('mysql+mysqlconnector://root:root@localhost/indoor_navigation')
    for cl in cluster_labels:
        for orp in ori_rp:
            my_cusor.execute(select_cluster,{ 'cluster_labels': cl, 'orientation': orp})
            df = DataFrame(my_cusor.fetchall())
            #df.columns = field_names
            field_names = [i[0] for i in my_cusor.description]
            df.columns = field_names  
            x_test, reference_point = get_req_data(df)
            test_mean = x_test.mean(axis = 0).to_frame().transpose()
            test_mean['cluster_labels'] = cl
            test_mean['orientation'] = orp
            test_mean['reference_point'] = reference_point
            #test_mean.to_sql('clustered_ref_point',con=engine,index=False,if_exists='append')
    my_cusor.close()   


def get_req_data(data):
    ref_points = []
    df =  pd.DataFrame(data)
    rssi_df = df[['0x0001','0x0002','0x0003','0x0004','0x0005','0x0006','0x0012','0x0013','0x0014','0x0015','0x0016']]
    rf_poits = df['reference_points']
    rf_poits = pd.DataFrame(rf_poits)
    ref_points = rf_poits.iloc[:,0].values.tolist()

    reference_point = list(dict.fromkeys(ref_points))
    return rssi_df, reference_point
  
create_coase_rf()
