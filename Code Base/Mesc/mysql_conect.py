import mysql.connector
import pandas as pd
from sqlalchemy import create_engine

#Write Csv to db using sqlalchemy
df = pd.read_csv('D:\\Project\\Dissertation\\Test Data\\Dataset_Merge\Merge Master\\total_train_master.csv')
df.columns = [c.lower() for c in df.columns]
#df = df.transpose()
#temp = df["labels"] + 1
#print(df.shape)

#df['cluster_labels'] = 'RP-' + temp.astype(str)

#temp_group = "RP-" +    str(temp)
engine = create_engine('mysql+mysqlconnector://root:root@localhost/indoor_nav_using_affinity_prop')
df.to_sql('evaluation_master',con=engine,index=False,if_exists='replace')


#Connect DB using mysql.connector
#mydb = mysql.connector.connect(
#    host = "localhost",
#    user = "root",
#    password = "root",
#    database = "indoor_navigation"
#)

#my_cusor = mydb.cursor()
# Creating DB
#my_cusor.execute("CREATE DATABASE indoor_navigation")

