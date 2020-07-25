import mysql.connector
import pandas as pd
from sqlalchemy import create_engine

#Write Csv to db using sqlalchemy
df = pd.read_csv('D:\\Project\\Dissertation\\Master\\labels.csv')
df.columns = [c.lower() for c in df.columns]
temp = df["labels"] + 1

df['cluster_labels'] = 'RP-' + temp.astype(str)

#temp_group = "RP-" +    str(temp)
#engine = create_engine('mysql+mysqlconnector://root:root@localhost/indoor_navigation')
#df.to_sql('rssi_kmean_cluster',con=engine,index=False,if_exists='append')


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

