#import mysql.connector
import pandas as pd
from sqlalchemy import create_engine

#Write Csv to db using sqlalchemy
df = pd.read_csv('D:\\Project\\Dissertation\\Master\\edge_direction_graph.csv')
df.columns = [c.lower() for c in df.columns]


engine = create_engine('mysql+mysqlconnector://root:root@localhost/indoor_navigation')
df.to_sql('edge_direction_graph',con=engine)

#import pandas as pd
#from sqlalchemy import create_engine

#df = pd.read_csv('D:\\Project\\Dissertation\\Master\\Loc_cordinates.csv')

#df.columns = [c.lower() for c in df.columns] #postgres doesn't like capitals or spaces



#engine = create_engine('postgresql://postgres:bejoy1993@localhost:5432/indoor_nav_db')
#df.to_sql("geo_loc_cord", engine)


