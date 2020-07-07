import pandas as pd
from sqlalchemy import create_engine

df = pd.read_csv('D:\\Project\\Dissertation\\Master\\Loc_cordinates.csv')

df.columns = [c.lower() for c in df.columns] #postgres doesn't like capitals or spaces


engine = create_engine('postgresql://postgres:bejoy1993@localhost:5432/indoor_nav_db')
df.to_sql("geo_loc_cord", engine)
