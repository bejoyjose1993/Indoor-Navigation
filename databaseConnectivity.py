import psycopg2

con = psycopg2.connect(
    host= "localhost",
    database="indoor_nav_db",
    user="postgres",
    password="bejoy1993",
    port= 5432)

#cursor
cur = con.cursor()
cur.execute("select ref_point, mess_point, x_axis, y_axis from loc_cordinates")
rows = cur.fetchall()

for r in rows:
    print(f"ref_point {r[0]} mess_point {r[1]} x_axis {r[2]} y_axis {r[3]}")

#Close
cur.close()    
con.close()