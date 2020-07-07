import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
path_to_database = 'D:\\Project\\Dissertation\\Master\\'

def load_data(path_to_data):
    # read test data
    FILE_NAME = path_to_data + '/master_data.csv'
    data = pd.read_csv(FILE_NAME,header=0)
    return data

def get_RSS(data):
    df =  pd.DataFrame(data)
    df = df[['0x0001','0x0002','0x0003','0x0004','0x0005','0x0006','0x0012','0x0013','0x0014','0x0015','0x0016']]
    return df
 
def get_geo_cord(data):
    df =  pd.DataFrame(data)
    df = df[['x_axis','y_axis','z_axis']]
    return df   

def plot_stats(rss_dB):
    rss_dB = rss_dB.to_numpy()
# Plot some statistics
    numFpPerAp = np.zeros((rss_dB.shape[1],1), dtype=np.int)
    for idx, vector in enumerate(rss_dB.T):
        a = np.logical_not(np.isnan(vector))
        numFpPerAp[idx] = sum(a)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(range(rss_dB.shape[1]), numFpPerAp[:,0] , 1)
    ax.set_title('Fingerprints per Access Point');
    ax.set_xlabel('access point ID');
    ax.set_ylabel('number of fingerprints');

    plt.draw()
    return

def plot_rp_grid(rp_m):
    rp_m = rp_m.to_numpy()
# Plot positions of fingerprints for all floors     
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(rp_m[:,0],rp_m[:,1], rp_m[:,2], c='b', marker='o')
    ax.set_title('Positions of fingerprints')
    ax.set_xlabel('easting (m)')
    ax.set_ylabel('northing (m)')
    ax.set_zlabel('height (m)')

    plt.draw()
    return

def plot_rp(rp_m):
    rp_m = rp_m.to_numpy()   
    fig = plt.figure()
    plt.scatter(rp_m[:,0],rp_m[:,1])
    plt.draw()
    return

def plot_fp_per_ap(rss_dB, rp_m, ap):
# Plot 3D scatter plot of RSS for single access point
    rp_m = rp_m.to_numpy()   
    rss_dB = rss_dB.to_numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(rp_m[:,0],rp_m[:,1], rp_m[:,2], c=rss_dB[:,ap], cmap='jet', marker='o')
    fig.colorbar(p, ax=ax)
    ax.set_title('Fingerprints of access point ' + repr(ap+1))
    ax.set_xlabel('easting (m)')
    ax.set_ylabel('northing (m)')
    ax.set_zlabel('height (m)')

    plt.draw()
    return


#Load Data
data = load_data(path_to_database)

#Get an process RSSI
rss_dB = get_RSS(data)
rss_dB[rss_dB==-79] = np.nan

#Get Geo Cordinates
rp_m = get_geo_cord(data)

# plot some statistics
plot_stats(rss_dB)
plot_rp(rp_m)
plot_rp_grid(rp_m)
selAp =  0x0001;
plot_fp_per_ap(rss_dB,rp_m,selAp)
plt.show()