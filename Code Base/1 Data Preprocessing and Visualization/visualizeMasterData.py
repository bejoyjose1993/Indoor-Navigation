import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
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
    ax.bar(range(rss_dB.shape[1]), numFpPerAp[:,0] ,  width=0.8, color = 'PURPLE')
    ax.set_title('Fingerprints per Access Point');
    ax.set_xlabel('access point ID');
    ax.set_ylabel('number of fingerprints');
    ax.set_xticklabels(('0','AP-1','AP-3','AP-5','AP-7','AP-9','AP-11'))
    

    plt.draw()
    return

def plot_rp_grid(rp_m):
    
    rp_m = rp_m.to_numpy()
# Plot positions of fingerprints for all floors     
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(rp_m[:,0],rp_m[:,1], rp_m[:,2], c='FORESTGREEN', marker='o')
    
    ax.set_title('3D-Positions of fingerprints')
    ax.set_xlabel('East - Direction (m)')
    ax.set_ylabel('South - Direction (m)')
    ax.set_zlabel('height (m)')
    plt.gca().invert_xaxis()
    #plt.gca().invert_yaxis()

    plt.draw()
    return

def plot_rp(rp_m):
    img = mpimg.imread("D:\\Project\\Dissertation\\floor_plan_samp.png")
    rp_m = rp_m.to_numpy()   
    fig = plt.figure()
    plt.imshow(img, extent=[36.15,0,-4.90,49])
    plt.scatter(rp_m[:,0],rp_m[:,1], color='FORESTGREEN')
    ax=plt.gca()  
    ax.yaxis.tick_right() 
    plt.title('Recorded Ground Truth points')
    plt.xlabel('East - Direction (m)')
    plt.ylabel('South - Direction (m)')
    #ax.set_title('Fingerprints of access point')
    #ax.set_xlabel
    #ax.set_ylabel('southing (m)')
    
    #plt.gca().invert_xaxis()
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
    ax.set_xlabel('East - Direction (m)')
    ax.set_ylabel('South - Direction (m)')
    ax.set_zlabel('height (m)')
    plt.gca().invert_xaxis()
    plt.draw()
    return

def plot_rss(rp_m, rss_dB, ap, pltype='plot3'):
# Plot surface, contour, etc. of RSS values for an access point.
# Plot signal strength values of given AP over given metric cartesian
# coordinate system.
    
    fig = plt.figure()
    X, Y = np.meshgrid(rp_m[0,:,0], rp_m[0,:,1],indexing='ij')
    ZI = griddata((rp_m[0,:,0], rp_m[0,:,1]), rss_dB[:,ap], (X, Y))
    if pltype == 'plot3':
        ax = fig.add_subplot(111, projection='3d')
        p = ax.scatter(rp_m[0,:,0],rp_m[0,:,1], rss_dB[:,ap], c='k', marker='o')
    elif pltype == 'contour':
        ax = fig.add_subplot(111)
        c = ax.contour(X, Y, ZI, cmap='jet')
        fig.colorbar(c, ax=ax)
    elif pltype == 'contour3':
        ax = fig.add_subplot(111, projection='3d')
        c = ax.contour(X, Y, ZI, cmap='jet')
        fig.colorbar(c, ax=ax)
        ax.set_zlabel('RSS (dB)')
#    elif pltype == 'contourf': # requires too much mem
#        ax = fig.add_subplot(111, projection='3d')
#        c = ax.contourf(X, Y, ZI, cmap='jet')
#        fig.colorbar(c, ax=ax)
#        ax.set_zlabel('RSS (dB)')
#    elif pltype == 'surf':
#        ax = fig.add_subplot(111, projection='3d')
#        c = ax.plot_surface(X, Y, ZI, cmap='jet')
#        fig.colorbar(c, ax=ax)
#        ax.set_zlabel('RSS (dB)')
    else:
        print('Unknown plot type')

    ax.set_xlabel('easting (m)')
    ax.set_ylabel('northing (m)')
    ax.set_title('RSS of access point ' + repr(ap+1) + ' on the selected floor')

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
#plot_stats(rss_dB)
#plot_rp(rp_m)
#plot_rp_grid(rp_m)
selAp =  4-1;
#plot_fp_per_ap(rss_dB,rp_m,selAp)


rp_m1 = rp_m.to_numpy()   
rss_dB1 = rss_dB.to_numpy()
hgt = np.unique(rp_m1[:,2])
#select the floor number for visualization; in this example it is 2
floor = 0; # choose between 0:4
idxRpFlX = np.where(rp_m1[:,2] == hgt[floor])
# num. of valid FP must be high
plot_rss(rp_m1[idxRpFlX, 0:2], rss_dB1[idxRpFlX], selAp, 'plot3')

plt.show()