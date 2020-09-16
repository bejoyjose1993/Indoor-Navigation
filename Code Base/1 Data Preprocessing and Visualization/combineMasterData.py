import pandas as pd
import os
import errno
# enter path to directory where data is stored

def load_data(path_to_data, f_name):

    # data
    
    FILE_NAME_RSS = path_to_data + f_name
    # read test data
    data = pd.read_csv(FILE_NAME_RSS,header=0)
    #data = pd.read_csv(FILE_NAME_RSS, names=['date','time','0x0001','0x0002','0x0003','0x0004','0x0005','0x0006','0x0012','0x0013','0x0014','0x0015','0x0016','x_axis','y_axis'])
    #X = genfromtxt(FILE_NAME_RSS, delimiter=',')
    #X[X==100] = np.nan
    return (data)


def get_master_data():
    initdata =[]
    columns = []
    #columns = ['date','time']
    dataframe = pd.DataFrame(initdata,columns=columns)

#    data_set_dir = 'D:\\Project\\Dissertation\\Data Set\\Formated_Dataset\\'
    data_set_dir = 'D:\\Project\\Dissertation\\Test Data\\Dataset_Merge\\Formated_Dataset\\'
    root, Rp_dir, files = os.walk(data_set_dir).__next__()
    for rp in Rp_dir:
        rp_set_dir = data_set_dir + rp + "\\"
        rt, P_dir, files = os.walk(rp_set_dir).__next__()
        for p_dir in P_dir:
            p_set_dir = rp_set_dir + p_dir + "\\"
            rt, P_dir, files = os.walk(p_set_dir).__next__()
            for f_name in files:
                rss_dB = load_data(p_set_dir,f_name)
                data =  pd.DataFrame(rss_dB)
                dataframe = pd.concat([dataframe,data])

    print(dataframe)
    #print(dataframe.info())
    create_master_data(dataframe)        
            
def create_master_data(dataframe):
    export_file_path = 'D:\\Project\\Dissertation\\Test Data\\Dataset_Merge\\Merge Master\\total_test_master.csv'
    if not os.path.exists(os.path.dirname(export_file_path)):
        try:
            os.makedirs(os.path.dirname(export_file_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    dataframe.to_csv (export_file_path, index = False, header=True)


get_master_data()