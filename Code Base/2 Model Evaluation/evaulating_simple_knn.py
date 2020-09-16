
from numpy import *
from pandas import DataFrame
import mysql.connector
import pandas as pd 
import time
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, ConfusionMatrixDisplay

path_to_test_data = "D:\\Project\\Dissertation\\Test Data\\Test_with_13_vehicles\\Test Master\\total_test_master.csv"
#path_to_test_data = "D:\\Project\\Dissertation\\Test Data\\Test_with_21_vehicles\\Test Master\\total_test_master.csv"
path_to_train_data = "D:\\Project\\Dissertation\\Test Data\\\Dataset_Merge\\Merge Master\\total_train_master.csv"
mydb = mysql.connector.connect(
    host = "localhost",
    user = "root",
    password = "root",
    database = "indoor_nav_using_affinity_prop"
)

def load_data():
    # read test data
    my_cusor = mydb.cursor()
    block = []
    select_stmt = "SELECT * FROM indoor_nav_using_affinity_prop.master_data"
    my_cusor.execute(select_stmt)
    data = DataFrame(my_cusor.fetchall())
    field_names = [i[0] for i in my_cusor.description]
    data.columns = field_names  
    my_cusor.close()    
    return data

def load_train_data(path_to_train):
    data = pd.read_csv(path_to_train, names=['date','time','0x0001','0x0002','0x0003','0x0004','0x0005','0x0006','0x0012','0x0013','0x0014','0x0015','0x0016','x_axis','y_axis','z_axis','orientation','reference_points','messurement_points'])
    data = data.iloc[1:,:]
    return data

def load_test_data(path_to_data):
    data = pd.read_csv(path_to_data, names=['date','time','0x0001','0x0002','0x0003','0x0004','0x0005','0x0006','0x0012','0x0013','0x0014','0x0015','0x0016','x_axis','y_axis','z_axis','orientation','reference_points','messurement_points'])
    data = data.iloc[1:,:]
    return data

def get_RSS(data):
    df =  pd.DataFrame(data)
    rss_df = df[['0x0001','0x0002','0x0003','0x0004','0x0005','0x0006','0x0012','0x0013','0x0014','0x0015','0x0016']]
    ref_df = df[['reference_points','messurement_points']]
    #loc_df = df[['reference_points','messurement_points']]
    return rss_df, ref_df     


#train = load_data()
train = load_train_data(path_to_train_data)

X_train, y_train  = get_RSS(train)
#11
#X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2,random_state=11)
test = load_test_data(path_to_test_data)
X_test, y_test  = get_RSS(test)

print(X_test)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

#X_train = df.drop(['reference_points','Messurement_points'],axis=1)
#y_train = df[['reference_points','Messurement_points']]
X_train = X_train.fillna(0)

#Time
tsum = 0
t = time.process_time()

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=1, p=2,
           weights='uniform')
x_test = X_test.fillna(0)
acc_pred = knn.predict(x_test)

#Time
tsum += time.process_time() - t

true_label = []
pred_label = []
for i in range(len(y_test)):
    y_temp = y_test.iloc[[i]].values
    Y_test = y_temp.ravel()
    true_label.append(Y_test[0])
    pred_label.append(acc_pred[i][0])
display_labels = ["RP-1", "RP-2", "RP-3", "RP-4", "RP-5", "RP-6", "RP-7", "RP-8", "RP-9", "RP-10", "RP-11"]
cm = confusion_matrix(pred_label, true_label,labels= display_labels)
print("confusion_matrix")
print(cm)
print("")


print("precision_score")
print(precision_score(true_label, pred_label,average='macro'))
print(precision_score(true_label, pred_label,average='micro'))
print(precision_score(true_label, pred_label,average='weighted'))
#print(precision_score(true_label, pred_label,average=None))

print("recall")
print(recall_score(true_label, pred_label,average='macro'))
print(recall_score(true_label, pred_label,average='micro'))
print(recall_score(true_label, pred_label,average='weighted'))

print("accuracy")
print(accuracy_score(true_label, pred_label))


#tn, fp, fn, tp = perf_measure(true_label, pred_label)
#print("tn",tn)
#print("fp",fp)
#print("fn",fn)
#print("tp",tp)
print()
print("Time")
print(tsum)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=display_labels)

disp = disp.plot()
plt.show()

#Time
