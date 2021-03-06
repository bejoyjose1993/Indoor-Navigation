# RSSI Based Indoor Navigation (Dissertation Topic)
<h2 id="overview">Overview</h2>

The dissertation's major motivation was to create a precise and accurate indoor navigation system by manipulating an impulsive received signal strength indicator (RSSI)
in a very challenging underground car park environment. To achieve our goals a testbed architecture and fingerprint database were created by recording and pre-processing sensor data by using a combination of solar-powered blue tooth sensors and the BLE-USB bridge receiver. A qualitative analysis was then conducted to evaluate its efficiency. This fingerprint database was further used with prominent machine learning techniques, pathfinding algorithms, floor maps, and an audio module to create a navigation application. Various machine learning algorithms like affinity propagation, k-nearest neighbours, k-means, support-vector machine, and logistic regression were studied in different environments, and affinity propagation was finally used due to its reliability and robust nature, to successfully navigate around the testbed. The entire process is divided into 2 phases: Online Phase and Offline Phase. 



<h2 id="test_bed_architecture">1.1 Testbed And Architecture (Online Phase)</h2>
Our testbed has been set up inside an underground basement car park. The car park consisted of 46 parking slots, out of which 2 are disabled parking spaces. It consists of a single entrance and exit gate and has two building entrances. The entire parking area can be subdivided into 17 alphabetic blocks. Apart from blocks F, G, J, and Q, all other 13 blocks contain three parking slots each. F and J consist of 2 slots, whereas G and Q contain 1. G1 and K1 are reserved disabled parking slots. The block and slot naming are created such that users can locate and perform seamless navigation inside the testbed. The Following captures the 3D view of the carpark chosen as the Testbed.

![Image of Testbed](https://github.com/bejoyjose1993/Indoor-Navigation/blob/master/Images/Block_exp-1.png)


<h2 id="test_bed_architecture">1.2 Reference Point And Measurement Point Architecture (Online Phase)</h2>
Planning and positioning of reference points is an essential task in fingerprint database creation. Reference points will be used as a part of the dataset label in the multilabel fingerprint database. Our architecture contains almost 14 reference points placed approximately at a distance of about 7.20 m away from each other, i.e., adjacent reference points RP-1 and RP-2 are placed 7.20 meters apart.The followings shows the selected and calibrated, Refrence and Measurement points used to collect the RF-Fingerprints.

![Image of Testbed](https://github.com/bejoyjose1993/Indoor-Navigation/blob/master/Images/Reference%20Points-2.png)


Each reference point has 5 measurement points, each resembling a cluster labeled P1, P2, P3, P4, and P5. The fingerprint database will be created by recording sensor data from each measurement point. It will become another crucial dataset label, which will help in determining the position of the user. X and Y coordinates are recorded for each measurement point. Measurements are manually taken with the help of 50-meter open reel tape. 
![Image of Testbed](https://github.com/bejoyjose1993/Indoor-Navigation/blob/master/Images/Messurement_Points_Data-2.png)



<h2 id="test_bed_architecture">1.3 Sensor Placement and Architecture (Online Phase)</h2>
Cypress BLE-Beacon is the sensor used in the testbed. 11 Beacons are set up along the navigation path. The below figure shows the placement of BLE beacons along with their Beacon Ids. Beacons are placed in such a way that signals from at least two beacons are received at every measurement and reference points. The Beacons are kept on the ceiling light upside down such that the lights consistently charge its photosensors, and it can seamlessly transmit with full power.

![Image of Testbed](https://github.com/bejoyjose1993/Indoor-Navigation/blob/master/Images/Sensor_location_ID.png)

<h2 id="test_bed_architecture">1.4 RF Fingerprinf Collection (Online Phase)</h2>
Sensor data is collected from each meaurement point using Cypres Ble Beacon application. Data of all four directions are collected i.e. East, West, North and South for enriching the dataset. RSSI values are then extracted, data is preprocessed and Finger database is created. 

## Code Description
### extractingFingerprint.py 
The above python file in the "Code Base/1 Data Preprocessing and Visualization" is used to extract the RSSI value and preprocess the dataset.

### combineMasterData.py  
The above file in the "Code Base/1 Data Preprocessing and Visualization" is used to combine all individual measurement point specific datset into single Master dataset.

### affinity_propagation_data_creation.py, coarse_rf_creation.py and kmeans_with_my_data.py 
These file are used to further preproces the dataset to make it suitable for affinity propagation, coarse localization and Kmeans models. In this cluster are created and cluster
details are stored along with master data in CSV files."Code Base/1 Data Preprocessing and Visualization" contains these files.

### import_csv_to_db.py  and database_entry.py 
These file are used to store the created CSV files into respective databases."Code Base/1 Data Preprocessing and Visualization" contains these files.

### Kmeans_with_saved_clusters.py  
This file was initial used to check the useability of the master data containing the cluster informations. "Code Base/1 Data Preprocessing and Visualization" contains these files.


<h2 id="test_bed_architecture">1.5 Data Visualization And Quality Analysis (Online Phase)</h2>

Below Images shows the 3D scatter plot generated during visualization. It is evident from Figures that signal strength of access points 1 and 4 is maximum around their corresponding blocks.

![Image of Testbed](https://github.com/bejoyjose1993/Indoor-Navigation/blob/master/Images/AP-1_Signal_Strength.png)
![Image of Testbed](https://github.com/bejoyjose1993/Indoor-Navigation/blob/master/Images/AP-4_Signal_Strength.png)


## Code Description
### visualizeMasterData.py  
This file present in "Code Base/1 Data Preprocessing and Visualization" was used to visualize the quality of collected master data.



<h2 id="test_bed_architecture">2.1 Module Evaluation (Offline Phase)</h2>

### Testcase Descriptions
 To analyze the effect of the environment on the model prediction, we have created various test scenarios. These test scenarios include test data collected from the testbed at various instances with different vehicle intensity levels.
Multiple scenarios have been created as follows, 
*	Scenario 1: Base scenarios as, using the original dataset by dividing it into train and test dataset.
* Scenario 2: Test data as fingerprints collected when 14 vehicles were present in the dataset.
*	Scenario 3: Test data as fingerprints collected when testbed had vehicle intensity as 23. 
*	Scenario 4: In our fourth scenario, we use a combination of 21 vehicle and original dataset to train models and use the 13 vehicle data to test the results.  
The different scenarios are taken to study various problems in RSSI based navigation.

We use the scenarios mentioned above and datasets to compute the performance of multiple models. We use distance-based models as evaluation models as RSSI localization is based on distance. Models such as K-Nearest Neighbour, K-Means, Stare Vector Machine, Logistic Regression, and Affinity propagation were studied and analyzed.

## Code Description
### evaluaing_logistic_regression.py, evaluating_svm_learn_model.py, evaulating_affinity_propagation.py, evaulating_coarse_knn.py evaulating_simple_knn.py and knn_learning.py
Above source code files present in "Code Base/2 Model Evaluation/" is used to evaluate performance of RSSI based preiction using different Machine learning models.
Accuracy, Precision Recall and Prediction Errors are calculated to find the best Model.
After carefull analysis we selected Affinity Propagation as our final model. The following figures shows the position error of the final selected model.
![Image of Testbed](https://github.com/bejoyjose1993/Indoor-Navigation/blob/master/Images/positioning_error_for_xaxis.png)
![Image of Testbed](https://github.com/bejoyjose1993/Indoor-Navigation/blob/master/Images/y_axis_Scenario_1.png)



<h2 id="test_bed_architecture">2.2 Audio Visual Module (Offline Phase)</h2>
The basis of visual module is a floor map of the location to be traversed. We use the MagicPlan app during the fieldwork phase to develop the floor map. The App requires manually capturing the corners of the testbed and then creating a 3D and 2D floor maps of the testbed. We have used a 2D floor map to navigate on the testbed. The floor map is then combined and aligned with the matplotlib graph structure and used to plot the user’s current location. User’s current location is designed to continually change color to distinguish from other points in the graph. We have tried to capture the direction of travel using directed triangles and destination is captured with a red circle. The purpose of this module is to visually locate and guide the user to the required destination. Visual module combined with the audio module becomes an effective means of navigation around the testbed.

![Image of Testbed](https://github.com/bejoyjose1993/Indoor-Navigation/blob/master/Images/Nav_using_corase_KNN.png)

## Code Description
### math_plot_image.py, ploting_obj.py, reading_the_latest_file.py 
The above python files in the "Code Base/3 Visual Module/" were used to create Visual Floormap module.

### Playing_Audio_Files.py, Text_to_speach.py, Text_to_speach_new.py 
The above python files in the "Code Base/4 Audio Module/" were used to create Audio guiding module.


<h2 id="test_bed_architecture">2.3 Application Workflow (Offline Phase)</h2>

Here we will discuss step by step functioning of our user application. The online phase's generated fingerprints are stored in the database as two separate tables, ‘master_data’ and aggregated ‘affinity_prop_master’ with 16,824 and 220 data, respectively. This data acts as a base for the offline application phase. The following step describes the application workflow.
*Step 1: The user collects the current RSSI data from the current location using the Cypress BLE beacon app and stores it in a CSV file.
*Step 2: The application initially trains the selected model with an aggregated ‘affinity_prop_master’ dataset.  This is the Coarse localization stage.
*Step 3: The user’s current location is predicted using the current RSSI readings using the model. This becomes the source node.
*Step 4: The user is prompted to enter the destination location. This is considered as the destination node.
*Step 5: Dijkstra’s pathfinding algorithm is used to compute the shortest path and calculate the distance from the source to the destination node. 
*Step 6: The selected model is then trained with the cluster member present in the ‘master_data’ dataset for the current Reference Point.
*Step 7: The user’s actual location is predicted, and the geo-coordinate for the user is computed. This constitutes the Fine localization stage.
*Step 8: Audio-Visual module is then used to locate the user, find the destination, and guide the user to the desired location.
*Step 9: The user traverses forward towards the prompted location and tries to collect the new RSSI readings.
*Step 10: If the current location matches the destination location, then “Destination Reached” else “Go to Step: 3”.
Thus, by integrating all the individual modules, the application is designed to allow users to navigate the indoor environment freely.

## Code Description
### corase_KNN_navigation.py, kmeans_rssi_navigation.py, knn_navigation.py, rssi_navigation_with_affinity_prop.py 
Finally we have successfully navigated using 4 differnt models. Thought Affinity Propagation was superior than other models, Hence chosen as our final model. The above files have the final navigation apllication codes. "Code Base/5 Navigation Models/" contains these files.

<h2 id="test_bed_architecture">3 Application Demo Folder</h2>
Application Demo folder has the video demo of the application.

<h2 id="test_bed_architecture">4 Database-Dump Folder</h2>
Database-Dump folder has the msql dump file used for our application.

<h2 id="test_bed_architecture">5 Datasets Folder</h2>
Datasets folder has both collected and formated dataset divided on the basis of refrence points, meassurement points and orinetations.

<h2 id="test_bed_architecture">6 Master Data Folder</h2>
Master Data folder contains multiple Master CSV files which was later stored in DB.

<h2 id="test_bed_architecture">7 Evaluation_Graphs Folder</h2>
Evaluation_Graphs folder contains various graphs and observations obtained during the study.

