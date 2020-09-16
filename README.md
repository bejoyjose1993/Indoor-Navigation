# RSSI Based Indoor Navigation (Dissertation Topic)
<h2 id="overview">Overview</h2>

The dissertation's major motivation was to create a precise and accurate indoor navigation system by manipulating an impulsive received signal strength indicator (RSSI)
in a very challenging underground car park environment. To achieve our goals a testbed architecture and fingerprint database were created by recording and pre-processing sensor data by using a combination of solar-powered blue tooth sensors and the BLE-USB bridge receiver. A qualitative analysis was then conducted to evaluate its efficiency. This fingerprint database was further used with prominent machine learning techniques, pathfinding algorithms, floor maps, and an audio module to create a navigation application. Various machine learning algorithms like affinity propagation, k-nearest neighbours, k-means, support-vector machine, and logistic regression were studied in different environments, and affinity propagation was finally used due to its reliability and robust nature, to successfully navigate around the testbed.



<h2 id="test_bed_architecture">Testbed And Architecture</h2>
Our testbed has been set up inside an underground basement car park. The car park consisted of 46 parking slots, out of which 2 are disabled parking spaces. It consists of a single entrance and exit gate and has two building entrances. The entire parking area can be subdivided into 17 alphabetic blocks. Apart from blocks F, G, J, and Q, all other 13 blocks contain three parking slots each. F and J consist of 2 slots, whereas G and Q contain 1. G1 and K1 are reserved disabled parking slots. The block and slot naming are created such that users can locate and perform seamless navigation inside the testbed. The Following captures the 3D view of the carpark chosen as the Testbed.

![Image of Testbed](https://github.com/bejoyjose1993/Indoor-Navigation/blob/master/Images/Block_exp-1.png)


<h2 id="test_bed_architecture">Reference Point And Measurement Point Architecture</h2>
Planning and positioning of reference points is an essential task in fingerprint database creation. Reference points will be used as a part of the dataset label in the multilabel fingerprint database. Our architecture contains almost 14 reference points placed approximately at a distance of about 7.20 m away from each other, i.e., adjacent reference points RP-1 and RP-2 are placed 7.20 meters apart.The followings shows the selected and calibrated, Refrence and Measurement points used to collect the RF-Fingerprints.

![Image of Testbed](https://github.com/bejoyjose1993/Indoor-Navigation/blob/master/Images/Reference%20Points-2.png)


Each reference point has 5 measurement points, each resembling a cluster labeled P1, P2, P3, P4, and P5. The fingerprint database will be created by recording sensor data from each measurement point. It will become another crucial dataset label, which will help in determining the position of the user. X and Y coordinates are recorded for each measurement point. Measurements are manually taken with the help of 50-meter open reel tape. 
![Image of Testbed](https://github.com/bejoyjose1993/Indoor-Navigation/blob/master/Images/Messurement_Points_Data-2.png)



<h2 id="test_bed_architecture">Reference Point And Measurement Point Architecture</h2>
Cypress BLE-Beacon is the sensor used in the testbed. 11 Beacons are set up along the navigation path. The below figure shows the placement of BLE beacons along with their Beacon Ids. Beacons are placed in such a way that signals from at least two beacons are received at every measurement and reference points. The Beacons are kept on the ceiling light upside down such that the lights consistently charge its photosensors, and it can seamlessly transmit with full power.

![Image of Testbed](https://github.com/bejoyjose1993/Indoor-Navigation/blob/master/Images/Sensor_location_ID.png)
