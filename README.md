# RSSI Based Indoor Navigation (Dissertation Topic)
<h2 id="overview">Overview</h2>

The dissertation's major motivation was to create a precise and accurate indoor navigation system by manipulating an impulsive received signal strength indicator (RSSI)
in a very challenging underground car park environment. To achieve our goals a testbed architecture and fingerprint database were created by recording and pre-processing sensor data by using a combination of solar-powered blue tooth sensors and the BLE-USB bridge receiver. A qualitative analysis was then conducted to evaluate its efficiency. This fingerprint database was further used with prominent machine learning techniques, pathfinding algorithms, floor maps, and an audio module to create a navigation application. Various machine learning algorithms like affinity propagation, k-nearest neighbours, k-means, support-vector machine, and logistic regression were studied in different environments, and affinity propagation was finally used due to its reliability and robust nature, to successfully navigate around the testbed.



<h2 id="test_bed_architecture">Testbed And Architecture</h2>
The Following captures the 3D view of the carpark chosen as the Testbed.

![Image of Testbed](https://github.com/bejoyjose1993/Indoor-Navigation/blob/master/Images/Block_exp-1.png)


<h2 id="test_bed_architecture">Reference Point And Measurement Point Architecture</h2>
The followings shows the selected and calibrated, Refrence and Measurement points used to collect the RF-Fingerprints.

![Image of Testbed](https://github.com/bejoyjose1993/Indoor-Navigation/blob/master/Images/Reference%20Points-2.png)
![Image of Testbed](https://github.com/bejoyjose1993/Indoor-Navigation/blob/master/Images/Messurement_Points_Data-2.png)


<h2 id="test_bed_architecture">Reference Point And Measurement Point Architecture</h2>
Below Image gives an overview of Sensor Placements Accross the Testbed.

![Image of Testbed](https://github.com/bejoyjose1993/Indoor-Navigation/blob/master/Images/Sensor_location_ID.png)
