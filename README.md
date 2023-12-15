# Intrusion Detection Systems
CSCI735 - Foundations of Intelligent Security Systems.
Rochester Institute of Technology, Fall 2023

This course project deals with initial exploration of modelling Intrusion 
detection systems. As a part of the project we aim to utilise machine learning and AI 
techniques to model a misuse based and anomaly based Intrusion detection 
system (IDS).
The project is broken into 3 phases with corresponding documentation for the same.

<b>Dataset </b> - https://www.kaggle.com/datasets/galaxyh/kdd-cup-1999-data <br>
File location in Project - [datasetKDD](data/datasetKDD)


1. <b> Phase 1 </b>

This phase focuses on identifying the data to work on, data preperation and 
cleaning as well as experimenting with current state of the art IDS like 
Snort and Suricata. From our initial exploration with the data set we had, 
we resorted to using classifier techniques to develop generic IDS in python.<br>
Phase 2 Code - [idsClassifier.py](idsClassifier.py)<br>
Phase 2 Report - [Project_Phase_1.pdf](Project_Phase_1.pdf)

2. <b> Phase 2 </b>

This phase focuses on realising the shortcomings in Phase 1 and using 
Artificial Neural Networks (ANN) to recreate the IDS models for misuse and 
anomaly based IDS.<br>
Phase 2 Code - [annClassifier.py](annClassifier.py)<br>
Phase 2 Report - [Project_Phase_2.pdf](Project_Phase_2.pdf)

3. <b> Phase 3 </b>
This phase deals with comparing the approaches in Phase 1 and 2, drawing up 
statistics and addressing shortcomings and room for improvement.
Phase 3 Report - [Project_Phase_3.pdf](Project_Phase_3.pdf)
