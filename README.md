# Machine Learning CS4375-A3.1
## Description
Perform k-means clustering. You are free to use any library or package 
such as scikitlearn. You have to try at least 5 different values of k and 
report the Sum of Squared Error (SSE) function, defined as:

SSE = SUM(dist^2 (mi, x))

where mi is the centroid of the ith cluster.

## Report Format (output):
Your report should be in the following format: 

| Experiment Number | Value of k | SSE Value |

## Contains:
- k-means.py 
- README.md

## Run Instructions 
Once in directory, enter one of two command (dependent on machine):

python **k-means.py** <br>
or <br>
python3 **k-means.py**

## Misc
Imported Libraries:
- import pandas as pd
- import requests
- import io
- from sklearn.cluster import KMeans
- from sklearn.preprocessing import MinMaxScaler
- import matplotlib.pyplot as plt

Link to data-set:
- https://archive.ics.uci.edu/ml/datasets/seeds