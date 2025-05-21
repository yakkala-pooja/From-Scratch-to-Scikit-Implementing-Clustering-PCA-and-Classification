''' Import Libraries'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class Classifier:
    ''' This is a class prototype for any classifier. It contains two empty methods: predict, fit'''
    def __init__(self):
        self.model_params = {}
        pass
    
    def predict(self, x):
        '''This method takes in x (numpy array) and returns a prediction y'''
        raise NotImplementedError
    
    def fit(self, dataframe):
        '''This method is used for fitting a model to data: x, y'''
        raise NotImplementedError
        
        
        
class KMeans(Classifier):
    '''No init function, as we inherit it from the base class'''
    def fit(self, data, k=2, tol = 0.01):
        '''k is the number of clusters, tol is our tolerance level'''
        '''Randomly choose k vectors from our data'''
        self.centroids = data[np.random.choice(data.shape[0], k, replace=False)]
        while True:
            cluster = [[] for _ in range(k)]
            for point in data:
                dist = [self.calc_distance(point, c) for c in self.centroids]
                clusterid = np.argmin(dist)
                cluster[clusterid].append(point)
            new = np.array([np.mean(c, axis=0) if len(c) > 0 else self.centroids[i] for i, c in enumerate(cluster)])
            if np.all(np.abs(new - self.centroids) < tol):
                break
            self.centroids = new
        self.cluster = cluster
        
    def predict(self, x):
        '''Input: a vector (x) to classify
           Output: an integer (classification) corresponding to the closest cluster
           Idea: you measure the distance (calc_distance) of the input to 
           each cluster centroid and return the closest cluster index'''
        dist = [self.calc_distance(x, centroid) for centriod in self.centriods]
        classification = np.argmin(distances)
        return classification
    
    def calc_distance(self, point1, point2):
        '''Input: two vectors (point1 and point2)
           Output: a single value corresponding to the euclidan distance betwee the two vectors'''
        ans = np.sqrt(np.sum((point1 - point2) ** 2))
        return ans
        
        
