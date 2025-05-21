import numpy as np

class FeatureReduction():
    
    def __init__(self):
        self.model_params = {}
    
    def fit(self, data):
        pass
    
    def predict(self, data):
        pass
    

class PrincipleComponentAnalysis(FeatureReduction):
    '''self.model_params is where you will save your principle components (up to LoV)'''
    ''' Its useful to use a projection matrix as your only param'''
    def fit(self, data, thresh=0.95, plot_var = True):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        stdata = (data - mean)/std
        
        covm = np.cov(stdata.T)
        eigenval, eigenvec = np.linalg.eig(covm)
        sortid = np.argsort(eigenval)[::-1]
        eigenval = eigenval[sortid]
        eigenvec = eigenvec[:, sortid]
        
        idx = np.argmax(np.abs(eigenvec), axis = 0)
        sign = np.sign(eigenvec[idx, range(eigenvec.shape[0])])
        eigenvec *= sign[np.newaxis, :]
        
        var = self.calc_variance_explained(eigenval)
        cumvar = np.cumsum(var)
        num = np.argmax(cumvar >= thresh) + 1
        
        self.model_params['projectionMat'] = eigenvec[:, :num]
        self.model_params['mean'] = mean
        self.model_params['std'] = std
        self.model_params['variance'] = var
        self.model_params['cumlativevar'] = cumvar
        
    def predict(self, data):
        stdata = (data - self.model_params['mean'])/self.model_params['std']
        return np.dot(stdata, self.model_params['projectionMat'])
    
    def calc_variance_explained(self, eigen_values):
        '''Input: list of eigen values
           Output: list of normalized values corresponding to percentage of information an eigen value contains'''
        totvar = np.sum(eigen_values)
        variance_explained = eigen_values / totvar
        return variance_explained