''' 
Guilherme Michel Lima de Carvalho

Implementação do classificador Naive-Bayes usando numpy

'''

import numpy as np


class NaiveBayes:
    def indices_data(self,y): 
        self.n_classes = len(np.unique(y))
        classes_indices = []
        for i in range(self.n_classes): 
            classes_indices.append(np.where(y==i)[0].tolist())
        return classes_indices

    def prob_classes(self,y,classes_indices):
        self.probs_class = []
        for i in range(len(self.classes_indices)):
            self.probs_class.append(len(self.classes_indices[i])/len(y))
        return self.probs_class

    def estimativas(self,X,classes_indices):
        n_classes = len(classes_indices) 
        n_variaveis = X.shape[1]  
        self.estimativas_medias = np.zeros((n_classes,n_variaveis),dtype=np.float64) 
        self.estimativas_var = np.zeros((n_classes,n_variaveis),dtype=np.float64) 
        for i in range(n_classes):
            self.estimativas_medias[i,:] = X[classes_indices[i]].mean(axis=0) 
            self.estimativas_var[i,:] = X[classes_indices[i]].var(axis=0) 
        return self.estimativas_medias,self.estimativas_var

    def gaussiana(self,media,var,x):
        return (1/np.sqrt(2*(np.pi)*var))*np.exp(-((x-media)**2)/(2*var))

    
    def ajusta(self,X,y):
        self.n_classes = len(np.unique(y)) 
        self.classes_indices = indices_data(y) 
        self.probs_class = self.prob_classes(y,self.classes_indices) 
        self.media, self.var = self.estimativas(X,self.classes_indices) 

    def prediz(self,X):
        n_classes = self.media.shape[0]
        n_observacoes = X.shape[0]
        probabilidade_predita = np.zeros((n_observacoes,n_classes),dtype=np.float64) 
        for i in range(n_classes):
            probabilidade_predita[:,i] = np.sum(np.log(gaussiana(self.media[i,:],self.var[i,:],X)),axis=1) + np.log(self.probs_class[i]) 
        return np.argmax(probabilidade_predita,axis=1)

