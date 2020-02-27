
import numpy as np 
import copy
import math
import statistics
import matplotlib.pyplot as plt

class LMS:

    def __init__(self, gradient_decent_type, learning_rate=1):
        self.learning_rate=learning_rate
        self.decent=gradient_decent_type
        self.total_error=math.inf
        self.prediction_weights=np.array(0)

    
    def fit(self, training_feats, training_labels,  numerical_cols=[],unknown_cols=[] ):
       
        n = len(training_labels)
        pred_features=training_feats
        #training_feats=np.transpose(training_feats)
        new_labels=np.zeros_like(training_labels,dtype=float)
        self.total_error= math.inf
        threshold=1e-6
        weights=np.zeros_like(training_feats[0], dtype=float)
        iter_count=0
        # print(training_feats.shape, training_labels.shape, weights.shape,weights[:,np.newaxis].T.shape)
        costs=[]
        iters=[]
        if self.decent=='batch':
            while self.total_error>threshold:
                dj=np.zeros_like(weights, dtype=float)
                for i,sample in enumerate(training_feats):
                    for j, feat in enumerate(weights):
                        # print(training_feats[i].shape,weights[:,np.newaxis].T.shape, training_labels[i].shape,
                        # weights.shape,training_labels[i]-np.dot(weights[:,np.newaxis].T,training_feats[i]),training_feats[i][j],np.transpose(weights)*training_feats[i]*training_feats[i][j]  )
                        # print(weights[:,np.newaxis].T,training_feats[i],np.dot(weights[:,np.newaxis].T,training_feats[i]),training_labels[i]-np.dot(weights[:,np.newaxis].T,training_feats[i]))
                        # print(weights.T.dot(training_feats[i]))
                        # wtxi=np.dot(weights.T,(training_feats[i]))
                        # print('dot is ',training_labels[i]- wtxi)
                        dj[j]+=-(training_labels[i]-weights.T.dot(training_feats[i]))*training_feats[i][j]
                # print(dj)
               
                # dj=dj/len(training_labels)
                new_weights=weights-self.learning_rate*dj
                # print(weights, new_weights)
                self.total_error= np.linalg.norm(weights-new_weights)
                # print(self.total_error, self.learning_rate)
                weights=new_weights
                costs.append(self.total_error)
                iters.append(iter_count)
                iter_count+=1
                if(iter_count%100 ==0):
                    self.learning_rate=self.learning_rate/2
        else:
            print('stochastic')
            while self.total_error>threshold:
                new_weights=np.zeros_like(weights, dtype=float)
                rand_feature=np.random.randint(0,len(training_labels))
                for j, feat in enumerate(weights):
                    # print(training_feats[i].shape,weights[:,np.newaxis].T.shape, training_labels[i].shape,
                    # weights.shape,training_labels[i]-np.dot(weights[:,np.newaxis].T,training_feats[i]),training_feats[i][j],np.transpose(weights)*training_feats[i]*training_feats[i][j]  )
                    # print(weights[:,np.newaxis].T,training_feats[i],np.dot(weights[:,np.newaxis].T,training_feats[i]),training_labels[i]-np.dot(weights[:,np.newaxis].T,training_feats[i]))
                    # print(weights.T.dot(training_feats[i]))
                    # wtxi=np.dot(weights.T,(training_feats[i]))
                    # print('dot is ',training_labels[i]- wtxi)
                    new_weights[j]=weights[j] +self.learning_rate*(training_labels[rand_feature]-weights[:,np.newaxis].T.dot(training_feats[rand_feature]))*training_feats[rand_feature][j]
            # print(dj)
               
                # print(weights, new_weights)
                self.total_error= np.linalg.norm(weights-new_weights)
                weights=new_weights
                costs.append(self.total_error)
                iters.append(iter_count)
                iter_count+=1
                if(iter_count%100 ==0):
                    self.learning_rate=self.learning_rate/2

        # plt.plot(costs)
        # plt.show()
        self.prediction_weights=weights
        return costs,iters, weights

    def predict(self, feats):
        print('predicting')
        preds=[]
        weights_t=np.transpose(self.prediction_weights)
        print(weights_t.shape,(self.prediction_weights),feats[0].shape)
        for i, sample in enumerate(feats):
            pred= self.prediction_weights[:,np.newaxis].T.dot(feats[i])
            preds.append(pred)

        return preds

        

