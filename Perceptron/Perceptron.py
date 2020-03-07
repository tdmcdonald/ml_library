import numpy as np 
import copy

class Perceptron:
    def __init__(self,mode ):
          self.prediction_weights= None
          self.mode=mode
    def fit(self,training_features, labels, learning_rate, epochs):
        training_feats=copy.deepcopy(training_features)
        training_labels=copy.deepcopy(labels)
        indices=np.arange(0,len(training_labels))
        if(self.mode=='margin'):
            weights=np.zeros(len(training_feats[0])+1, dtype=float)
            # print(weights)
            b=0
            for j,v in enumerate(weights):
                weights[j]+=.0000001
            for t in np.arange(0, epochs):
                np.random.shuffle(indices)
                # print(indices)
                training_feats=copy.deepcopy(training_features[indices,])
                training_labels=copy.deepcopy(labels[indices])
                # print(training_labels.shape)
                for i,sample in enumerate(training_feats):
                    y_prime=training_labels[i]*weights[1:,np.newaxis].T.dot(training_feats[i])+weights[0]
                    if y_prime[0] <=0: #incorrect pred
                        weights[1:]= weights[1:]+learning_rate*(training_labels[i]*training_feats[i])
                        weights[0] += training_labels[i]*learning_rate
            self.prediction_weights=weights
            return weights
        elif(self.mode=='vote'):
            weights={}
            weights[0]=np.zeros(len(training_feats[0])+1, dtype=float)
            m=0
            C={}
            C[0]=0
            for t in np.arange(0, epochs):
                np.random.shuffle(indices)
                training_feats=copy.deepcopy(training_features[indices,])
                training_labels=copy.deepcopy(labels[indices])
                for i,sample in enumerate(training_feats):
                    y_prime=training_labels[i]*weights[m][1:,np.newaxis].T.dot(training_feats[i])+weights[m][0]
                    # y_prime=self.getVotedPrediction(weights,C,training_feats[i])
                    if y_prime[0]<=0: #incorrect pred
                        weights[m+1]=np.zeros(len(training_feats[0])+1, dtype=float)
                        weights[m+1][1:]= weights[m][1:]+learning_rate*(training_labels[i]*training_feats[i])
                        weights[m+1][0] += weights[m][0]+training_labels[i]*learning_rate
                        m=m+1
                        C[m]=1
                    else:
                        C[m]+=1
                # print(m)
            return_val=[]
            for i in np.arange(0,m):
                return_val.append((weights[i],C[i]))
            # print('len k is ', m)
            self.prediction_weights=return_val
            return return_val


        elif(self.mode=='average'):
            weights=np.zeros(len(training_feats[0])+1, dtype=float)
            ave=np.zeros(len(training_feats[0])+1, dtype=float)

            for t in np.arange(0, epochs):
                np.random.shuffle(indices)
                training_feats=copy.deepcopy(training_features[indices,])
                training_labels=copy.deepcopy(labels[indices])
                for i,sample in enumerate(training_feats):
                    y_prime=training_labels[i]*weights[1:,np.newaxis].T.dot(training_feats[i])+weights[0]
                    if y_prime[0] <=0: #incorrect pred
                        weights[1:]= weights[1:]+learning_rate*(training_labels[i]*training_feats[i])
                        weights[0] += training_labels[i]*learning_rate
                    ave=ave+weights
            self.prediction_weights=ave
            return weights
        else:
            print('mode not valid')
            return
    def getVotedPrediction(self,weights, counts, sample):
        pred=0
        for m in weights:
            pred+=counts[m]*np.sign(weights[m][1:,np.newaxis].T.dot(sample)+weights[m][0])
        if(pred>0):
            return 1
        else:
            return -1
    def sign(self, weights, sample):
        res= weights[:,np.newaxis].T.dot(sample)
        if res>0:
            return 1
        else:
            return -1
    def predict(self,feats):
        print('predicting')

        preds=[]
        if(self.mode=="margin"):
            weights_t=np.transpose(self.prediction_weights)
            for i, sample in enumerate(feats):
                pred= self.prediction_weights[1:,np.newaxis].T.dot(feats[i])+self.prediction_weights[0]
                if(pred>0):
                    preds.append(1)
                else:
                    preds.append(-1)
        elif(self.mode=="vote"):
            for i, sample in enumerate(feats):
                pred=0
                for m,v in enumerate(self.prediction_weights):
                    # print(m,v)
                    pred+=v[1]*np.sign((v[0][1:,np.newaxis].T.dot(feats[i])+v[0][0]))
                    
                if(pred>=0):
                    preds.append(1)
                else:
                    preds.append(-1)
        elif(self.mode=="average"):
            for i, sample in enumerate(feats):
                pred= self.prediction_weights[1:,np.newaxis].T.dot(feats[i])+self.prediction_weights[0]
                if(pred>=0):
                    preds.append(1)
                else:
                    preds.append(-1)

        return preds
