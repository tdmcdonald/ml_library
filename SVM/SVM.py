import numpy as np 
import copy
from scipy.optimize import minimize
class SVM:
    def __init__(self,mode ):
          self.prediction_weights= None
          self.mode=mode
          self.lam=0
          self.train_dat=None
    def fit(self,training_features, labels, learning_rate, epochs, C=1,y_0=1,lamb=2 ):
        training_feats  =copy.deepcopy(training_features)
        training_labels=copy.deepcopy(labels)
        indices=np.arange(0,len(training_labels))
        N=len(training_feats)   
        self.lam=lamb
        np.random.seed(41)
        if(self.mode=='primal'):
            weights=np.zeros(len(training_feats[0])+1, dtype=float)
            # print(weights)
            it=1
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
                    if y_prime[0] <=1: #incorrect pred
                        weights[1:]= weights[1:]-learning_rate*weights[1:]+learning_rate*(training_labels[i]*training_feats[i]*C*N)
                        weights[0] -= training_labels[i]*learning_rate
                    else:
                        weights[1:]= weights[1:]-learning_rate*weights[1:]
                        weights[0] =  weights[0]-learning_rate*weights[0]

                # learning_rate=y_0/(1+(y_0/C)*it)
                learning_rate=y_0/(1+it)

                it+=1
                # print(learning_rate)
            self.prediction_weights=weights
            return weights
        elif(self.mode=='dual'):
            weights=np.zeros(len(training_feats[0])+1, dtype=float)
            m=0
            print(training_feats.shape,training_labels.shape)
            # C={}
            # C[0]=0
            def find_hes(X,Y):
                hes = np.zeros((X.shape[0],X.shape[0]))
                for row in range(X.shape[0]):
                    for col in range(X.shape[0]):
                        hes[row,col] = np.dot(X[row,:],X[col,:])*Y[row]*Y[col]
                return hes
            
            H=find_hes(training_feats,training_labels)

            def loss(alphas):
                return 0.5 * np.dot(alphas.T, np.dot(H, alphas)) - np.sum(alphas)
                
            def jac(alphas):
                return np.dot(alphas.T,H)-np.ones(alphas.shape[0])

            A = training_labels# sum of alphas is zero
            cons = ({'type':'eq',
                'fun':lambda alphas: np.dot(A,alphas),
                'jac':lambda alphas: A},
                {'type':'eq',
                'fun':lambda alphas: np.dot(A,alphas),
                'jac':lambda alphas: A})    
            bounds = [(0,None)]*training_feats.shape[0] #alpha>=0
            x0 = np.random.rand(training_feats.shape[0])
            opt={}
            opt['maxiter']=None
            sol = minimize(loss, x0, jac=jac, constraints=cons, method='SLSQP', bounds = bounds,options=opt)
            res=sol.x
            res[np.isclose(res, C)] = C
            # print(res)
            svs =  np.where((0.0001 < res) & (res < C))[0]
            # print(svs)
            alphas=sol.x[svs]
            # print(alphas)
            # print(svs)
            w = np.sum(alphas*training_labels[svs]*training_feats[svs,:].T,axis = 1)[:,np.newaxis]
            b = np.mean(training_labels[svs]-np.dot(training_feats[svs,:],w))
         
            for i,we in enumerate(w):
                weights[i+1]=w[i]
            weights[0]=b
            self.prediction_weights=weights
            return weights


        elif(self.mode=='kernel'):
            self.train_dat=training_features
            weights=np.zeros(len(training_feats[0])+1, dtype=float)
            ave=np.zeros(len(training_feats[0])+1, dtype=float)
            def gaussian_kernel(x, y):
                return np.exp(-np.sum(np.square(x - y)) / self.lam)
            kernel = lambda x, y: gaussian_kernel(x, y)


            def gram(X, k):
                N = len(X)
                K = np.empty((N, N))
                for i in range(N):
                    for j in range(N):
                        K[i, j] = k(X[i], X[j])

                return K
            weights=np.zeros(len(training_feats[0])+1, dtype=float)
            m=0
            print(training_feats.shape,training_labels.shape)
            # C={}
            # C[0]=0
            K = gram(training_feats, kernel)

            def find_hes(X,Y):
                hes = np.zeros((X.shape[0],X.shape[0]))
                for row in range(X.shape[0]):
                    for col in range(X.shape[0]):
                        hes[row,col] = np.dot(X[row,:],X[col,:])*Y[row]*Y[col]
                return hes
            
            H=find_hes(K,training_labels)

            def loss(alphas):
                return 0.5 * np.dot(alphas.T, np.dot(H, alphas)) - np.sum(alphas)
                
            def jac(alphas):
                return np.dot(alphas.T,H)-np.ones(alphas.shape[0])

            A = training_labels# sum of alphas is zero
            cons = {'type':'eq',
                'fun':lambda alphas: np.dot(A,alphas),
                'jac':lambda alphas: A}
            bounds = [(0,None)]*training_feats.shape[0] #alpha>=0
            x0 = np.random.rand(training_feats.shape[0])
            opt={}
            opt['maxiter']=1000
            sol = minimize(loss, x0, jac=jac, constraints=cons, method='SLSQP', bounds = bounds,options=opt)

            svs = sol.x>0.001
            alphas=sol.x[svs]
            # print(svs)
            w = np.sum(alphas*training_labels[svs]*training_feats[svs,:].T,axis = 1)[:,np.newaxis]
            b = np.mean(training_labels[svs]-np.dot(training_feats[svs,:],w))
         
            for i,we in enumerate(w):
                weights[i+1]=w[i]
            weights[0]=b
            self.prediction_weights=weights
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
        if(self.mode=="primal"):
            weights_t=np.transpose(self.prediction_weights)
            for i, sample in enumerate(feats):
                pred= self.prediction_weights[1:,np.newaxis].T.dot(feats[i])+self.prediction_weights[0]
                if(pred>0):
                    preds.append(1)
                else:
                    preds.append(-1)
        elif(self.mode=="dual"):
            weights_t=np.transpose(self.prediction_weights)
            for i, sample in enumerate(feats):
                pred= self.prediction_weights[1:,np.newaxis].T.dot(feats[i])+self.prediction_weights[0]
                if(pred>0):
                    preds.append(1)
                else:
                    preds.append(-1)
        elif(self.mode=="kernel"):
            def gaussian_kernel(x, y):
                return np.exp(-np.sum(np.square(x - y)) / self.lam)
            kernel = lambda x, y: gaussian_kernel(x, y)
            for i, samp in enumerate(feats):
                kernel_eval = np.array([kernel(samp, x_m) for x_m, a_m in zip(self.train_dat, self.prediction_weights[1:,])])
                print(kernel_eval)
                pred= self.prediction_weights[1:,np.newaxis].T.dot(kernel_eval)+self.prediction_weights[0]
                print(pred)
                if(pred>=0):
                    preds.append(1)
                else:
                    preds.append(-1)

        return preds
