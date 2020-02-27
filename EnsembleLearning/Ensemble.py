import numpy as np 
import copy
import math
import statistics
import DecisionTree


class AdaBoost:
    
    def __init__(self, iterations):
        self.iterations= iterations
        self.alphs=[]
        self.dts=[]
    def find_error(self, predictions, labels,weights ):
        error=0
        for i,p in enumerate(predictions):
            # print(weights[i],p,labels[i])
            error+=weights[i]*p*labels[i]  #TODO weights should be per sample not per feature
        error=.5-.5*error
        # print(error)
        return error

    def fit(self, training_feats, training_labels,  numerical_cols=[],unknown_cols=[] ):
        n = len(training_labels)
        pred_features=training_feats
        #training_feats=np.transpose(training_feats)
        indices=np.arange(0,n)
        
        icount=0
       
        # print(training_labels, icount, icount/len(training_labels))
        weights = np.zeros([len(training_labels)])
        weights.fill(1/n)
        self.alphas=[]
        self.dts=[]
        errors=[]
        iters=[]
        for i in np.arange(0,self.iterations):
            sampled=np.random.choice(indices,len(indices))
            # print('sampeld is ',sampled)
            new_feats=copy.deepcopy(training_feats)
            new_labels=copy.deepcopy(training_labels)
            # new_weights=copy.deepcopy(weights)
            # for i, ind in enumerate(sampled):
            #     new_feats[i]=training_feats[ind]
            #     new_labels[i]=training_labels[ind]
                # new_weights[i]=weights[ind]
            new_feats_t=np.transpose(new_feats)
            dti=DecisionTree.DecisionTree('entropy',1)
            # print(numerical_cols)
            z=np.sum(weights)
            # weights=weights/z
            dti.fit(new_feats_t,new_labels, numerical_cols, weights=weights)
            preds= dti.predict(new_feats)
            # weights=weights/z
            e = self.find_error(preds, new_labels,weights)
            # print('z is ', z, 'sum is ', np.sum(weights))
            e+=.000001
            # print('e is ',e)
            alpha= .5*math.log((1-e)/e)
            # print(weights)
            self.alphas.append(alpha)
            self.dts.append(dti)
            for j,w in enumerate(weights):
                # print('prev weihgt ', weights[i], -alpha, training_labels[i], preds[i],math.exp(-alpha*training_labels[i]*preds[i]))
                weights[j]=w*math.exp(-alpha*new_labels[j]*preds[j])

                # print('new weihgt ', weights[i])
            weights=weights/sum(weights)
            iters.append(i)
            errors.append(e)
        return errors, iters






    def predict(self, feats):
        # print('predicting')
        preds=[]
        # print(type(feats))
        features=np.asarray(copy.copy(feats),dtype=str)
        # print(type(features[0][0]))
        # print(features)
        for i, s in enumerate(features):
            sum=0
            # print(s)
            
            for j in np.arange(0,len(self.alphas)):
                # print(s)
                # print(features)
                asdf=self.dts[j].predict([s])
                sum+=self.alphas[j]*self.dts[j].predict([s])[0]
            # print(sum)
            if(sum<0):
                preds.append(-1)
            else:
                preds.append(1)
        return preds



class BaggedTrees:
    
    def __init__(self, iterations):
        self.iterations= iterations
        self.dts=[]
    
    def fit(self, training_feats, training_labels,  numerical_cols=[],unknown_cols=[] ):
        n = len(training_labels)
        pred_features=training_feats
        #training_feats=np.transpose(training_feats)
        indices=np.arange(0,n)
        # print('len indices is ', len(indices), len(training_labels), len(training_feats))
        icount=0
       
        # print(training_labels, icount, icount/len(training_labels))
     
        self.dts=[]
        for i in np.arange(0,self.iterations):
            idx = np.random.randint(0,len(training_feats),size=(len(training_feats)))
            sampled=training_feats[idx]
            # print('sampeld is ',sampled)
            # new_feats=copy.deepcopy(training_feats)
            # for i, ind in enumerate(sampled):
            #     new_feats[i]=new_feats[ind]
            new_feats=np.transpose(sampled)
            # print('start')
            dti=DecisionTree.DecisionTree('entropy',16)
            # print(numerical_cols)
            # print('mid')
            dti.fit(new_feats,training_labels)
            # print('done w tree')
            # preds= dti.predict(pred_features)

            self.dts.append(dti)
            
         



    def getMode(self,labels):
        counts={}
        for label in labels:
            if label not in counts:
                counts[label]=0
            counts[label]+=1
        m=max(counts, key=counts.get)
        return m


    def predict(self, feats):
        # print('predicting')
        preds=[]
        # print(type(feats))
        features=np.asarray(copy.copy(feats),dtype=str)
        # print(type(features[0][0]))
        # print(features)
        for i, s in enumerate(features):
            sum=0
            # print(s)
            res=[]
            for j,tree in enumerate(self.dts):
                # print(s)
                # print(features)
               res.append(self.dts[j].predict([s])[0])
            # print(sum)
            preds.append(self.getMode(res))
        return preds



class RandomForrest:
    
    def __init__(self, iterations, sample_num):
        self.iterations= iterations
        self.sample_num= sample_num
        self.dts=[]
    
    def fit(self, training_feats, training_labels,  numerical_cols=[],unknown_cols=[] ):
        n = len(training_labels)
        pred_features=training_feats
        #training_feats=np.transpose(training_feats)
        indices=np.arange(0,n)
        # print('len indices is ', len(indices), len(training_labels), len(training_feats))
        icount=0
       
        # print(training_labels, icount, icount/len(training_labels))
     
        self.dts=[]
        for i in np.arange(0,self.iterations):

            sampled=np.random.choice(indices,len(indices))
            # print('sampeld is ',sampled)
            new_feats=copy.deepcopy(training_feats)
            for i, ind in enumerate(sampled):
                new_feats[i]=new_feats[ind]
            new_feats=np.transpose(new_feats)
            dti=DecisionTree.DecisionTree('entropy', 4,random=True, sample_num=self.sample_num)
            # print(numerical_cols)
            dti.fit(new_feats,training_labels, numerical_cols)
            preds= dti.predict(pred_features)

            self.dts.append(dti)
            
         



    def getMode(self,labels):
        counts={}
        for label in labels:
            if label not in counts:
                counts[label]=0
            counts[label]+=1
        m=max(counts, key=counts.get)
        return m


    def predict(self, feats):
        # print('predicting')
        preds=[]
        # print(type(feats))
        features=np.asarray(copy.copy(feats),dtype=str)
        # print(type(features[0][0]))
        # print(features)
        for i, s in enumerate(features):
            res=[]
            for j,tree in enumerate(self.dts):
                # print(s)
                # print(features)
               res.append(self.dts[j].predict([s])[0])
            # print(sum)
            preds.append(self.getMode(res))
        return preds


