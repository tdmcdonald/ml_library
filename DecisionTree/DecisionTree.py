import numpy as np 
import copy
import math
import statistics

class Node:
    def __init__(self):
        self.label=None
        self.attr = None
        self.value= None
        self.next = []
   
class DecisionTree:
    def __init__(self, criterion, max_depth):
        self.criterion=criterion
        self.max_depth= max_depth
        self.features = {} #key is column, value is set of values for that feature
        self.labels=set()
        self.root_node= None
        self.attributes= None

    def bestSplit(self, A, labels): #A will have the same dims as self.features
        # print('in best split a Dims are ',A.shape,'labels',labels)
        if self.criterion=='information_gain':
            print('info gain')
            label_counts={}
            for label in labels: #this increments the lable counts to find global entropy
                if label not in label_counts:
                    label_counts[label]= 0
                label_counts[label]+=1
            
            current_entropy= 0.0
            tot=len(labels)
            for label in label_counts: #calculate global entropy
                current_entropy+= -(label_counts[label]/tot)*math.log2(label_counts[label]/tot)
                # print('for label ', label, 'count is ', label_counts[label],'ent',current_entropy)
            # print('current entropy is ',current_entropy,'len labels',tot)
            feature_counts={}
            feature_outcome_counts={}
            for k,l in enumerate(A): #for every attribute
                # print(k)
                if k not in feature_counts:
                    feature_counts[k]={}
                    feature_outcome_counts[k]={}

                for v in A[k]:      #for every sample in attribute accumulate total outcomes
                    if v not in feature_outcome_counts[k]:
                        # feature_counts[k][v]={}
                        feature_outcome_counts[k][v]=0
                    feature_outcome_counts[k][v]+=1

                for value in A[k]: #set value to 0
                    if value not in feature_counts[k]:
                        feature_counts[k][value]={}
                    for l in self.labels:
                        if l not in feature_counts[k][value]:
                            feature_counts[k][value][l]=0



            # print('feat outcome counts ', feature_outcome_counts)
            # print('feat  counts ', feature_counts)

            for attribute,l in enumerate(A): #loop through columns to determine information gain for each attr
                for i,sample in enumerate(A[attribute]): #loop through samples to accumulate values
                    # print(attribute, sample)
                    for label in self.labels:
                        # print('label is ', labels[i],'compared against ',label)
                        if labels[i]==label:
                            # print(attribute,sample,label)
                            feature_counts[attribute][sample][label]+=1
                            
            # print('yoyo',feature_counts)
            feature_entropy={}  
            for attribute,l in enumerate(feature_counts): #loop through columns to determine information gain for each attr
                if attribute not in feature_entropy:
                    feature_entropy[attribute]={}
                for value in feature_counts[attribute]: #loop through samples to accumulate values
                    for label in self.labels:
                        if value not in feature_entropy[attribute]:
                            feature_entropy[attribute][value]=0
                        # print('attr is ', attribute,'val  is ', value,'label is', label,'cout is ', feature_counts[attribute][value][label],'outcome count ',feature_outcome_counts[attribute][value],'ent',feature_entropy[attribute][value])
                        
                        if feature_counts[attribute][value][label] !=0 and feature_outcome_counts[attribute][value]!=0:
                            feature_entropy[attribute][value] += -(feature_counts[attribute][value][label]/feature_outcome_counts[attribute][value])*math.log2(feature_counts[attribute][value][label]/feature_outcome_counts[attribute][value])
           

           #Feat outcome count is indexed by value
            expected_entropy={}
            for attr in feature_entropy:
                if attr not in expected_entropy:
                    expected_entropy[attr]=0
                for ent in feature_entropy[attr]:
                    # print('feature entropy is ', feature_entropy[attr][ent],feature_outcome_counts[attr][ent])
                    expected_entropy[attr]+=feature_entropy[attr][ent]/feature_outcome_counts[attr][ent]
            ig={}
            max_ig=0
            max_ig_attr=None
            for attribute in feature_entropy:
                ig[attribute]=current_entropy-expected_entropy[attribute]
                # print('for attribute ',attribute,'ig is ', ig[attribute], 'child entr', feature_entropy[attribute])
                if ig[attribute]>max_ig:
                    max_ig=ig[attribute]
                    max_ig_attr= attribute
            print('max ig is ', max_ig, 'attr', max_ig_attr)
            return max_ig_attr
                   
                    
                    
        # else if self.criterion == 'majority_error':

        # else if self.criterion == 'gini':
        else:
            return "whoops"


    def id3(self, S, attributes, labels):
        print('id3')
        first=labels[0]
        switch=0
        for i,l in enumerate(labels):
            if labels[i]!=first:
                switch=1
        if switch==0:
            print('all labels are the same and ')
            if len(attributes)==0:
                print('len of attr is 0')
                leaf= Node()
                leaf.label=  statistics.mode(labels)
                return leaf
            else:
                print('most common attr is ', first)
                leaf= Node()
                leaf.label=first
                return leaf
        else:
            root= Node()
            a = self.bestSplit(S, labels)
            best_attr=attributes[a]
            root.attr=best_attr
            print('starting id3 main statement size of attr is ',len(attributes), 'a is ', a, 'best attr is ',best_attr) 
            for v in self.features[a]: # want sv to be S where A=v v will be all values a can have & create branch NOTE: s.features isa  set
                count=0
                mask = np.ones(len(S[a]), dtype= bool)
                Sv = copy.deepcopy(S)
                temp_labels=copy.deepcopy(labels)
                temp_attributes=copy.deepcopy(attributes)
                for i,x in enumerate(S[a]):
                    # print('i is ', i, 'S[a][i] is ',S[a][i],'v is ',v)
                    if S[a][i] != v:
                        mask[i]=False #want to append all features that have this value for this attr. add feat col
                    else:
                        count+=1
                print('v is ', v, 'count is ', count)
                # print(Sv.shape)
                # print('mask shape',mask.shape)
                Sv=Sv[:,mask]
                # print('after mask sv ',Sv.shape)
                Sv=np.delete(Sv, a, axis=0)
                # print('label shape',labels.shape)
                temp_labels=labels[mask]
                # print(labels.shape)
                if(count==0):
                    leaf= Node()
                    leaf.label=  statistics.mode(labels) 
                    leaf.value= v
                    root.next.append(leaf) #create leaf node
                    
                else:
                    temp_attributes=np.delete(attributes, a)
                    print('APPENDING NEXT TO ATTRIBUTE' , best_attr)
                    nxt= self.id3(Sv,temp_attributes,temp_labels)
                    nxt.value=v
                    root.next.append(nxt)
                #run id3 now
                    
            return root
            
            

    def print(self, root, indent=""):
        print(indent,'node attribute ', root.attr,' label ', root.label)
        if(root == None):
            print(indent,'is a leaf node')
            return
        for r in root.next:
            print(indent, 'has next node ', )
            return self.print(r, indent +"      ")

    #features as features x samples 
    #labels correspond to line in features
    def fit(self, training_features, training_labels, criterion = 'IG',max_depth =None ):
        print('fit')
        # first looop through features to define self.features
        for i,feature in enumerate(training_features):
            for sample in training_features[i]:
                if i not in self.features:
                    self.features[i]=set()
                self.features[i].add(sample)
        print(self.features)

        for label in training_labels:
            self.labels.add(label)

        print(self.labels)
        self.attributes= ['a','b','c','d','e','f'] # attributes contains index to each attr
        self.root = self.id3(training_features, self.attributes, training_labels)
        self.print(self.root)
        return self



    def predict(self, features):
        print('predict')






    
    
