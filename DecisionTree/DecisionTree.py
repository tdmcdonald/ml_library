import numpy as np 
import copy
import math
import statistics

class Node:
    def __init__(self):#roots point to value nodes, value nodes point to 
        self.label=None #if leaf node, this is the label
        self.attr = None #if root node, this is the attribute
        self.value= None #if value node this is the value
        self.next = []
        self.type=None
   
class DecisionTree:
    def __init__(self, criterion, max_depth):
        self.criterion=criterion
        self.max_depth= max_depth
        self.features = {} #key is column, value is set of values for that feature
        self.labels=set()
        self.root_node= None
        self.attributes= None
        self.current_depth=-1
        self.numerical_cols= None
        self.medians=None
        self.unknown_cols=None
        self.modes=None

    def bestSplit(self, A, labels): #A will have the same dims as self.features
        # print('in best split a Dims are ',A.shape,'labels',labels)
        if self.criterion=='entropy':
            # print('entropy')
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
                for val in feature_entropy[attr]:
                    # print('feature entropy is ', feature_entropy[attr][ent],feature_outcome_counts[attr][ent])
                    expected_entropy[attr]+=feature_entropy[attr][val]*(feature_outcome_counts[attr][val]/(tot))
            ig={}
            max_ig=0
            max_ig_attr=None
            for attribute in feature_entropy:
                ig[attribute]=current_entropy-expected_entropy[attribute]
                # print('for attribute ',attribute,'ig is ', ig[attribute], 'child entr', feature_entropy[attribute])
                if ig[attribute]>max_ig:
                    max_ig=ig[attribute]
                    max_ig_attr= attribute
                if(max_ig_attr==None):
                    max_ig_attr=attribute
            # print('max ig is ', max_ig, 'attr', max_ig_attr)
            return max_ig_attr
                   
                
        elif self.criterion == 'gini':
            label_counts={}
            for label in labels: #this increments the lable counts to find global entropy
                if label not in label_counts:
                    label_counts[label]= 0
                label_counts[label]+=1
            
            current_gini= 1.0
            tot=len(labels)
            for label in label_counts: #calculate global entropy
                current_gini-=(label_counts[label]/tot)**2
                # print('for label ', label, 'count is ', label_counts[label],'ent',current_gini)
            # print('current entropy is ',current_gini,'len labels',tot)
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
            feature_gini={}  
            for attribute,l in enumerate(feature_counts): #loop through columns to determine information gain for each attr
                if attribute not in feature_gini:
                    feature_gini[attribute]={}
                for value in feature_counts[attribute]: #loop through samples to accumulate values
                    for label in self.labels:
                        if value not in feature_gini[attribute]:
                            feature_gini[attribute][value]=1.0
                        # print('attr is ', attribute,'val  is ', value,'label is', label,'cout is ', feature_counts[attribute][value][label],'outcome count ',feature_outcome_counts[attribute][value],'ent',feature_gini[attribute][value])
                        feature_gini[attribute][value] -= (feature_counts[attribute][value][label]/feature_outcome_counts[attribute][value])**2
           

           #Feat outcome count is indexed by value
            expected_gini={}
            for attr in feature_gini:
                if attr not in expected_gini:
                    expected_gini[attr]=0
                for val in feature_gini[attr]:
                    # print('feature entropy is ', feature_gini[attr][ent],feature_outcome_counts[attr][ent])
                    expected_gini[attr]+=feature_gini[attr][val]*(feature_outcome_counts[attr][val]/(tot))
            ig={}
            max_ig=0
            max_ig_attr=None
            for attribute in feature_gini:
                ig[attribute]=current_gini-expected_gini[attribute]
                # print('for attribute ',attribute,'ig is ', ig[attribute], 'child entr', feature_gini[attribute])
                if ig[attribute]>max_ig:
                    max_ig=ig[attribute]
                    max_ig_attr= attribute
                if(max_ig_attr==None):
                    max_ig_attr=attribute
            # print('max ig is ', max_ig, 'attr', max_ig_attr)
            return max_ig_attr
#########################################################################
###############MAJORITY ERROR############################################
#########################################################################
        elif self.criterion == 'majority_error':
            label_counts={}
            label_sum=0
            for label in labels: #this increments the lable counts to find global entropy
                if label not in label_counts:
                    label_counts[label]= 0
                label_counts[label]+=1
                label_sum+=1
            
            # print(labels)
            # for attribute,l in enumerate(A): #loop through columns to determine information gain for each attr
            maj_label=max(label_counts, key=label_counts.get)
            # print(maj_label, label_counts)
            # print(majority_label)

            current_me= 1-label_counts[maj_label]/label_sum
            # print('CURRENT ME IS ',current_me)
            # ME=
            tot=len(labels)
            # for label in label_counts: #calculate global entropy
                # current_gini-=(label_counts[label]/tot)**2
                # print('for label ', label, 'count is ', label_counts[label],'ent',current_gini)
            # print('current entropy is ',current_gini,'len labels',tot)
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
            majority_errors={}  
            for attribute,l in enumerate(feature_counts): #loop through columns to determine information gain for each attr
                if attribute not in majority_errors:
                    majority_errors[attribute]={}
                for value in feature_counts[attribute]: #loop through samples to accumulate values
                        if value not in majority_errors[attribute]:
                            majority_errors[attribute][value]=0.0
                        max_label= max(feature_counts[attribute][value], key=feature_counts[attribute][value].get)
                        sum = 0
                        for label in self.labels:
                            sum+=feature_counts[attribute][value][label]
                        majority_errors[attribute][value]=1- feature_counts[attribute][value][max_label]/sum
                        # print(majority_errors[attribute][value], 'couunt is ',feature_counts[attribute][value][max_label], 'sum is ', sum)
            
           #Feat outcome count is indexed by value
            # print('mes',majority_errors)
            expected_me={}
            for attr in majority_errors:
                if attr not in expected_me:
                    expected_me[attr]=0
                for val in majority_errors[attr]:
                    # print('fraction is ',feature_outcome_counts[attr][val], tot)
                    expected_me[attr]+=majority_errors[attr][val]*(feature_outcome_counts[attr][val]/(tot))
            ig={}
            max_ig=0
            max_ig_attr=None
            # print(expected_me)
            for attribute in majority_errors:
                ig[attribute]=current_me-expected_me[attribute]
                # print('for attribute ',attribute,'ig is ', ig[attribute], 'child entr', majority_errors[attribute])
                if ig[attribute]>max_ig:
                    max_ig=ig[attribute]
                    max_ig_attr= attribute
            if(max_ig_attr==None):
                max_ig_attr=attribute
            # print('max ig is ', max_ig, 'attr', max_ig_attr)
            # print(ig['asdf']) 
            return max_ig_attr

        else:
            return "whoops"


    def id3(self, S, attributes, labels, depth=0):
        # print('id3')
        first=labels[0]
        switch=0
        for i,l in enumerate(labels):
            if labels[i]!=first:
                switch=1
        if switch==0:
            # print('all labels are the same and ')
            if len(attributes)==0:
                # print('len of attr is 0')
                leaf= Node()
                leaf.type='leaf'
                leaf.label= self.getMaxLabel(labels) 
                return leaf
            else:
                # print('most common attr is ', first)
                leaf= Node()
                leaf.type='leaf'
                leaf.label=first
                return leaf
        else:
            root= Node()
            root.type='root'
            depth=depth+1
            # print('MY DEPTH IS ',depth)
            a = self.bestSplit(S, labels)
            best_attr=attributes[a]
            root.attr=best_attr
            if depth ==self.max_depth:
                # print('in max depth', 'best attr', best_attr,'attributes are ', attributes, 'a is ',a)
                for v in self.features[best_attr]:
                    branch= Node()
                    branch.type = 'branch'
                    branch.value=v
                    root.next.append(branch)
                    leaf= Node()
                    leaf.type ='leaf'
                    leaf.label= self.getMaxLabel(labels) 
                    # print('setting label to ',leaf.label)
                    branch.next.append(leaf) #create leaf node
                return root
            # print('starting id3 main statement size of attr is ',len(attributes), 'a is ', a, 'best attr is ',best_attr) 
            # print('features at a are',self.features[best_attr])
            for v in self.features[best_attr]: # want sv to be S where A=v v will be all values a can have & create branch NOTE: s.features isa  set
                branch= Node()
                branch.type = 'branch'
                branch.value=v
                root.next.append(branch)
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
                # print('v is ', v, 'count is ', count)
                # print(Sv.shape)
                # print('mask shape',mask.shape)
                Sv=Sv[:,mask]
                # print('after mask sv ',Sv.shape)
                Sv=np.delete(Sv, a, axis=0)
                # print('label shape',labels.shape)
                temp_labels=labels[mask]
                # print(labels.shape)
                if(count==0):
                    # print('adding leaf for root attr ', best_attr)
                    leaf= Node()
                    leaf.type ='leaf'
                    leaf.label= self.getMaxLabel(labels) 
                    branch.next.append(leaf)
                else:
                    temp_attributes=np.delete(attributes, a)
                    # print('BEST ATTR INDEX IS ', a, 'and attre is ', best_attr)
                    # print('attributes before delete are', attributes)
                    # print('after delete are ', temp_attributes)
                    # print('APPENDING NEXT TO ATTRIBUTE' , best_attr)
                    nxt_root= self.id3(Sv,temp_attributes,temp_labels,depth)
                    # nxt_root.value=v
                    branch.next.append(nxt_root)
                #run id3 now
                    
            return root
            
            
    def getMaxLabel(self,labels):
        # print('HERERERERERERERER')
        counts={}
        for label in labels:
            if label not in counts:
                counts[label]=0
            counts[label]+=1
        m=max(counts, key=counts.get)

        # print('labels', labels,'max',m, 'counts',counts)
        return  m


    def print(self, root, indent=""):
        print(indent,'node type is  ', root.type,'len next is ', len(root.next))
        if root.type=='root':
            print(indent,'and ATTR is ',root.attr)
        if root.type=='branch':
            print(indent,'and VALUE is ',root.value)
        if root.type=='leaf':
            print(indent,'and LABEL is ',root.label)
        if(len(root.next) == 0):
            return
        for r in root.next:
            print(indent, 'has next node ',r.type)
            self.print(r, indent +"      ")

    #features as features x samples 
    #labels correspond to line in features
    def fit(self, training_feats, training_labels,  numerical_cols=[],unknown_cols=[] ):
        # print('fit',training_feats)
        # first looop through features to define self.features
        training_features=copy.copy(training_feats)
        self.numerical_cols=numerical_cols
        self.unknown_cols=unknown_cols
        # print('starting fit')
        if(len(self.numerical_cols)>0):    
            medians={}
            for i,v in enumerate(training_features):
                # print(training_features[i])
                if i in numerical_cols:
                    # print(training_features[i].dtype)
                    y=training_features[i].astype(np.float)
                    medians[i]=statistics.median(y)
                    # medians[i]=statistics.median(training_features[i])
                    # print('median is ', medians[i])
            for i,v in enumerate(training_features):
                if i in numerical_cols:
                    for j,a in enumerate(training_features[i]):
                        if(float(training_features[i][j])>=medians[i]):
                            training_features[i][j]='above'
                        else:
                            training_features[i][j]='below'
        # print(self.features)
            self.medians=medians
        if(len(self.unknown_cols)>0):    
            modes={}
            for i,v in enumerate(training_features):
                # print(training_features[i])
                if i in self.unknown_cols:
                    # print(training_features[i].dtype)
                    # print('before temp ', training_features[i])
                    temp=training_features[i][training_features[i]!='unknown']
                    # print('after temp',temp)
                    modes[i]=statistics.mode(training_features[i])
                    # medians[i]=statistics.median(training_features[i])
                    # print('median is ', medians[i])
            for i,v in enumerate(training_features):
                if i in self.unknown_cols:
                    for j,a in enumerate(training_features[i]):
                        if(training_features[i][j]=="unknown"):
                            training_features[i][j]=modes[i]

            self.modes=modes
        # print("i AM HERE",modes)
        for i,feature in enumerate(training_features):
            for sample in training_features[i]:
                if i not in self.features:
                    self.features[i]=set()
                self.features[i].add(sample)
        for label in training_labels:
            self.labels.add(label)
        # print('self labels', self.labels)
        # print(self.labels)
        self.attributes= np.arange(0,len(self.features)) # attributes contains index to each attr
        self.root = self.id3(training_features, self.attributes, training_labels)
        # self.print(self.root)
        return self



    def find_label(self, feature, root):
        if root.type=='root':
            # print('root and attr is ',root.attr)
            for r in root.next:
                if r.value==feature[root.attr]:
                    return self.find_label(feature, r)
        elif root.type=='branch':
            # print('branch and value is ',root.value,' root next is ', len(root.next))
            return self.find_label(feature, root.next[0])
        elif root.type=='leaf':
            # print('leaf and label is ',root.label)
            return root.label
        
            

    def predict(self, feats):
        # print('predict',features)
        results=[]
        features=copy.copy(feats)
        if(len(self.numerical_cols)>0):    
            for i,v in enumerate(features):
                # print(i,features[i])
                for j,a in enumerate(features[i]):
                    if j in self.numerical_cols:
                        # print(j)
                        if(float(features[i][j])>=self.medians[j]):
                            features[i][j]='above'
                        else:
                            features[i][j]='below'
        if(len(self.unknown_cols)>0):    
            for i,v in enumerate(features):
                for j,a in enumerate(features[i]):
                    if j in self.unknown_cols:
                        if(features[i][j]=="unknown"):
                            features[i][j]=self.modes[j]
        for feat in features:
            # print('feature is',feat)
            res= self.find_label(feat,self.root)
            # print('label is ',res)
            results.append(res)
            # attr=self.root.attr
            # print(attr, feat[attr])
        return results
            

# feature is ['low' 'vhigh' '4' '4' 'big' 'med']
# root and attr is  5
# branch and value is  med
# root and attr is  3
# branch and value is  4
# root and attr is  4
# label is  None




    
    
