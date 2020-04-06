from sklearn import datasets
import pandas as pd
import numpy as np
 #loading data 
dataFolder='C:\\dev\\data\\interestProfile\\'
province=pd.read_csv(dataFolder+'provincie.csv',header=0,sep=';',encoding= 'ISO-8859-1')
dataForModelling18062018=pd.read_csv(dataFolder+'dataForModelling18062018.csv',header=0,encoding= 'ISO-8859-1')
df=dataForModelling18062018
#%%
#only select the active people and subscript the newsletter
df=df.loc[(df['CD_STATUS_MCD']=='ACTIEF') & (df['NEWSLETTER']=='Y')]
df.drop(['CD_STATUS_MCD','NEWSLETTER'],inplace=True,axis=1)
#wdat.dtypes
#add provience to the wdat 
province.drop(['pc2','PLAATS','GEMEENTE'],axis=1, inplace=True)
df = pd.merge(df, province,  left_on='CD_PLAATS', right_on='PC', how='left')
df.iloc[1,:]
df.drop(['CD_PLAATS','PC'],inplace=True,axis=1)

#check all na values in each column
df.isna().sum()

#count the na in pronvince feature
df.PROVINCIE.fillna('Onbekend',inplace=True)
df['PROVINCIE'].value_counts()

df.dtypes
#remove the space of strings
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

#remove rarely levels from cateogorical variables
df['SECTOR_MMS_BIGGEST_JOB'].value_counts(dropna=False)
df=df[(df['SECTOR_MMS_BIGGEST_JOB']!='Energie') &(df['SECTOR_MMS_BIGGEST_JOB']!='Onbekend')& (df['SECTOR_MMS_BIGGEST_JOB']!='NONE')]


df['TYPE_DLN_BIGGEST_JOB'].value_counts(dropna=False)
df=df[(df['TYPE_DLN_BIGGEST_JOB']!='FPU' ) ]

df['OMS_SMLVRM'].value_counts(dropna=False)
df=df[(df['OMS_SMLVRM'] !='Onbekend' )]

df['SECTOR_MMS_LATEST_JOB'].value_counts(dropna=False)
df=df[df['SECTOR_MMS_LATEST_JOB']!='NONE' ]
#%%
#splite the df into clicked and non_clicked groups
df['Target'].value_counts(dropna=False)
df_clicked=df[df.Target.notnull()]
df_non_clicked=df[df.Target.isnull()]

#get the dummies of Target column
target_enc= pd.get_dummies(df_clicked['Target'])
df_clicked=df_clicked.join(target_enc)

#splite the 
#import random
#random.seed(123)
df_clicked.drop('Target',inplace=True,axis=1)
n=100000
df_ml=df_clicked.sample(n,replace=False,random_state=123)
#%%
from sklearn.model_selection import train_test_split
from sklearn.datasets import *
from sklearn.tree import DecisionTreeClassifier
# import labelencoder
from sklearn.preprocessing import LabelEncoder, StandardScaler# instantiate labelencoder object
from sklearn.ensemble import RandomTreesEmbedding, RandomForestClassifier


X = df_ml.iloc[:,0:17]
y = df_ml.iloc[:,21]

#label encoding.
#Categorical boolean mask
categorical_feature_mask = X.dtypes==object# filter categorical columns using mask and turn it into a list
categorical_cols = X.columns[categorical_feature_mask].tolist()

le = LabelEncoder()

# apply le on categorical feature columns
X[categorical_cols] = X[categorical_cols].apply(lambda col: le.fit_transform(col))

X = pd.get_dummies(X, prefix_sep='_', drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0 )

scl = StandardScaler()

X_train = scl.fit_transform(X_train)
X_test = scl.transform(X_test)


estimator_list = []

estimators = RandomForestClassifier(n_estimators=50,max_depth=5, random_state=0)
estimators.fit(X_train,y_train)
print(y_test.value_counts()/y_test.shape[0])
#%%build a graph of each decision tree
import networkx as nx
import matplotlib.pyplot as plt
from node2vec import Node2Vec

def create_graph_list(estimators_trees):
    G_list = []
    
    G=nx.Graph()
    for estimator in estimators_trees.estimators_ :
        n_nodes = estimator.tree_.node_count
        children_left = estimator.tree_.children_left
        children_right = estimator.tree_.children_right
        feature = estimator.tree_.feature
        threshold = estimator.tree_.threshold
        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, -1,0)]  # seed is the root node id and its parent depth

        while len(stack) > 0:
            node_id, parent_depth,parent_id = stack.pop()
            node_depth[node_id] = parent_depth + 1
            G.add_node(str(node_id))
            G.add_edge(str(parent_id),str(node_id))
            # If we have a test node
            if (children_left[node_id] != children_right[node_id]):
        
                stack.append((children_left[node_id], parent_depth + 1,node_id))
                stack.append((children_right[node_id], parent_depth + 1,node_id))
            else:
                is_leaves[node_id] = True
        #append created graph        
        G_list.append(G)
        #return multi graph
    return G_list
#%%Build the node2Vec model for all decision trees of the RF

G_list = create_graph_list(estimators)
#nx.draw_networkx(G_list[0])
'''for G in G_list:#print all graphs regarding the number of n_estimators
    nx.draw(G,with_labels = True)
    plt.show()'''

#build the node2vec node embedding for the tree structure graph 
node2vec_list = []
for G in G_list:
    node2vec = Node2Vec(G, dimensions=3, walk_length=3, num_walks=10)
    node2vec_list.append(node2vec)

## Learn embeddings of each decision tree
models_list = []

for node2v in node2vec_list:
    model = node2v.fit(window=10, min_count=1)
    models_list.append(model)
    
#test the embeddings of node 1 in the  different  decision tree 
#different node graph/decision tree has different number of nodes
#the embedding of each node in the different node2Vec is different
model_tree0=models_list[0]['10']
model_tree1=models_list[1]['10']

#%%
# 50 decision trees (estimators) constructures 50 correponding node graph, 
# after we built the decision trees, we put a sample/data point into the decision tree
# we can calculate the exit nodes of each point/sample of a tree structure.

# Eg. if we have 22621 points in the training
# for each piont in each decision tree will have the exit node regarding the tree structure.
#Therefore, within a decision tree/ one out of 50 decision trees, we will have node path vector= a levae_id=[22621*1]
#when we know the exit node, we can calculate the embeddings of this node under this tree structure/this decision tree.

reps = []# store all embedings from 50 different decision trees/node graphs of all training data points
#a row represnts each data point,  two sequential columns represent tan embedding from one node graph/decision tree 
reps_test = []

estimator= estimators.estimators_[0]
model=models_list[0]

for estimator,model in zip(estimators.estimators_,models_list):

    leave_id = estimator.apply(X_train)# the exit node number of each data/sample in the training dataset

    vect_rep = []

    for i in leave_id:
        vect_rep.append(model[str(i)])# calculate the embedding of each leave_id (the embeddings 
        #of each training data point)
    
    leave_id_test = estimator.apply(X_test)

    vect_rep_test = []

    for i in leave_id_test:
        vect_rep_test.append(model[str(i)])
    
    if reps == []:
        reps.append(vect_rep)
        reps_test.append(vect_rep_test)
    else:
        reps= np.array(reps).squeeze()
        reps_test = np.array(reps_test).squeeze()
        reps = np.hstack((reps,np.array(vect_rep))) 
        reps_test = np.hstack((reps_test,np.array(vect_rep_test)))
        print(np.shape(reps))
#%%
data = np.array(reps)
data_test = np.array(reps_test)

X_train2= np.hstack((X_train,data))
X_test2= np.hstack((X_test,data_test))

from sklearn.decomposition import PCA
#find the optimal components number
pca = PCA()
pca.fit(X_train2)
print(pca.explained_variance_ratio_)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
#choose the number of components by the percentage of the variance
#pcaw=PCA(0.4).fit(X_train2)#explained 90% of the variance =13 , 80%=11
#pcaw.n_components_
#%%
'''
pca = PCA(n_components=18)
pca.fit(X_train2)
print(pca.explained_variance_ratio_)
data_p = pca.transform(X_train2)
data_t = pca.transform(X_test2)

pcs = ['pc' + str(i+1) for i in range(pca.n_components_)]

df_train = pd.DataFrame(data_p,columns = pcs)
df_train['target'] = y_train.values

df_train['target'].value_counts(dropna=False)

df_test = pd.DataFrame(data_t,columns = pcs)
df_test['target'] = y_test.values


import seaborn as sns

sns.pairplot(df_test, hue='target')
'''

#plt.plot(data_p[:,0],data_p[:,1],'*b')
#plt.plot(data_t[:,0],data_t[:,1],'*r')
#%%
def performance(y_test,prediction_t):
    print(confusion_matrix(y_test,prediction_t))
    print(classification_report(y_test,prediction_t))
    print(accuracy_score(y_test, prediction_t))
   # print(y_test.value_counts()/y_test.shape[0])  

#%%
#X_train : original 
#data: embeddings 
#X_train2: concatnate = original + embeddings'''

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score

def ml_algorithms(data,y_train,data_test,y_test):
    rlts=[]
    print('-----KNN-----------')
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(data, y_train)
    prediction_t = neigh.predict(data_test)
    knn_acc=accuracy_score(y_test, prediction_t)
    performance(y_test,prediction_t)
    rlts.append(knn_acc)
    print('-----DecisionTreeClassifier-----------')   
    dec = DecisionTreeClassifier(max_depth=10, random_state=0)
    dec.fit(data, y_train) 
    prediction_t_ds = dec.predict(data_test)
    dec_acc=accuracy_score(y_test, prediction_t_ds)
    performance(y_test,prediction_t_ds)
    rlts.append(dec_acc)    
    print('-----RF-----------')
    clf = RandomForestClassifier(n_estimators=200, max_depth= None, random_state=0)
    clf.fit(data, y_train)
    prediction_t_rf = clf.predict(data_test)
    rf_acc=accuracy_score(y_test, prediction_t)
    performance(y_test,prediction_t_rf)
    rlts.append(rf_acc)
    print('-----LG+Linear-----------')
    clf = LogisticRegression()
    clf.fit(data, y_train)
    prediction_t_lg = clf.predict(data_test)
    prediction_tp= clf.predict_proba(data_test)
    lg_acc=accuracy_score(y_test, prediction_t_lg)
    performance(y_test,prediction_t_lg)
    
    rlts.append(lg_acc)
    print('-----LG+Ploynomial-----------')
    plf = PolynomialFeatures(2)
    datap = plf.fit_transform(data)
    datap_test = plf.transform(data_test)
    clfp = LogisticRegression()
    clfp.fit(datap, y_train)
    prediction_tpol = clfp.predict(datap_test)
    lg_ploy_acc=accuracy_score(y_test, prediction_tpol)
    #
    prediction_tpol_prob = clfp.predict_proba(datap_test)
    performance(y_test,prediction_tpol)
    rlts.append(lg_ploy_acc)
    print(rlts)
    return prediction_tp,prediction_tpol_prob

#%%
#X_train : original , 
prediction_tn,non_need=ml_algorithms(X_train,y_train,X_test,y_test)
print(y_test.value_counts()/y_test.shape[0])

#data: embeddings 

#ml_algorithms(data,y_train,data_test,y_test)
#X_train2: concatnate = original + embeddings'''
prediction_tp,prediction_tpol_prob=ml_algorithms(X_train2,y_train,X_test2,y_test)
#%%
from sklearn.metrics import roc_curve, auc

fpr_pol, tpr_pol,_ = roc_curve((y_test==True).apply(int),prediction_tpol_prob[:,1])

fpr, tpr,_ = roc_curve((y_test==True).apply(int),prediction_tp[:,1])
fprn, tprn,_ = roc_curve((y_test==True).apply(int),prediction_tn[:,1])

print('AUC for Node2Vec Logistic + Poly features + Normal Features : ', auc(fpr_pol, tpr_pol))
print('AUC for Node2Vec Logistic + Linear Features + Normal Features : ', auc(fpr, tpr))
print('AUC for Normal Features LogisticNormal Features Logistic : ', auc(fprn, tprn))

plt.plot(fpr_pol, tpr_pol, 'g', label='Node2Vec Logistic + Poly features + Normal Features')
plt.plot(fpr, tpr, 'r', label='Node2Vec Logistic + Linear Features + Normal Features')
plt.plot(fprn, tprn,'b', label='Normal Features Logistic')
plt.legend()
#%%
