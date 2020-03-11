import networkx as nx
from gensim.models import Word2Vec, keyedvectors
from node2vec import Node2Vec
import pandas as pd
from sklearn import tree, preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
import  matplotlib.pyplot as plt
import numpy as np
import re


def convert_cate_num_freq(df):
    '''
    the value of the one hot encode equals to the Frequency levels.
    '''
    ohe = pd.DataFrame(index=df.index)
    for col in df:
        dummies = pd.get_dummies(df[col], prefix=col)
        ohe = pd.concat([ohe, dummies.div(dummies.shape[1])], axis=1)
    return ohe


def normalization(df):
    '''
    for the numeric features : normalize all values [ 0 , 1]
    '''
    # print(df.drop(['Unnamed: 0'],axis=1).transpose())
    min_max_scaler = preprocessing.MinMaxScaler()
    df_normalized = pd.DataFrame(min_max_scaler.fit_transform(df))

    # add columns and index to the normalizaed table
    df_normalized.columns = df.columns
    df_normalized.index = df.index
    return df_normalized


def get_numeric_df(df):
    '''
    For the numerical features, we normalized them .
    For the categorical features, we convert them to the numerical features and assigne the frequency value to each feature.
    '''
    # select the numerical features
    df_continuous_normalized = df.select_dtypes(include=['number'])

    if not df_continuous_normalized.empty: df_continuous_normalized = normalization(df_continuous_normalized)

    # select the categorical features, be aware of the data types in your original data types.
    df_categorical_hotencoded = df.select_dtypes(include=['object', 'category'])  # customerized

    # call the convert_cate_num_freq function for convert the categorical features into the numerical features and get the frequence values
    if not df_categorical_hotencoded.empty: df_categorical_hotencoded = convert_cate_num_freq(df_categorical_hotencoded)

    return pd.concat([df_continuous_normalized, df_categorical_hotencoded], axis=1)

def correlation(col1, col2):
    return None


dataFolder='C:\\Users\\xg16137\\OneDrive - APG\\My Documents\\Data\\'
df = pd.read_csv(dataFolder+'dataForModelling.csv')

del df['Unnamed: 0']
del df['KLTID']
del df['CD_PLAATS']
df = df[df['mostVisitTopic'].notna()]
df['NETTOPENSION_IND'] = df['NETTOPENSION_IND'] == 'Y'
df['OMS_GESLACHT'] = df['OMS_GESLACHT'] == 'Man'
df['NEWSLETTER'] = df['NEWSLETTER'] == 'Y'
df['PARTNER_AT_ABP'] = df['PARTNER_AT_ABP'] == 'Y'


# print(df['mostVisitTopic'].value_counts())

X = df.drop('mostVisitTopic', axis=1)
y = df['mostVisitTopic']
X = get_numeric_df(X)
# print(X.info())

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
# clf = tree.DecisionTreeClassifier(random_state=42)
# cross_val_score(clf, X, y, cv=10)
# clf = clf.fit(X_train, y_train)
# tree.plot_tree(clf)

# df_cat = df.drop('mostVisitTopic', axis=1).select_dtypes(include=['object', 'category'])
df_num = df.drop('mostVisitTopic', axis=1).select_dtypes(include=['number'])
# scores = []
# for cat in df_cat:
#     temp = []
#     for num in df_num:
#         clf = tree.DecisionTreeClassifier(random_state=42)
#         temp.append(sum(cross_val_score(clf, df[[num]], df[[cat]], cv=10))/10)
#     scores.append(temp)
# print(df_num.columns)
# for s in scores:
#     print(s)
# print(df_cat.columns)
# clf = tree.DecisionTreeClassifier(random_state=42).fit(df[['INCOME_PARTTIME_LATEST_JOB']],df[['SECTOR_MMS_LATEST_JOB']])
clf = tree.DecisionTreeClassifier(random_state=42, min_samples_split=40).fit(df_num,df['SECTOR_MMS_LATEST_JOB'])
# print(df_num.head())

# plt.figure()
# o = tree.plot_tree(clf, max_depth=10)
# print(o)
# plt.show()

treePathTxt="C:\\Users\\xg16137\\PycharmProjects\\TreeGraphEmbedding2\\data\\tree.txt"
treePathDot="C:\\Users\\xg16137\\PycharmProjects\\TreeGraphEmbedding2\\data\\tree.dot"
dotfile = open(treePathDot, 'w')
tree.export_graphviz(clf, out_file = dotfile)
dotfile.close()

file = open(treePathDot, 'r')#READING DOT FILE
content = file.readlines()

pattern ='->'
indicesArrow = [i for i, x in enumerate(content) if re.search(pattern, x)]
edges=list()
for i in indicesArrow:
    tempEdge= re.findall(r"[-+]?\d*\.\d+|\d+", content[i])
    edges.append(np.asarray(tempEdge))

G=nx.Graph()
for i, idx in enumerate(edges):
     G.add_edge((edges[i][0]).astype(int),(edges[i][1]).astype(int))
     G = G.to_undirected()
print(G.edges)

node2vec= Node2Vec(G, dimensions=64, walk_length=6, num_walks=60, p=1,q=1)
# Learn embedding
model = node2vec.fit(window=10, min_count=1,batch_words=4)
embeddings_dict={}
for node in G.nodes:
    embeddings_dict[node]=model.wv.get_vector(node)

embeddings_df = pd.DataFrame(embeddings_dict)

embeddings_df.to_csv (r'C:\Users\xg16137\PycharmProjects\TreeGraphEmbedding2\data\tree_embedding.csv' ,sep=',', encoding='utf-8' )


