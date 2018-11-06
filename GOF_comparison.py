import pyodbc
import pandas as pd
import numpy as np
from time import time
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pylab as plt
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.neighbors import NearestCentroid
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# function to fit and predict using the specified classifier
def Fit(modelname, name, docs_train, docs_test, y_train, y_test):
    t0 = time()
    model = modelname
    # scale for MLP as it is sensitive to feature scaling
    if (name == "Multi-layer Perceptron"): 
        scaler = StandardScaler(with_mean=False)  
        scaler.fit(docs_train)  
        docs_train = scaler.transform(docs_train)  
        docs_test = scaler.transform(docs_test)  
    # model fit    
    model.fit(docs_train, y_train) 
    # predict test set
    y_pred = model.predict(docs_test) 
    # get run time
    runtime = time() - t0
    # predict training set
    y_train_pred = model.predict(docs_train)
    # accuracy on test set
    score = accuracy_score(y_test, y_pred)
    # accuracy on training set
    trainScore = accuracy_score(y_train, y_train_pred)
    print ("ModelName:", name, "TestScore:", score, "TrainScore:", trainScore, "RunTime:", runtime)
    return [model, name, score, trainScore, runtime]

sopdb = pyodbc.connect("Driver={SQL Server Native Client 11.0};"
                    "Server=WSTLG01;"
                    "Database=LeadGen;"
                    "Trusted_Connection=yes;")

sql = '''SELECT [office]
            ,CASE WHEN [ldc_appointingcompanyidName] IS NULL THEN '' ELSE [ldc_appointingcompanyidName] END as ldc_appointingcompanyidName
            ,CASE WHEN [ldc_invoicetoorganisationidName] IS NULL THEN '' ELSE [ldc_invoicetoorganisationidName] END as ldc_invoicetoorganisationidName
            ,CASE WHEN [ldc_introducedbyorganisationidName] IS NULL THEN '' ELSE [ldc_introducedbyorganisationidName] END as ldc_introducedbyorganisationidName
            ,DATEDIFF(day, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) as duration
            ,CONVERT(int, [Ldc_AppointmentFees] / [ExchangeRate]) as fee
            ,CASE WHEN [Ldc_Description] IS NULL THEN '' ELSE [Ldc_Description] END as Ldc_Description
            ,[Value]
        FROM [LeadGen].[dbo].[Imported_SOP_Appointments]
        WHERE CONVERT(int, [Ldc_AppointmentFees] / [ExchangeRate]) >= 200'''

dt = pd.read_sql(sql, sopdb)
dt = dt.dropna()

h, w = dt.shape

txt_array = ['' for i in range(h)]
cols = ['ldc_appointingcompanyidName', 'ldc_invoicetoorganisationidName', 'ldc_introducedbyorganisationidName', 'Value', 'Ldc_Description']
for i in range(h):
    for col in cols:
        txt_array[i] = txt_array[i] + " " + dt.iloc[i][col]

txt_array = np.array(txt_array)

# tfidf it
vectorizer = CountVectorizer(stop_words='english', max_features = 1500, ngram_range=(1,3))
corpus_vec = vectorizer.fit_transform(txt_array)

print(corpus_vec.shape)

# for col in cols:
#     i = 0
#     odict = {}
#     for sf in set(dt[col]):
#         odict[sf] = i
#         i = i + 1
#     dt[col] = dt[col].map(odict)
#     dt[col] = pd.to_numeric(dt[col], errors='coerce')

dt['fee'] = pd.to_numeric(dt['fee'], errors='coerce').astype(int) // 20 + 1
dt['duration'] = pd.to_numeric(dt['duration'], errors='coerce')

cold = np.array([dt['duration']])
cvec = corpus_vec.toarray()
X = np.concatenate((cvec, cold.T), axis=1)
y = dt['fee']

print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# list of classifier names
names = ["Logistic Regression", "Bernoulli NB", "Ridge Classifier", 
         "Decision Tree", "Random Forest", "Multi-layer Perceptron"]
# list of classifiers
classifiers = [
    LogisticRegression(),
    BernoulliNB(),
    RidgeClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    MLPClassifier()
    ]
clsfrs = zip(names, classifiers)

r =  [[0 for x in range(5)] for y in range(len(names))]
j = 0
# run fit and predict on all specified classifiers and store results in a temporary array
for n, c in clsfrs: 
    model, name, score, trainScore, runtime = Fit(c, n, X_train, X_test, y_train, y_test)
    r[j] = [model, name, score, trainScore, runtime]
    j = j + 1

# order by test score in descending order with the best performing classifier on top
scores = np.array(r)[:, 2]
ri = np.argsort(scores)[::-1] 
# write the ordered result to file
with open ('C:\\Users\\AdmYL\\Desktop\\PriceOptimizer\\model_comparison.txt', 'w', encoding='utf-8') as m: 
    for i in range(ri.size):
        m.write("ModelName: {:s}  TestScore: {:f}  TrainScore: {:f}  RunTime: {:.2f}s \n\r\n\r".format(r[ri[i]][1], r[ri[i]][2], r[ri[i]][3], r[ri[i]][4]))