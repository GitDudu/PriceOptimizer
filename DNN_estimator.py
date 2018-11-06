import pyodbc, io, re, pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from time import time
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# regex matching NSP numbers
rgxNSP = r'NSP\D{0,2}\/{0,1}\d{2}\D{0,2}\/{0,1}\d{2}\D{0,2}\/{0,1}\d*'
# regex matching date formats
rgxDate = r'(?=(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\d{4}(?=\b)|\d{1,2}(?=st|nd|rd|th|\b)))\w{0,10}\W{0,10}(\d{1,2}(?=st|nd|rd|th|\b)|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\D{0,10}(\d{4}(?=\b)|\d{1,2}(?=st|nd|rd|th|\b))'

def replaceNoise(item):
    # using regex to remove noisy information
    item = item.lower()
    item = re.sub(rgxNSP, '', item)
    item = re.sub(rgxDate, '', item)
    item = re.sub('dated', '', item)
    item = re.sub('facilities', 'facility', item)
    return item

def numRound(n, base=5):
    return int(base * round(float(n)/base))

sopdb = pyodbc.connect("Driver={SQL Server Native Client 11.0};"
                    "Server=WSTCRMDB02;"
                    "Database=SOP_MSCRM;"
                    "Trusted_Connection=yes;")

sql = '''SELECT DISTINCT
            CASE WHEN Ldc_LeadSource IS NULL THEN 0 ELSE Ldc_LeadSource END as 'Ldc_LeadSource',
            CASE WHEN Ldc_Status IS NULL THEN 0 ELSE Ldc_Status END as 'Ldc_Status',
            CASE WHEN Ldc_Trustee IS NULL THEN 0 ELSE Ldc_Trustee END as 'Ldc_Trustee',
            CASE WHEN Ldc_FeeType IS NULL THEN 0 ELSE Ldc_FeeType END as 'Ldc_FeeType',
            CASE WHEN ga_PerpetualAppointment IS NULL THEN 0 ELSE ga_PerpetualAppointment END as 'ga_PerpetualAppointment',
            CASE WHEN [CustomerTypeCode] IS NULL THEN 0 ELSE [CustomerTypeCode] END as 'CustomerTypeCode',
            CASE WHEN c.OwnerIdName IS NULL THEN '' ELSE c.OwnerIdName END as 'OwnerIdName',
            CASE WHEN c.OwningBusinessUnit IS NULL THEN '' ELSE convert(nvarchar(100), c.OwningBusinessUnit) END as 'OwningBusinessUnit',
            CASE WHEN c.StateCode IS NULL THEN '' ELSE c.StateCode END as 'StateCode',
            CASE WHEN c.TransactionCurrencyId IS NULL THEN '' ELSE convert(nvarchar(100), c.TransactionCurrencyId) END as 'TransactionCurrencyId',
            CASE WHEN OpenRevenue_State IS NULL THEN '' ELSE OpenRevenue_State END as 'OpenRevenue_State',
            CASE WHEN [Ldc_CountryIncorporatedIn] IS NULL THEN 0 ELSE [Ldc_CountryIncorporatedIn] END as 'Ldc_CountryIncorporatedIn',
            CONVERT(int, [Ldc_AppointmentFees] / a.[ExchangeRate]) as 'fee',
            CASE WHEN [Ldc_AcceptanceDate] IS NOT NULL AND [Ldc_TerminationDate] IS NOT NULL AND DATEDIFF(day, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) > 0 
	            THEN DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) ELSE 0 END as 'duration',
            CASE WHEN [Ldc_Description] IS NULL THEN '' ELSE [Ldc_Description] END as 'Ldc_Description',
            CASE WHEN [Ldc_Description] IS NULL THEN 0 ELSE LEN([Ldc_Description]) END as 'Len_Ldc_Description'
        FROM [SOP_MSCRM].[dbo].[Ldc_appointment] a
        INNER JOIN [SOP_MSCRM].[dbo].[Account] c
        ON a.ldc_appointingcompanyid = c.AccountId AND 
            TRY_CONVERT(UNIQUEIDENTIFIER, a.ldc_appointingcompanyid) IS NOT NULL AND 
            TRY_CONVERT(UNIQUEIDENTIFIER, c.AccountId) IS NOT NULL
        WHERE Ldc_Trustee = 1 AND 
            CONVERT(int, [Ldc_AppointmentFees] / a.[ExchangeRate]) >= 300 AND 
            CONVERT(int, [Ldc_AppointmentFees] / a.[ExchangeRate]) <= 1500 AND 
            --DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) <= 10 AND 
            (Ldc_Description not like '%extension%' and Ldc_Description not like '%expansion%') AND
            [Ldc_AcceptanceDate] > '01 Oct 2016' '''

dt = pd.read_sql(sql, sopdb)
dt = dt.dropna()
dt['Ldc_Description'] = dt['Ldc_Description'].apply(replaceNoise)

tcols = ['OwnerIdName', 'OwningBusinessUnit', 'TransactionCurrencyId', 'Ldc_CountryIncorporatedIn']
for col in tcols:
    i = 0
    odict = {}
    for sf in set(dt[col]):
        odict[sf] = i
        i = i + 1
    # if '' in odict: 
    #     odict[''] = 999
    dt[col] = dt[col].map(odict)

desc_key_words = ['facility','isda','intercreditor','repurchase','purchase','agency_agreement',
    'loan_agreement','lease_agreement','aircraft_lease','deed','security','share',
    'series','swap','supplemental','indenture','underwriting_agreement','credit','guarantee','notes_due']
    # 'accession','trust_deed','issue_deed','senior_notes','new_notes', 'term_loan','global_master','credit_agreement','registration_statement',
    # 'shareholders','mortgage','aircraft_sale','novation','subscription_agreement','fee_letter','subordination'

def keyWordCount(s):
    if (re.match(r'^\d{1,2}\s', s)):
        return s.split(' ')[0]
    else:
        i = 0
        for k in desc_key_words:
            if (k in s):
                i = i + 1
        return i if i > 0 else 1

# dt['Ldc_Description'] = dt['Ldc_Description'].apply(keyWordCount)

cols = ['Ldc_LeadSource','Ldc_Status','Ldc_Trustee','Ldc_FeeType','ga_PerpetualAppointment','CustomerTypeCode','OwnerIdName','OwningBusinessUnit',
    'StateCode','TransactionCurrencyId','OpenRevenue_State','Ldc_CountryIncorporatedIn','duration']
for col in cols:
    dt[col] = pd.to_numeric(dt[col], errors='coerce')
dt['fee'] = pd.to_numeric(dt['fee'], errors='coerce').astype(int)

def word_count(s, w):
    return s.count(w)

for col in desc_key_words: 
    dt.loc[:, col] = pd.Series(dt['Ldc_Description'].apply(lambda x: word_count(x, col.replace('_', ' '))), index=dt.index)

dt.loc[:, 'sum_key_word'] = pd.Series(dt[desc_key_words].sum(axis=1), index=dt.index)

cols.extend(desc_key_words)
cols.append('sum_key_word')
# cols.append('fee')
X = dt[cols]
y = dt['fee']

# print(X.corr())
# X.corr().to_csv('C:\\Users\\AdmYL\\Desktop\\PriceOptimizer\\col_correlation_with_fee.txt')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

STEPS = 5000

colLeadSource = tf.feature_column.categorical_column_with_identity(key="Ldc_LeadSource", num_buckets=8)
colStatus = tf.feature_column.categorical_column_with_identity(key="Ldc_Status", num_buckets=8)
colTrustee = tf.feature_column.categorical_column_with_identity(key="Ldc_Trustee", num_buckets=7)
colFeeType = tf.feature_column.categorical_column_with_identity(key="Ldc_FeeType", num_buckets=4)
colCustomerTypeCode = tf.feature_column.categorical_column_with_identity(key="CustomerTypeCode", num_buckets=13)
colOwnerIdName = tf.feature_column.categorical_column_with_identity(key="OwnerIdName", num_buckets=50)
colCountry = tf.feature_column.categorical_column_with_identity(key="Ldc_CountryIncorporatedIn", num_buckets=250)
colOwningBUnit = tf.feature_column.categorical_column_with_identity(key="OwningBusinessUnit", num_buckets=4)
colCurrency = tf.feature_column.categorical_column_with_identity(key="TransactionCurrencyId", num_buckets=4)
colOpenRevenueState = tf.feature_column.categorical_column_with_identity(key="OpenRevenue_State", num_buckets=8)

feature_columns = [
    # tf.feature_column.indicator_column(colLeadSource),
    tf.feature_column.indicator_column(colStatus),
    # tf.feature_column.indicator_column(colTrustee),
    # tf.feature_column.indicator_column(colFeeType),
    # tf.feature_column.numeric_column(key="ga_PerpetualAppointment"),
    # tf.feature_column.indicator_column(colCustomerTypeCode),
    # tf.feature_column.indicator_column(colOwnerIdName),
    # tf.feature_column.indicator_column(colOwningBUnit),
    tf.feature_column.indicator_column(colCountry),
    # tf.feature_column.indicator_column(colCurrency),
    # tf.feature_column.numeric_column(key="StateCode"),
    # tf.feature_column.indicator_column(colOpenRevenueState),
    tf.feature_column.numeric_column(key="duration"),
]

for col in desc_key_words: 
    feature_columns.append(tf.feature_column.numeric_column(key=col))
# feature_columns.append(tf.feature_column.numeric_column(key="sum_key_word"))

model = tf.estimator.DNNRegressor(
    model_dir='C:\\Users\\AdmYL\\Desktop\\PriceOptimizer\\DNNModel',
    hidden_units=[1024, 512], 
    feature_columns=feature_columns)

def input_fn(features, labels, batch_size):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    # Shuffle, repeat, and batch the examples.
    return dataset.shuffle(10000).batch(batch_size).repeat().make_one_shot_iterator().get_next() 

def input_train_set():
    features = {
        'Ldc_LeadSource': np.array(X_train['Ldc_LeadSource']),
        'Ldc_Status': np.array(X_train['Ldc_Status']),
        'Ldc_Trustee': np.array(X_train['Ldc_Trustee']),
        'Ldc_FeeType': np.array(X_train['Ldc_FeeType']),
        'ga_PerpetualAppointment': np.array(X_train['ga_PerpetualAppointment']),
        'CustomerTypeCode': np.array(X_train['CustomerTypeCode']),
        'OwnerIdName': np.array(X_train['OwnerIdName']),
        'OwningBusinessUnit': np.array(X_train['OwningBusinessUnit']),
        'StateCode': np.array(X_train['StateCode']),
        'TransactionCurrencyId': np.array(X_train['TransactionCurrencyId']),
        'OpenRevenue_State': np.array(X_train['OpenRevenue_State']),
        'Ldc_CountryIncorporatedIn': np.array(X_train['Ldc_CountryIncorporatedIn']),
        'duration': np.array(X_train['duration'])
    }
    for col in desc_key_words: 
        features[col] = np.array(X_train[col])
    features['sum_key_word'] = np.array(X_train['sum_key_word'])
    labels = np.array(y_train)
    return input_fn(features, labels, 32)

def input_test_set():
    features = {
        'Ldc_LeadSource': np.array(X_test['Ldc_LeadSource']),
        'Ldc_Status': np.array(X_test['Ldc_Status']),
        'Ldc_Trustee': np.array(X_test['Ldc_Trustee']),
        'Ldc_FeeType': np.array(X_test['Ldc_FeeType']),
        'ga_PerpetualAppointment': np.array(X_test['ga_PerpetualAppointment']),
        'CustomerTypeCode': np.array(X_test['CustomerTypeCode']),
        'OwnerIdName': np.array(X_test['OwnerIdName']),
        'OwningBusinessUnit': np.array(X_test['OwningBusinessUnit']),
        'StateCode': np.array(X_test['StateCode']),
        'TransactionCurrencyId': np.array(X_test['TransactionCurrencyId']),
        'OpenRevenue_State': np.array(X_test['OpenRevenue_State']),
        'Ldc_CountryIncorporatedIn': np.array(X_test['Ldc_CountryIncorporatedIn']),
        'duration': np.array(X_test['duration'])
    }
    for col in desc_key_words: 
        features[col] = np.array(X_test[col])
    features['sum_key_word'] = np.array(X_test['sum_key_word'])
    labels = np.array(y_test)
    return input_fn(features, labels, 32)

# Train the model.
model.train(input_fn=input_train_set, steps=STEPS)

# Evaluate how the model performs on data it has not yet seen.
eval_result = model.evaluate(input_fn=input_test_set, steps=STEPS * 2)

# The evaluation returns a Python dictionary. The "average_loss" key holds the
# Mean Squared Error (MSE).
average_loss = eval_result["average_loss"]

with open ('C:\\Users\\AdmYL\\Desktop\\PriceOptimizer\\DNNRegressor_comparison.txt', 'a', encoding='utf-8') as m: 
    m.write("RMS error for the test set: {:.2f}\n\r\n\r".format(average_loss**0.5))

# Convert MSE to Root Mean Square Error (RMSE).
print("\nRMS error for the test set: {:.2f}".format(average_loss**0.5))