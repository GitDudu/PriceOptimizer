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
            CASE WHEN c.StateCode IS NULL THEN 99 ELSE c.StateCode END as 'StateCode',
            CASE WHEN c.TransactionCurrencyId IS NULL THEN '' ELSE convert(nvarchar(100), c.TransactionCurrencyId) END as 'TransactionCurrencyId',
            CASE WHEN OpenRevenue_State IS NULL THEN '' ELSE OpenRevenue_State END as 'OpenRevenue_State',
            CASE WHEN [Ldc_CountryIncorporatedIn] IS NULL THEN 0 ELSE [Ldc_CountryIncorporatedIn] END as 'Ldc_CountryIncorporatedIn',
            CONVERT(int, [Ldc_AppointmentFees] / a.[ExchangeRate]) as 'fee',
            CASE WHEN [Ldc_AcceptanceDate] IS NOT NULL AND [Ldc_TerminationDate] IS NOT NULL AND DATEDIFF(day, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) > 0 
	            THEN DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) ELSE 0 END as 'duration',
            CASE WHEN [Ldc_Description] IS NULL THEN '' ELSE [Ldc_Description] END as 'Ldc_Description',
            CASE WHEN [Ldc_AcceptanceDate] IS NOT NULL THEN DATEPART(month, [Ldc_AcceptanceDate]) ELSE 0 END as 'month'
        FROM [SOP_MSCRM].[dbo].[Ldc_appointment] a
        INNER JOIN [SOP_MSCRM].[dbo].[Account] c
        ON a.ldc_appointingcompanyid = c.AccountId AND 
            TRY_CONVERT(UNIQUEIDENTIFIER, a.ldc_appointingcompanyid) IS NOT NULL AND 
            TRY_CONVERT(UNIQUEIDENTIFIER, c.AccountId) IS NOT NULL
        WHERE Ldc_Trustee = 1 AND 
            CONVERT(int, [Ldc_AppointmentFees] / a.[ExchangeRate]) >= 300 AND 
            CONVERT(int, [Ldc_AppointmentFees] / a.[ExchangeRate]) <= 1500 AND 
            (not (ldc_notes like '%discount%' or ldc_notes like '%reduce%' or ldc_notes like '%reduction%' or ldc_notes like '%[%] of%') or Ldc_Notes is null) AND            (Ldc_Description not like '%extension%' and Ldc_Description not like '%expansion%') AND
            [Ldc_AcceptanceDate] > '01 Oct 2016' '''

dt = pd.read_sql(sql, sopdb)
dt = dt.dropna()
dt['Ldc_Description'] = dt['Ldc_Description'].apply(replaceNoise)

tcols = ['OwnerIdName', 'OwningBusinessUnit', 'Ldc_CountryIncorporatedIn','TransactionCurrencyId']
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

# cols = ['Ldc_LeadSource','Ldc_Status','Ldc_Trustee','Ldc_FeeType','ga_PerpetualAppointment','CustomerTypeCode','OwnerIdName','OwningBusinessUnit',
#     'StateCode','TransactionCurrencyId','OpenRevenue_State','Ldc_CountryIncorporatedIn','duration']
cols = ['Ldc_CountryIncorporatedIn','Ldc_Status','duration','month']
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

X = np.array(dt[cols])
y = np.array(dt['fee'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

def build_model():
    model = keras.Sequential([
        keras.layers.Dense(128, activation=tf.nn.tanh, input_shape=(X_train.shape[1],)),
        keras.layers.Dense(64, activation=tf.nn.tanh),
        keras.layers.Dense(32, activation=tf.nn.tanh),
        keras.layers.Dense(1)
    ])

    optimizer = tf.train.AdamOptimizer(0.001) #GradientDescentOptimizer, RMSPropOptimizer

    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
    return model

model = build_model()

class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 500

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

# Store training stats
history = model.fit(X_train, y_train, batch_size=32, epochs=EPOCHS,
                    validation_split=0.2, verbose=0, shuffle=True,
                    callbacks=[early_stop, PrintDot()])

def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']), label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']), label = 'Valuation Loss')
    plt.legend()
    plt.savefig('C:\\Users\\AdmYL\\Desktop\\PriceOptimizer\\tensorflow_comparison.png')

plot_history(history)

scores = model.evaluate(X_test, y_test, verbose=0)

print("Testing set %s: %.2f" % (model.metrics_names[1], scores[1]))
# print(" Mean Abs Error: £{:7.2f}".format(mae))

with open ('C:\\Users\\AdmYL\\Desktop\\PriceOptimizer\\tensorflow_comparison.txt', 'a', encoding='utf-8') as m: 
    m.write("Testing set %s: %.2f" % (model.metrics_names[1], scores[1]))
    # m.write("Testing set Mean Abs Error: £{:7.2f} \n\r\n\r".format(mae))