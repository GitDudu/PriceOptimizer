import pyodbc, io, re, pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from time import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

plt.style.use('seaborn-poster')

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
            CASE WHEN Ldc_Status IS NULL THEN 0 ELSE Ldc_Status END as 'Ldc_Status',
            CASE WHEN [Ldc_CountryIncorporatedIn] IS NULL THEN 0 ELSE [Ldc_CountryIncorporatedIn] END as 'Ldc_CountryIncorporatedIn',
            CONVERT(int, [Ldc_AppointmentFees] / a.[ExchangeRate]) as 'fee',
            CASE WHEN [Ldc_AcceptanceDate] IS NOT NULL AND [Ldc_TerminationDate] IS NOT NULL AND DATEDIFF(day, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) > 0 
	            THEN DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) ELSE 0 END as 'duration',
            CASE WHEN [Ldc_Description] IS NULL THEN '' ELSE [Ldc_Description] END as 'Ldc_Description',
            CASE WHEN [Ldc_Description] IS NULL THEN 0 ELSE len([Ldc_Description]) END as 'Len_Ldc_Description',
            CASE WHEN [Ldc_AcceptanceDate] IS NOT NULL THEN DATEPART(month, [Ldc_AcceptanceDate]) ELSE 0 END as 'month',
            (select count(l.ldc_appointingcompanyidName) from [SOP_MSCRM].[dbo].[Ldc_appointment] l 
		        where l.[Ldc_AcceptanceDate] > '01 Oct 2016' and l.ldc_appointingcompanyidName = a.ldc_appointingcompanyidName) as 'app_count',
            CASE WHEN (select count(l.ldc_appointingcompanyidName) from [SOP_MSCRM].[dbo].[Ldc_appointment] l 
		        where l.[Ldc_AcceptanceDate] > '01 Oct 2016' and l.ldc_appointingcompanyidName = a.ldc_appointingcompanyidName) > 3 THEN 1 ELSE 0 END as 'is_frequent'
        FROM [SOP_MSCRM].[dbo].[Ldc_appointment] a
        INNER JOIN [SOP_MSCRM].[dbo].[Account] c
        ON a.ldc_appointingcompanyid = c.AccountId AND 
            TRY_CONVERT(UNIQUEIDENTIFIER, a.ldc_appointingcompanyid) IS NOT NULL AND 
            TRY_CONVERT(UNIQUEIDENTIFIER, c.AccountId) IS NOT NULL
        WHERE Ldc_Trustee = 1 AND 
            CONVERT(int, [Ldc_AppointmentFees] / a.[ExchangeRate]) >= 380 AND 
            CONVERT(int, [Ldc_AppointmentFees] / a.[ExchangeRate]) <= 1500 AND 
            (not (ldc_notes like '%discount%' or ldc_notes like '%reduce%' or ldc_notes like '%reduction%' or ldc_notes like '%[%] of%') or Ldc_Notes is null) AND            
            (Ldc_Description not like '%extension%' and Ldc_Description not like '%expansion%') AND
            not (
                (CONVERT(int, [Ldc_AppointmentFees] / a.[ExchangeRate]) < 465 AND DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) = 2) OR
                (CONVERT(int, [Ldc_AppointmentFees] / a.[ExchangeRate]) < 550 AND DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) = 3) OR
                (CONVERT(int, [Ldc_AppointmentFees] / a.[ExchangeRate]) < 635 AND DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) = 4) OR
                (CONVERT(int, [Ldc_AppointmentFees] / a.[ExchangeRate]) < 720 AND DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) = 5) OR
                (CONVERT(int, [Ldc_AppointmentFees] / a.[ExchangeRate]) < 805 AND DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) = 6) OR
                (CONVERT(int, [Ldc_AppointmentFees] / a.[ExchangeRate]) < 890 AND DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) = 7) OR
                (CONVERT(int, [Ldc_AppointmentFees] / a.[ExchangeRate]) < 975 AND DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) = 8) OR
                (CONVERT(int, [Ldc_AppointmentFees] / a.[ExchangeRate]) < 1060 AND DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) = 9) OR
                (CONVERT(int, [Ldc_AppointmentFees] / a.[ExchangeRate]) < 1145 AND DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) = 10) OR
                (CONVERT(int, [Ldc_AppointmentFees] / a.[ExchangeRate]) < 1230 AND DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) = 11) OR
                (CONVERT(int, [Ldc_AppointmentFees] / a.[ExchangeRate]) < 1315 AND DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) = 12) OR
                (CONVERT(int, [Ldc_AppointmentFees] / a.[ExchangeRate]) < 1400 AND DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) = 13) OR
                (CONVERT(int, [Ldc_AppointmentFees] / a.[ExchangeRate]) < 1485 AND DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) = 14) OR
                (CONVERT(int, [Ldc_AppointmentFees] / a.[ExchangeRate]) < 1570 AND DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) = 15) OR
                (CONVERT(int, [Ldc_AppointmentFees] / a.[ExchangeRate]) < 1655 AND DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) = 16) OR
                (CONVERT(int, [Ldc_AppointmentFees] / a.[ExchangeRate]) < 1740 AND DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) = 17) OR
                (CONVERT(int, [Ldc_AppointmentFees] / a.[ExchangeRate]) < 1825 AND DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) = 18) OR
                (CONVERT(int, [Ldc_AppointmentFees] / a.[ExchangeRate]) < 1910 AND DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) = 19) OR
                (CONVERT(int, [Ldc_AppointmentFees] / a.[ExchangeRate]) < 1995 AND DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) = 20) OR
                (CONVERT(int, [Ldc_AppointmentFees] / a.[ExchangeRate]) < 2080 AND DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) = 21) OR
                (CONVERT(int, [Ldc_AppointmentFees] / a.[ExchangeRate]) < 2165 AND DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) = 22) OR
                (CONVERT(int, [Ldc_AppointmentFees] / a.[ExchangeRate]) < 2250 AND DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) = 23) OR
                (CONVERT(int, [Ldc_AppointmentFees] / a.[ExchangeRate]) < 2335 AND DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) = 24) OR
                (CONVERT(int, [Ldc_AppointmentFees] / a.[ExchangeRate]) < 2420 AND DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) = 25) OR
                (CONVERT(int, [Ldc_AppointmentFees] / a.[ExchangeRate]) < 2505 AND DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) > 25)
            ) AND 
            [Ldc_AcceptanceDate] > '01 Oct 2016' '''

dt = pd.read_sql(sql, sopdb)
dt = dt.dropna()
dt['Ldc_Description'] = dt['Ldc_Description'].apply(replaceNoise)

tcols = ['Ldc_CountryIncorporatedIn']
# turn chosen columns into categorical data
for col in tcols:
    i = 0
    odict = {}
    for sf in set(dt[col]):
        odict[sf] = i
        i = i + 1
    dt[col] = dt[col].map(odict)

# list of key words in the description column
desc_key_words = ['facility','isda','intercreditor','repurchase','purchase','agency_agreement','loan_agreement',
    'lease_agreement','aircraft_lease','deed','security','share','amend','series','swap','supplement','indenture',
    'underwriting_agreement','credit','guarantee','notes_due','other related agreement','other related agreements']

# categorical columns
cols_ct = ['Ldc_CountryIncorporatedIn','Ldc_Status','month']
for col in cols_ct:
    dt[col] = pd.to_numeric(dt[col], errors='coerce')

# continuous columns
cols_tu = ['app_count', 'duration', 'Len_Ldc_Description'] #'is_frequent'
for col in cols_tu: 
    dt[col] = pd.to_numeric(dt[col], errors='coerce').astype(int)
dt['fee'] = pd.to_numeric(dt['fee'], errors='coerce').astype(int)

# function to count key words in the descriptions
def word_count(s, w):
    return s.count(w)

# key words columns, all are continuous
for col in desc_key_words: 
    dt.loc[:, col] = pd.Series(dt['Ldc_Description'].apply(lambda x: word_count(x, col.replace('_', ' '))), index=dt.index)
dt.loc[:, 'sum_key_word'] = pd.Series(dt[desc_key_words].sum(axis=1).apply(lambda x: 1 if x == 0 else x), index=dt.index)

continuous_cols = desc_key_words
continuous_cols.extend(cols_tu)
continuous_cols.append('sum_key_word')
categorical_cols = cols_ct

cols = continuous_cols + categorical_cols

X = dt[cols]
y = dt['fee']

# split train and test data sets
spl = np.random.rand(len(dt)) < 0.8
X_train = X[spl]
X_test = X[~spl]
y_train = y[spl]
y_test = y[~spl]

X_train_continuous = X_train[continuous_cols]
X_train_categorical = X_train[categorical_cols]

X_test_continuous = X_test[continuous_cols]
X_test_categorical = X_test[categorical_cols]

# interestingly, no normalization produces better results on this data set, but only mildly better though
# normalizing the continuous columns of both train and test sets to have 0 mean and std of 1
# mean = X_train_continuous.mean(axis=0)
# std = X_train_continuous.std(axis=0)
# X_train_continuous = (X_train_continuous - mean) / std
# X_test_continuous = (X_test_continuous - mean) / std

# Define the embedding inputs
ctry_embedding_input = keras.layers.Input(shape=(1,), dtype='int32')
status_embedding_input = keras.layers.Input(shape=(1,), dtype='int32') 
month_embedding_input = keras.layers.Input(shape=(1,), dtype='int32') 

# Define the continuous variables input
continuous_input = keras.layers.Input(shape=(X_train_continuous.shape[1], ))

# define the embedding layers and flatten them
ctry_embedding = keras.layers.Embedding(output_dim=5, input_dim=len(X_train_categorical['Ldc_CountryIncorporatedIn'])+1, input_length=1)(ctry_embedding_input)
ctry_embedding = keras.layers.Reshape((5,))(ctry_embedding)

status_embedding = keras.layers.Embedding(output_dim=2, input_dim=8, input_length=1)(status_embedding_input)
status_embedding = keras.layers.Reshape((2,))(status_embedding)

month_embedding = keras.layers.Embedding(output_dim=2, input_dim=13, input_length=1)(month_embedding_input)
month_embedding = keras.layers.Reshape((2,))(month_embedding)

# combine all inputs
all_input = keras.layers.concatenate([continuous_input, ctry_embedding, status_embedding, month_embedding])

# Define the model
dense0 = keras.layers.Dense(units=256, activation=tf.nn.relu)(all_input)
dense1 = keras.layers.Dense(units=128, activation=tf.nn.relu)(dense0)
dense2 = keras.layers.Dense(units=64, activation=tf.nn.relu)(dense1)
dense3 = keras.layers.Dense(units=32, activation=tf.nn.relu)(dense2)
dense4 = keras.layers.Dense(units=16, activation=tf.nn.relu)(dense3)
predictions = keras.layers.Dense(1)(dense4)

model = keras.models.Model(inputs=[continuous_input, ctry_embedding_input, status_embedding_input, month_embedding_input], outputs=predictions)

model.compile(loss='mse', optimizer=tf.train.AdamOptimizer(0.001), metrics=['mae'])

class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 500

# stop the training when performance is idle 
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)

# Store training stats
history = model.fit([X_train_continuous, X_train_categorical['Ldc_CountryIncorporatedIn'], X_train_categorical['Ldc_Status'], X_train_categorical['month']], y_train,
                    batch_size=32, epochs=EPOCHS, validation_split=0.2, verbose=0, shuffle=True,
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

scores = model.evaluate([X_test_continuous, X_test_categorical['Ldc_CountryIncorporatedIn'], X_test_categorical['Ldc_Status'], X_test_categorical['month']], y_test, verbose=0)

print("Testing set %s: %.2f" % (model.metrics_names[1], scores[1]))

with open ('C:\\Users\\AdmYL\\Desktop\\PriceOptimizer\\tensorflow_comparison.txt', 'a', encoding='utf-8') as m: 
    m.write("Testing set %s: %.2f" % (model.metrics_names[1], scores[1]))

# save the prediction result in to a spreadsheet for further analysis
pred = model.predict([X_test_continuous, X_test_categorical['Ldc_CountryIncorporatedIn'], X_test_categorical['Ldc_Status'], X_test_categorical['month']], verbose=0)
rl = list(zip(pred, y_test, X_test_continuous['sum_key_word'], X_test_continuous['duration'], X_test_continuous['app_count'], X_test_categorical['Ldc_CountryIncorporatedIn'], X_test_categorical['Ldc_Status'], X_test_categorical['month']))
df_rl = pd.DataFrame(data=rl, columns=['prediction', 'y_test', 'sum_key_word', 'duration', 'app_count', 'Ldc_CountryIncorporatedIn', 'Ldc_Status', 'month'])
df_rl.to_csv('C:\\Users\\AdmYL\\Desktop\\PriceOptimizer\\tensorflow_pred_result.csv', index=False, header=True)