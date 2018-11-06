import pyodbc, nltk, re, string, collections
import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from nltk.util import ngrams
from nltk.corpus import stopwords

# regex matching NSP numbers
rgxNSP = r'NSP\D{0,2}\/{0,1}\d{2}\D{0,2}\/{0,1}\d{2}\D{0,2}\/{0,1}\d*'
# regex matching date formats
rgxDate = r'(?=(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\d{4}(?=\b)|\d{1,2}(?=st|nd|rd|th|\b)))\w{0,10}\W{0,10}(\d{1,2}(?=st|nd|rd|th|\b)|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\D{0,10}(\d{4}(?=\b)|\d{1,2}(?=st|nd|rd|th|\b))'

def replaceNoise(item):
    # using regex to remove noisy information
    item = re.sub(rgxNSP, '', item)
    item = re.sub(rgxDate, '', item)
    item = re.sub('dated', '', item)
    return item

sopdb = pyodbc.connect("Driver={SQL Server Native Client 11.0};"
                    "Server=WSTCRMDB02;"
                    "Database=SOP_MSCRM;"
                    "Trusted_Connection=yes;")

# sql = '''
# SELECT 
#     Ldc_Trustee,
# 	ldc_appointingcompanyidName,
# 	CASE WHEN [Ldc_Description] IS NULL THEN '' ELSE REPLACE(REPLACE([Ldc_Description], CHAR(13), ''), CHAR(10), '') END as 'Ldc_Description',
# 	CASE WHEN [Ldc_AcceptanceDate] IS NOT NULL OR [Ldc_TerminationDate] IS NOT NULL AND DATEDIFF(day, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) > 0 
# 	THEN DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) ELSE 0 END as duration,
# 	CONVERT(int, [Ldc_AppointmentFees] / l.[ExchangeRate]) as fee,
# 	Ldc_Notes
# FROM [SOP_MSCRM].[dbo].[Ldc_appointment] l
# WHERE [Ldc_AcceptanceDate] > '01 Oct 2016' and  CONVERT(int, [Ldc_AppointmentFees] / [ExchangeRate]) >= 1000 and 
# 	(Ldc_Description not like '%extension%' and Ldc_Description not like '%expansion%')
# 	--and (ldc_notes like '%discount%' or ldc_notes like '%reduce%' or ldc_notes like '%reduction%' or ldc_notes like '%[%] of%')
# '''

# sql = '''
# declare @range int = 20
# SELECT 
#     Ldc_Trustee,
# 	ldc_appointingcompanyidName,
# 	CASE WHEN [Ldc_Description] IS NULL THEN '' ELSE REPLACE(REPLACE([Ldc_Description], CHAR(13), ''), CHAR(10), '') END as 'Ldc_Description',
# 	CASE WHEN [Ldc_AcceptanceDate] IS NOT NULL OR [Ldc_TerminationDate] IS NOT NULL AND DATEDIFF(day, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) > 0 
# 	THEN DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) ELSE 0 END as duration,
# 	CONVERT(int, [Ldc_AppointmentFees] / l.[ExchangeRate]) as fee,
# 	--DATEPART(month, [Ldc_AcceptanceDate]) as 'month',
# 	Ldc_Notes
# FROM [SOP_MSCRM].[dbo].[Ldc_appointment] l
# WHERE [Ldc_AcceptanceDate] > '01 Oct 2016' and
# 	(Ldc_Description not like '%extension%' and Ldc_Description not like '%expansion%')
# 	and CONVERT(int, [Ldc_AppointmentFees] / [ExchangeRate]) >= 0 AND CONVERT(int, [Ldc_AppointmentFees] / [ExchangeRate]) <= 1610 
# 	and (Ldc_Trustee != 1 or Ldc_Trustee is null) 
# 	and ((ldc_notes like '%discount%' or ldc_notes like '%reduce%' or ldc_notes like '%reduction%' or ldc_notes like '%[%] of%')
# 	or not (CONVERT(int, [Ldc_AppointmentFees] / [ExchangeRate]) between 400 - @range and 400 + @range
# 	OR CONVERT(int, [Ldc_AppointmentFees] / [ExchangeRate]) between 485 - @range and 485 + @range
# 	OR CONVERT(int, [Ldc_AppointmentFees] / [ExchangeRate]) between 570 - @range and 570 + @range
# 	OR CONVERT(int, [Ldc_AppointmentFees] / [ExchangeRate]) between 655 - @range and 655 + @range
# 	OR CONVERT(int, [Ldc_AppointmentFees] / [ExchangeRate]) between 740 - @range and 740 + @range
# 	OR CONVERT(int, [Ldc_AppointmentFees] / [ExchangeRate]) between 825 - @range and 825 + @range
# 	OR CONVERT(int, [Ldc_AppointmentFees] / [ExchangeRate]) between 910 - @range and 910 + @range
# 	OR CONVERT(int, [Ldc_AppointmentFees] / [ExchangeRate]) between 995 - @range and 995 + @range
# 	OR CONVERT(int, [Ldc_AppointmentFees] / [ExchangeRate]) between 1080 - @range and 1080 + @range
# 	OR CONVERT(int, [Ldc_AppointmentFees] / [ExchangeRate]) between 1165 - @range and 1165 + @range
# 	OR CONVERT(int, [Ldc_AppointmentFees] / [ExchangeRate]) between 1250 - @range and 1250 + @range
# 	OR CONVERT(int, [Ldc_AppointmentFees] / [ExchangeRate]) between 1335 - @range and 1335 + @range
# 	OR CONVERT(int, [Ldc_AppointmentFees] / [ExchangeRate]) between 1420 - @range and 1420 + @range
# 	OR CONVERT(int, [Ldc_AppointmentFees] / [ExchangeRate]) between 1505 - @range and 1505 + @range
# 	OR CONVERT(int, [Ldc_AppointmentFees] / [ExchangeRate]) between 1590 - @range and 1590 + @range)) '''

sql = '''
SELECT
	ldc_appointingcompanyidName,
	case when ldc_introducedbycontactidYomiName is not null and ldc_introducedbyorganisationidName is not null then 1 else 0 end as 'isIntroduced',
    CASE WHEN [Ldc_Description] IS NULL THEN '' ELSE REPLACE(REPLACE([Ldc_Description], CHAR(13), ''), CHAR(10), '') END as 'Ldc_Description',
	CASE WHEN [Ldc_AcceptanceDate] IS NOT NULL OR [Ldc_TerminationDate] IS NOT NULL AND DATEDIFF(day, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) > 0 
	THEN DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) ELSE 0 END as duration,
	CONVERT(int, [Ldc_AppointmentFees] / l.[ExchangeRate]) as fee,
	ldc_notes
FROM [SOP_MSCRM].[dbo].[Ldc_appointment] l
WHERE [Ldc_AcceptanceDate] > '01 Oct 2016' AND 
	(Ldc_Description not like '%extension%' and Ldc_Description not like '%expansion%') AND 
	( 
		CONVERT(int, [Ldc_AppointmentFees] / l.[ExchangeRate]) >= 1000 AND (
		(ldc_notes like '%discount%' or ldc_notes like '%reduce%' or ldc_notes like '%reduction%' or ldc_notes like '%[%] of%') OR 
		(
		(CONVERT(int, [Ldc_AppointmentFees] / l.[ExchangeRate]) < 465 AND DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) = 2) OR
		(CONVERT(int, [Ldc_AppointmentFees] / l.[ExchangeRate]) < 550 AND DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) = 3) OR
		(CONVERT(int, [Ldc_AppointmentFees] / l.[ExchangeRate]) < 635 AND DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) = 4) OR
		(CONVERT(int, [Ldc_AppointmentFees] / l.[ExchangeRate]) < 720 AND DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) = 5) OR
		(CONVERT(int, [Ldc_AppointmentFees] / l.[ExchangeRate]) < 805 AND DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) = 6) OR
		(CONVERT(int, [Ldc_AppointmentFees] / l.[ExchangeRate]) < 890 AND DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) = 7) OR
		(CONVERT(int, [Ldc_AppointmentFees] / l.[ExchangeRate]) < 975 AND DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) = 8) OR
		(CONVERT(int, [Ldc_AppointmentFees] / l.[ExchangeRate]) < 1060 AND DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) = 9) OR
		(CONVERT(int, [Ldc_AppointmentFees] / l.[ExchangeRate]) < 1145 AND DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) = 10) OR
		(CONVERT(int, [Ldc_AppointmentFees] / l.[ExchangeRate]) < 1230 AND DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) = 11) OR
		(CONVERT(int, [Ldc_AppointmentFees] / l.[ExchangeRate]) < 1315 AND DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) = 12) OR
		(CONVERT(int, [Ldc_AppointmentFees] / l.[ExchangeRate]) < 1400 AND DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) = 13) OR
		(CONVERT(int, [Ldc_AppointmentFees] / l.[ExchangeRate]) < 1485 AND DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) = 14) OR
		(CONVERT(int, [Ldc_AppointmentFees] / l.[ExchangeRate]) < 1570 AND DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) = 15) OR
		(CONVERT(int, [Ldc_AppointmentFees] / l.[ExchangeRate]) < 1655 AND DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) = 16) OR
		(CONVERT(int, [Ldc_AppointmentFees] / l.[ExchangeRate]) < 1740 AND DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) = 17) OR
		(CONVERT(int, [Ldc_AppointmentFees] / l.[ExchangeRate]) < 1825 AND DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) = 18) OR
		(CONVERT(int, [Ldc_AppointmentFees] / l.[ExchangeRate]) < 1910 AND DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) = 19) OR
		(CONVERT(int, [Ldc_AppointmentFees] / l.[ExchangeRate]) < 1995 AND DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) = 20) OR
		(CONVERT(int, [Ldc_AppointmentFees] / l.[ExchangeRate]) < 2080 AND DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) = 21) OR
		(CONVERT(int, [Ldc_AppointmentFees] / l.[ExchangeRate]) < 2165 AND DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) = 22) OR
		(CONVERT(int, [Ldc_AppointmentFees] / l.[ExchangeRate]) < 2250 AND DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) = 23) OR
		(CONVERT(int, [Ldc_AppointmentFees] / l.[ExchangeRate]) < 2335 AND DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) = 24) OR
		(CONVERT(int, [Ldc_AppointmentFees] / l.[ExchangeRate]) < 2420 AND DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) = 25) OR
		(CONVERT(int, [Ldc_AppointmentFees] / l.[ExchangeRate]) < 2505 AND DATEDIFF(year, [Ldc_AcceptanceDate], [Ldc_TerminationDate]) > 25)
		)
		)
	)
'''

dt = pd.read_sql(sql, sopdb)
dt = dt.dropna()
dt['Ldc_Description'] = dt['Ldc_Description'].apply(replaceNoise)

text = " ".join(d for d in dt['Ldc_Description'])

# get rid of punctuation (except periods!)
punctuationNoPeriod = "[" + re.sub(r"\.","",string.punctuation) + "]"
text = re.sub(punctuationNoPeriod, "", text)

# Create and generate a word cloud image:
wordcloud = WordCloud(width=1200, height=600, max_words=50, background_color="white").generate(text)

# Display the generated image:
plt.figure(figsize=(20,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('C:\\Users\\AdmYL\\Desktop\\PriceOptimizer\\LdcDescWorldCloud.png')

# calculate and print out the top n most common bigrams
# tokenized = text.split()
# tokenized = [t for t in tokenized if t not in stopwords.words('english')]
# bigrams = ngrams(tokenized, 2)
# bigramFreq = collections.Counter(bigrams)
# print(bigramFreq.most_common(500))