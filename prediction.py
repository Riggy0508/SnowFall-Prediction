import numpy as np
import pandas as pd
from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objs as go
import tarfile
import gzip
import re
import os
import datetime as dt
import warnings
warnings.filterwarnings("ignore")

init_notebook_mode(connected=True)


year_num = 20


differ_year_Weather_data = os.listdir("D:/Big data/archive/gsod_all_years")
differ_year_Weather_data.sort()
differ_year_Weather_data = differ_year_Weather_data[-year_num:]
#need to check
years = [int(re.findall('\d+',diff_year)[0]) for diff_year in differ_year_Weather_data]

station_map = pd.read_csv('D:/Big data/archive/isd-history.csv')
station_map = station_map.replace([0.0, -999.0, -999.9],np.nan)
station_map = station_map[pd.notnull(station_map['LAT']) & pd.notnull(station_map['LON'])]
#need to check
station_map = station_map[[int(re.findall('^\d{4}', str(last_year))[0])==max(years) for last_year in station_map['END']]]
station_map = station_map[[int(re.findall('^\d{4}', str(start_year))[0])<=min(years) for start_year in station_map['BEGIN']]]


station_map['LBL'] = station_map[['STATION NAME','STATE','CTRY']].apply(lambda x: x.str.cat(sep=', '), axis=1)
station_map['ELEV_LBL'] = station_map['ELEV(M)'].apply(lambda x: 'Elevation: '+str(x)+' m' if ~np.isnan(x) else np.nan)
station_map['LBL'] = station_map[['LBL','ELEV_LBL']].apply(lambda x: x.str.cat(sep='<br>'), axis=1)
station_map = station_map.drop(['STATION NAME','STATE','ELEV_LBL','ICAO','BEGIN','END'], axis=1)


df = pd.DataFrame([])
df_day = pd.DataFrame([])

def preprocess_station_file_content(data_present):
    headers=data_present.pop(0)
    headers=[headers[ind] for ind in [0,1,2,3,4,8,11,12,13]]
    for d in range(len(data_present)):
        data_present[d]=[data_present[d][ind] for ind in [0,1,2,3,5,13,17,18,19]]
    data_present=pd.DataFrame(data_present, columns=headers)
    data_present.rename(columns={'STN---': 'USAF'}, inplace=True)
    data_present['MAX'] = data_present['MAX'].apply(lambda x: re.sub("\*$","",x))
    data_present['MIN'] = data_present['MIN'].apply(lambda x: re.sub("\*$","",x))

    data_present['PRCP'] = data_present['PRCP'].apply(lambda x: re.sub(x,x[:-1],x))
    data_present[['WBAN','TEMP','DEWP','WDSP','MAX','MIN','PRCP']] = data_present[['WBAN','TEMP','DEWP','WDSP','MAX','MIN','PRCP']].apply(pd.to_numeric)
    data_present['YEARMODA']=pd.to_datetime(data_present['YEARMODA'], format='%Y%m%d', errors='ignore')
    data_present['YEAR']=pd.DatetimeIndex(data_present['YEARMODA']).year
    data_present['MONTH']=pd.DatetimeIndex(data_present['YEARMODA']).month
    data_present['DAY']=pd.DatetimeIndex(data_present['YEARMODA']).day
    return data_present

for diff_year in differ_year_Weather_data:
    print(diff_year)
    i=0
    tar = tarfile.open("D:/Big data/archive/gsod_all_years/"+diff_year, "r")
    print(len(tar.getmembers()[1:]))
    for single_file in tar.getmembers()[1:]:
        name_parts = re.sub("\.op\.gz$","",re.sub("^\./","",single_file.name)).split("-")
        usaf = name_parts[0]
        wban = int(name_parts[1])
        if station_loc[(station_loc['USAF']==usaf) & (station_loc['WBAN']==wban)].shape[0]!=0:
            i=i+1
            f=tar.extractfile(member)
            f=gzip.open(f, 'rb')
            data_present=[re.sub(" +", ",", line.decode("utf-8")).split(",") for line in f.readlines()]
            data_present=preprocess_station_file_content(data_present)
            df_day = df_day.append(data_present[data_present['YEARMODA']==data_present['YEARMODA'].max()])
            data_present = data_present.groupby(['USAF','WBAN','YEAR','MONTH']).agg('median').reset_index()
            df = df.append(data_present)
    tar.close()

df_location = pd.merge(df, station_map, how='inner', on=['USAF','WBAN'])
df_location.to_csv('D:/Big data/csv/sample.csv')

import findspark
findspark.init()

import pyspark.pandas
 #import databricks.koalas

import numpy
 #import matplotlib.pyplot as plt
import pandas

from pyspark.sql import SparkSession
# from pyspark.ml.regression import RandomForestRegressor
# from pyspark.ml.evaluation import RegressionEvaluator
# from sklearn.ensemble import RandomForestRegressor

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
sc = SparkContext.getOrCreate();
sqlContext = SQLContext(sc)
snwFlPred = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('D:/Big data/csv/sample.csv')
snwFlPred.take(1)


import six
for i in snwFlPred.columns:
    if not( isinstance(snwFlPred.select(i).take(1)[0][0], six.string_types)):
        print( "Correlation to PRCP for ", i, snwFlPred.stat.corr('PRCP',i))

#code to assemble the vecors as feature and prcp
from pyspark.ml.feature import VectorAssembler
vectorAssembler = VectorAssembler(inputCols = ['TEMP','DEWP','WDSP','MAX','MIN'], outputCol = 'features')
vsnwFlPred = vectorAssembler.transform(snwFlPred)
vsnwFlPred = vsnwFlPred.select(['features', 'PRCP','LAT','LON'])
vsnwFlPred.show(3)

#split the data into training and testing data
splits = vsnwFlPred.randomSplit([0.7, 0.3])
train_df = splits[0]
test_df = splits[1]
# from pyspark.ml.regression import RandomForestRegressor
# from pyspark.ml.evaluation import RegressionEvaluator
# from sklearn.ensemble import RandomForestRegressor

# #Reading the data
# #data = spark.read.format("libsvm").load("D:/Big data/csv/sample.csv")
# #data= pd.read_csv('D:/Big data/csv/sample.csv')
# #pyspark.read.option("header","false").csv("D:/Big data/csv/sample.csv")

# data= pyspark.pandas.read_csv('D:/Big data/csv/sample.csv',headers= False)

# # just checking
# #print(data)
# #print(type(data))

# #(trainingData, testData) = data.randomSplit([0.7, 0.3])

# splits = data.to_spark().randomSplit([0.7, 0.3], seed=12)
# trainingData = splits[0].to_koalas()
# testData = splits[1].to_koalas()

# #splits = data.randomSplit([0.7, 0.3], 24)
# #trainingData = splits[0].to_koalas()
# #testData = splits[1].to_koalas()

# train_x= trainingData.iloc [:,5]
# train_y= trainingData.iloc [:,10]
# test_x= testData.iloc [:,5]

# #print(train_x)

# #print(train_y)

# regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
# #regressor.fit(train_x.to_frame(), train_y)

# train_x = train_x.to_numpy()
# train_y = train_y.to_numpy()
# test_x = test_x.to_numpy()

# #make them 2D-arrays
# train_x.reshape(-1,1)
# train_y.reshape(-1,1)
# test_x.reshape(-1,1)

# regressor.fit(train_x.reshape(-1,1), train_y.reshape(-1,1))

# #regressor.fit(train_x, train_y)
# #test_x = testData.iloc [:, 5] # ” : ” means it will select all rows
# y_pred = regressor.predict(test_x.reshape(-1,1))


#gradient boost regressor
from pyspark.ml.regression import GBTRegressor
gradient_boost = GBTRegressor(featuresCol = 'features', labelCol = 'PRCP', maxIter=10)
gradient_boost_model = gradient_boost.fit(train_df)
gradient_boost_predictions = gradient_boost_model.transform(test_df)
gradient_boost_predictions.select('prediction', 'PRCP', 'features').show(5)

#evaluating gradient boost regression test data
gradient_boost_evaluator = RegressionEvaluator(labelCol="PRCP", predictionCol="prediction", metricName="rmse")
rmse = gradient_boost_evaluator.evaluate(gbt_predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
gradient_boost_predictions.toPandas().to_csv('D:/Big data/csv/prediction.csv')