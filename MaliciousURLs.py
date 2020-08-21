#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Check the current context
spark


# In[2]:


#---------------------------
# Dependencies
#---------------------------

import sys
import time
import random
import numpy as np
import matplotlib.pyplot as plt

from pyspark import SparkContext, SQLContext

from pyspark.ml import Pipeline
from pyspark.ml import linalg as ml_linalg
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler

from pyspark.mllib import linalg as mllib_linalg
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel, RandomForest
from pyspark.mllib.classification import    SVMWithSGD, SVMModel,    LogisticRegressionWithLBFGS, LogisticRegressionModel,    NaiveBayes, NaiveBayesModel
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.regression import LabeledPoint

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import *


# In[3]:


#---------------------------
# Some utility functions
#---------------------------

def mllib_vector(v):
    if isinstance(v, ml_linalg.SparseVector):
        return mllib_linalg.SparseVector(v.size, v.indices, v.values)
    if isinstance(v, ml_linalg.DenseVector):
        return mllib_linalg.DenseVector(v.values)
    raise ValueError("Unsupported type {0}".format(type(v)))

def label_name(label):
    return 'Malicious URL' if label == 1.0 else 'Benign URL'

line = 28 * '-'


# In[4]:


#---------------------------
# Training Algorithms
#---------------------------

class NvBayes():
    name = "NaiveBayes"
    def train(self, training_data):
        return NaiveBayes.train(training_data, 1.0)

class LogisticReg():
    name = "LogReg"
    def train(self, training_data):
        return LogisticRegressionWithLBFGS.train(training_data)

class SVMs():
    name = "SVMs"
    def train(self, training_data):
        return SVMWithSGD.train(training_data, iterations = 100)

class DecTree():
    name = "DecTree"
    def train(self, training_data):
        return DecisionTree.trainClassifier(
            training_data, numClasses = 2, categoricalFeaturesInfo = {}, 
            impurity = 'gini', maxDepth = 5, maxBins = 32
        )

class RandForest():
    name = "RandomForest"
    def train(self, training_data):
        return RandomForest.trainClassifier(
            training_data, numClasses = 2, categoricalFeaturesInfo = {}, numTrees = 32
        )


# In[5]:


#---------------------------
# Dataset functions
#---------------------------

adultDataSchema = StructType([
    StructField("label", DoubleType(), True), \
    
    StructField("URL", StringType(), True), \
    StructField("URL_LENGTH", DoubleType(), True), \
    StructField("NUMBER_SPECIAL_CHARACTERS", DoubleType(), True), \
    StructField("CHARSET", StringType(), True), \
    StructField("SERVER", StringType(), True), \
    StructField("CACHE_CONTROL", StringType(), True), \
    StructField("CONTENT_LENGTH", DoubleType(), True), \
    StructField("WHOIS_COUNTRY", StringType(), True), \
    StructField("WHOIS_STATEPROV", StringType(), True), \
    StructField("WHOIS_REGDATE", StringType(), True), \
    StructField("UPDATE_DATE", StringType(), True), \
    StructField("WHITIN_DOMAIN", StringType(), True), \
    StructField("TCP_CONVERSATION_EXCHANGE", DoubleType(), True), \
    StructField("DIST_REMOTE_TCP_PORT", DoubleType(), True), \
    StructField("REMOTE_IPS", DoubleType(), True), \
    StructField("APP_BYTES", DoubleType(), True), \
    StructField("UDP_PACKETS", StringType(), True), \
    StructField("TCP_URG_PACKETS", StringType(), True), \
    StructField("SOURCE_APP_PACKETS", DoubleType(), True), \
    StructField("REMOTE_APP_PACKETS", DoubleType(), True), \
    StructField("SOURCE_APP_BYTES", DoubleType(), True), \
    StructField("REMOTE_APP_BYTES", DoubleType(), True), \
    StructField("APP_PACKETS", DoubleType(), True), \
    StructField("DNS_QUERY_TIMES", DoubleType(), True), \
    StructField("TIPO", DoubleType(), True), \

    StructField("WHOIS_REGDATE_YEAR", DoubleType(), True), \
    StructField("WHOIS_REGDATE_MONTH", DoubleType(), True), \
    StructField("WHOIS_REGDATE_DAY_OF_MONTH", DoubleType(), True), \
    StructField("WHOIS_REGDATE_DAY_OF_WEEK", DoubleType(), True), \
    StructField("WHOIS_REGDATE_HOUR", DoubleType(), True), \
    StructField("WHOIS_REGDATE_MINUTE", DoubleType(), True), \
    StructField("WHOIS_REGDATE_SECOND", DoubleType(), True), \
    StructField("WHOIS_REGDATE_MILLISECOND", DoubleType(), True), \

    StructField("UPDATE_DATE_YEAR", DoubleType(), True), \
    StructField("UPDATE_DATE_MONTH", DoubleType(), True), \
    StructField("UPDATE_DATE_DAY_OF_MONTH", DoubleType(), True), \
    StructField("UPDATE_DATE_DAY_OF_WEEK", DoubleType(), True), \
    StructField("UPDATE_DATE_HOUR", DoubleType(), True), \
    StructField("UPDATE_DATE_MINUTE", DoubleType(), True), \
    StructField("UPDATE_DATE_SECOND", DoubleType(), True), \
    StructField("UPDATE_DATE_MILLISECOND", DoubleType(), True)
])
    
def loadData(dataUrl):
    return spark.read         .format("csv")         .load(dataUrl, schema = adultDataSchema).na.fill(0)

def transformData(data):

    numeric_cols = [
        "label",\
        "URL_LENGTH", "NUMBER_SPECIAL_CHARACTERS", "CONTENT_LENGTH",\
        "TCP_CONVERSATION_EXCHANGE", "DIST_REMOTE_TCP_PORT", "REMOTE_IPS", "APP_BYTES",\
        "SOURCE_APP_PACKETS", "REMOTE_APP_PACKETS", "SOURCE_APP_BYTES",\
        "REMOTE_APP_BYTES", "APP_PACKETS", "DNS_QUERY_TIMES",\
        "WHOIS_REGDATE_YEAR", "WHOIS_REGDATE_MONTH", "WHOIS_REGDATE_DAY_OF_MONTH", "WHOIS_REGDATE_DAY_OF_WEEK",\
        "WHOIS_REGDATE_HOUR", "WHOIS_REGDATE_MINUTE", "WHOIS_REGDATE_SECOND", "WHOIS_REGDATE_MILLISECOND",\
        "UPDATE_DATE_YEAR", "UPDATE_DATE_MONTH", "UPDATE_DATE_DAY_OF_MONTH", "UPDATE_DATE_DAY_OF_WEEK",\
        "UPDATE_DATE_HOUR", "UPDATE_DATE_MINUTE", "UPDATE_DATE_SECOND", "UPDATE_DATE_MILLISECOND"
    ]
    categorical_cols = [
        "URL", "CHARSET", "SERVER", "CACHE_CONTROL", "WHOIS_COUNTRY", "WHOIS_STATEPROV", "WHOIS_REGDATE", \
        "UPDATE_DATE", "WHITIN_DOMAIN", "UDP_PACKETS", "TCP_URG_PACKETS", "TIPO"
    ]
    
    # Removes the URL feature    
    categorical_cols.remove('URL')
    
    # Indexing
    indexers = [StringIndexer().setHandleInvalid("keep").setInputCol(categorical_col).setOutputCol('indexed' + categorical_col) for categorical_col in categorical_cols]
    for indexer in indexers:
        data = indexer.fit(data).transform(data) 
    
    # Assembling
    assembler = VectorAssembler()        .setInputCols(['indexed'  + categorical_col for categorical_col in categorical_cols] + numeric_cols)        .setOutputCol("features")        .setHandleInvalid("keep")
    data = assembler.transform(data)

    # Creating LabelPoints
    data = data.select(col("label"), col("features")).rdd.map(        lambda row: LabeledPoint(row.label, mllib_vector(row.features))     )
    
    return data


# In[6]:


#---------------------------
# Configuration
#---------------------------

dataUrl = "c:/temp/BigML_Dataset_5ee5cc28ace11f2ac2003edc.csv"

#---------------------------
# Execution
#---------------------------

spark = SparkSession.builder        .master("local[*]")        .appName("temp")        .getOrCreate()
sc = spark.sparkContext.getOrCreate()
spark.catalog.clearCache()

print(line)
print('Loading data from \'%s\'..' % dataUrl)
data = loadData(dataUrl)

print(line)
print("- %d entries" % data.count())
print("- %d features" % len(data.columns))

print(line)
print('Transforming data..')
transformedData = transformData(data)
training_data, test_data = transformedData.randomSplit([0.7, 0.3], seed = random.randint(0, sys.maxsize))

algos = [NvBayes(), LogisticReg(), SVMs(), DecTree(), RandForest()]

names = []
times = []
metrics = []
labels = sorted(transformedData.map(lambda lp: lp.label).distinct().collect())

for algo in algos:
    print(line)
    print('Training ' + algo.name + '...')
    
    t0 = time.time()
    
    model = algo.train(training_data)
    
    execTime = time.time() - t0

    predictions = model.predict(test_data.map(lambda lp: lp.features))
    labels_and_predictions = test_data.map(lambda lp: lp.label).zip(predictions.map(lambda lp: float(lp)))
    metricz = MulticlassMetrics(labels_and_predictions)
    
    names.append(algo.name)
    times.append(execTime)
    metrics.append(metricz)
    
    print("Execution Time: %f" % execTime)
    
    for label in labels:
        print("Model Precision (%s): %.3f" % (label_name(label), metricz.precision(label)))
        print("Model Recall (%s): %.3f" % (label_name(label), metricz.recall(label)))
        print("Model F-measure (%s): %.3f" % (label_name(label), metricz.fMeasure(label)))

print(line)
print('Ploting graphs..')

x = np.arange(len(names))
width = 0.3
fig, ax = plt.subplots()
rect = ax.bar(x, times, width, label = 'Execution Time (s)')
ax.set_ylabel('Execution Time (s)')
ax.set_title('Execution time of different classification algorithms')
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.legend()
fig.tight_layout()
plt.show()

for label in labels:
    precisions = list(map(lambda met: met.precision(label), metrics))
    recalls = list(map(lambda met: met.recall(label), metrics))
    fMeasures = list(map(lambda met: met.fMeasure(label), metrics))

    fig, ax = plt.subplots()
    
    metricsValues = [precisions, recalls, fMeasures]
    metricsNames = ['Precision', 'Recall', 'F-measure']
    metricsColors = ['green', 'orange', 'purple']
    for arr, metric, color in zip(metricsValues, metricsNames, metricsColors):
        pos = x + (metricsValues.index(arr) - 1) * width
        rect = ax.bar(pos, arr, width, label = '%s (%s)' % (metric, label_name(label)), color = color)

    ax.set_ylabel('Metric value')
    ax.set_title('Metrics (%s) of different classification algorithms' % label_name(label))
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    fig.tight_layout()
    plt.show()

print(line)
print('Done.')



