# Databricks notebook source
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import linear_model 
import numpy as np
from pyspark.sql import functions as fn
from pyspark.ml import Pipeline
from pyspark.ml import regression
from pyspark.ml import feature
from pyspark.sql import SQLContext
import pyspark.ml.tuning as tune
from pyspark.ml.evaluation import RegressionEvaluator

# COMMAND ----------

# MAGIC %sh wget https://www.dropbox.com/s/9xf7gwlal3dj4bv/HEALTHRESPONACTISUMMARYMERGED.csv?dl=1 -O combinedHealth.csv

# COMMAND ----------

healthCombined = pd.read_csv('combinedHealth.csv')
healthCombined.head()
#healthCombined = spark.read.csv('file:///databricks/driver/combinedHealth.csv')

# COMMAND ----------

list(healthCombined)

# COMMAND ----------

healthCombinedEdited = healthCombined.loc[healthCombined['ERBMI'] != -1]

# COMMAND ----------

healthCombined.EUPRPMEL.value_counts(dropna = False), healthCombined.EUPRPMEL.count()
#healthCombinedEdited.loc[healthCombinedEdited.tewhere == -3].tewhere

# COMMAND ----------

healthCombinedEdited = healthCombinedEdited.loc[healthCombinedEdited['EUDIETSODA'].isin([1,2,3])]
#healthCombinedEdited = healthCombinedEdited.loc[~healthCombinedEdited['TRERNWA'].isin([-1])]
healthCombinedEdited = healthCombinedEdited.loc[healthCombinedEdited['EUEXERCISE'].isin([1,2,])]
healthCombinedEdited = healthCombinedEdited.loc[healthCombinedEdited['EEINCOME1'].isin([1,2,3])]
healthCombinedEdited = healthCombinedEdited.loc[~healthCombinedEdited['EUEXFREQ'].isin([-1])]
healthCombinedEdited = healthCombinedEdited.loc[~healthCombinedEdited['EUFASTFD'].isin([-2, -3])]
#healthCombinedEdited = healthCombinedEdited.loc[~healthCombinedEdited['EUFASTFDFRQ'].isin([-1,-2])]
healthCombinedEdited = healthCombinedEdited.loc[~healthCombinedEdited['EUFDSIT'].isin([-2])]
healthCombinedEdited = healthCombinedEdited.loc[~healthCombinedEdited['EUGENHTH'].isin([-2])]
healthCombinedEdited = healthCombinedEdited.loc[~healthCombinedEdited['EUMEAT'].isin([-1,-2])]
#healthCombinedEdited = healthCombinedEdited.loc[~healthCombinedEdited['EUMILK'].isin([-1])]
#healthCombinedEdited = healthCombinedEdited.loc[~healthCombinedEdited['EUEATSUM'].isin([-1])]
healthCombinedEdited = healthCombinedEdited.loc[~healthCombinedEdited['tewhere'].isin([-1])]

#'TRERNWA', 'EUFASTFDFRQ','EUMILK','EUEATSUM',
healthCombinedCleaned = healthCombinedEdited[['ERBMI', 'ERTPREAT', 'ERTSEAT', 'EUDIETSODA',  'EUEXERCISE', 'TEAGE',  'EEINCOME1', 'EUEXFREQ', 'EUFASTFD',  'EUFFYDAY', 'EUFDSIT', 'EUGENHTH'
                                             , 'EUGROSHP', 'EUMEAT',  'EUPRPMEL', 'TUACTIVITY_N',  'tuactdur24', 'tewhere', 'TESEX']]
healthCombinedCleaned.info()

# COMMAND ----------

 healthCombined.loc[healthCombined['EUPRPMEL'].isin([1,2,3])]

# COMMAND ----------

cleanedPlot = healthCombined.loc[healthCombined['EUPRPMEL'].isin([1,2,3])]

# COMMAND ----------

cleanedPlot

# COMMAND ----------

cleanedPlot = healthCombined.loc[healthCombined['EUPRPMEL'].isin([1,2,3])]

plt.figure()
fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')

ax = fig.add_subplot(111) 
ax2 = ax.twinx() 


width = 0.4
cleanedPlot.BestScaled.plot(kind='bar', color='red', ax=ax, width=width, position=0, legend = True)
cleanedPlot.Scaled.plot(kind='bar', color='blue', ax=ax, width=width, position=1, legend = True)

display()

# COMMAND ----------

healthCombinedCleaned.describe()

# COMMAND ----------

plt.hist(healthCombinedCleaned.ERTPREAT)
display()

# COMMAND ----------

sns.jointplot( x = 'ERTPREAT', y = 'ERBMI', data = healthCombinedCleaned)
display()

# COMMAND ----------

sqlContext = SQLContext(sc)

# COMMAND ----------

new_df = sqlContext.createDataFrame(healthCombinedCleaned)
display(new_df)

# COMMAND ----------

plt.figure()
pair = sns.pairplot(new_df.toPandas()[['ERBMI', 'ERTPREAT', 'ERTSEAT', 'EUDIETSODA',  'EUEXERCISE', 'TEAGE', 'TRERNWA', 'EEINCOME1', 'EUEXFREQ', 'EUFASTFD', 'EUFASTFDFRQ', 'EUFFYDAY', 'EUFDSIT', 'EUGENHTH'
                                             , 'EUGROSHP', 'EUMEAT', 'EUMILK', 'EUPRPMEL', 'TUACTIVITY_N', 'EUEATSUM', 'tuactdur24', 'tewhere', 'TESEX']])
display()

# COMMAND ----------

training, test = new_df.randomSplit([0.7,0.3],0)

# COMMAND ----------

training.count()

# COMMAND ----------

test.count()

# COMMAND ----------

vecScaled = feature.VectorAssembler(inputCols = [ 'ERTPREAT', 'ERTSEAT', 'EUDIETSODA',  'EUEXERCISE', 'TEAGE',  'EEINCOME1', 'EUEXFREQ', 'EUFASTFD',  'EUFFYDAY', 'EUFDSIT', 'EUGENHTH'
                                             , 'EUGROSHP', 'EUMEAT',  'EUPRPMEL', 'TUACTIVITY_N',  'tuactdur24', 'tewhere', 'TESEX'], outputCol = 'features')

# COMMAND ----------

scaled = feature.StandardScaler(inputCol='features', outputCol='sclaedFeatures')

# COMMAND ----------

regScaled = regression.LinearRegression(labelCol = 'ERBMI', featuresCol = 'sclaedFeatures', maxIter=5)

# COMMAND ----------

regUnscaled = regression.LinearRegression(labelCol = 'ERBMI', featuresCol = 'features', regParam=0, elasticNetParam = 0)

# COMMAND ----------

vecIntercept = feature.VectorAssembler(inputCols=[], outputCol='emptyFeatures')

# COMMAND ----------

regIntercept = regression.LinearRegression(labelCol= 'ERBMI', featuresCol= 'emptyFeatures')

# COMMAND ----------

pipeIntercept = Pipeline(stages = [vecIntercept, regIntercept])

# COMMAND ----------

interceptModel = pipeIntercept.fit(training)

# COMMAND ----------

rmse = fn.sqrt(fn.avg((fn.col('ERBMI') - fn.col('prediction'))**2))

# COMMAND ----------

interceptModel.transform(test).select(rmse).show()

# COMMAND ----------

PipeUnscaled = Pipeline(stages = [vecScaled, regUnscaled])

# COMMAND ----------

PipeScaled = Pipeline(stages = [vecScaled, scaled, regScaled])

# COMMAND ----------

unScaledModel = PipeUnscaled.fit(training)

# COMMAND ----------

unScaledModel.transform(test).select(rmse).show()

# COMMAND ----------

linModelUnscaled = unScaledModel.stages[-1]

# COMMAND ----------

scaledModel = PipeScaled.fit(training)

# COMMAND ----------

scaledModel.transform(test).select(rmse).show()

# COMMAND ----------

linModelUnscaled.coefficients

# COMMAND ----------

linModelScaled.coefficients

# COMMAND ----------

valuesDF = pd.DataFrame( healthCombinedCleaned.drop('ERBMI', axis = 1).columns)

# COMMAND ----------

valuesDF['NotScaled'] = linModelUnscaled.coefficients

# COMMAND ----------

valuesDF['Scaled'] = linModelScaled.coefficients

# COMMAND ----------

valuesDF

# COMMAND ----------

valuesDF.columns = ['feature', 'notScaled', 'Scaled']

# COMMAND ----------

valuesDF.columns

# COMMAND ----------

plt.figure()
fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
sns.barplot( y = 'Scaled', x = 'feature', data = valuesDF)
plt.xticks(rotation = 60)
display()

# COMMAND ----------

plt.figure()
fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
sns.barplot( y = 'notScaled', x = 'feature', data = valuesDF)
plt.xticks(rotation = 60)
display()

# COMMAND ----------

plt.figure()
fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')

ax = fig.add_subplot(111) 
ax2 = ax.twinx() 

width = 0.4
indexDf.notScaled.plot(kind='bar', color='red', ax=ax, width=width, position=0, legend = True)
indexDf.Scaled.plot(kind='bar', color='blue', ax=ax, width=width, position=1, legend = True)

display()

# COMMAND ----------

#plt.figure()
#fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
#
#ax = fig.add_subplot(111) 
#ax2 = ax.twinx() 

#width = 0.4
#indexDf.abs().notScaled.plot(kind='bar', color='red', ax=ax, width=width, position=0, legend = True)
#indexDf.abs().Scaled.plot(kind='bar', color='blue', ax=ax, width=width, position=1, legend = True)

#display()

# COMMAND ----------

grid = tune.ParamGridBuilder()

# COMMAND ----------

grid = grid.addGrid(regScaled.elasticNetParam, [0, 0.2, 0.4, 0.6, 0.8, 1])

# COMMAND ----------

grid = grid.addGrid(regScaled.regParam, np.arange(0,.1,.01))

# COMMAND ----------

grid = grid.build()

# COMMAND ----------

evaluator = RegressionEvaluator(labelCol=regScaled.getLabelCol(), predictionCol=regScaled.getPredictionCol())

# COMMAND ----------

crossPipe = Pipeline(stages = [vecScaled, scaled, regScaled])

# COMMAND ----------

#cv = tune.CrossValidator(estimator = crossPipe, estimatorParamMaps = grid, evaluator= evaluator, numFolds = 3)

# COMMAND ----------

from pyspark.sql.functions import rand
from pyspark.ml.tuning import CrossValidator, CrossValidatorModel

# COMMAND ----------

a = list()
b = list()
c = list()

# COMMAND ----------

class CrossValidatorVerbose(CrossValidator):
    
    def _fit(self, dataset):
        est = self.getOrDefault(self.estimator)
        epm = self.getOrDefault(self.estimatorParamMaps)
        numModels = len(epm)
        eva = self.getOrDefault(self.evaluator)
        metricName = eva.getMetricName()

        nFolds = self.getOrDefault(self.numFolds)
        seed = self.getOrDefault(self.seed)
        h = 1.0 / nFolds

        randCol = self.uid + "_rand"
        df = dataset.select("*", rand(seed).alias(randCol))
        metrics = [0.0] * numModels

        for i in range(nFolds):
            foldNum = i + 1
            print("Comparing models on fold %d" % foldNum)

            validateLB = i * h
            validateUB = (i + 1) * h
            condition = (df[randCol] >= validateLB) & (df[randCol] < validateUB)
            validation = df.filter(condition)
            train = df.filter(~condition)

            for j in range(numModels):
                paramMap = epm[j]
                model = est.fit(train, paramMap)
                # TODO: duplicate evaluator to take extra params from input
                metric = eva.evaluate(model.transform(validation, paramMap))
                metrics[j] += metric

                avgSoFar = metrics[j] / foldNum
                print("params: %s\t%s: %f\tavg: %f" % (  
                  {param.name: val for (param, val) in paramMap.items()},
                    metricName, metric, avgSoFar))

                for (param, val) in paramMap.items():
                  a.append(param.name)
                  b.append(val)
                  c.append(metric)
                
        if eva.isLargerBetter():
            bestIndex = np.argmax(metrics)
        else:
            bestIndex = np.argmin(metrics)

        bestParams = epm[bestIndex]
        bestModel = est.fit(dataset, bestParams)
        avgMetrics = [m / nFolds for m in metrics]
        bestAvg = avgMetrics[bestIndex]
        print("Best model:\nparams: %s\t%s: %f" % (
            {param.name: val for (param, val) in bestParams.items()},
            metricName, bestAvg))
        
        return self._copyValues(CrossValidatorModel(bestModel, avgMetrics))

# COMMAND ----------

cvVer = CrossValidatorVerbose(estimator = crossPipe, estimatorParamMaps = grid, evaluator= evaluator, numFolds = 3)

# COMMAND ----------

varStore = cvVer.fit(training)

# COMMAND ----------

testStore = varStore.transform(test)

# COMMAND ----------

testStore.select(rmse).show()

# COMMAND ----------

linearDict = {}
for i in range(0, len(a), 2):
  linearDict[a[i] + " " + `b[i]` + " " + a[i + 1] + " " + `b[i + 1]`] = c[i]

# COMMAND ----------

for key, value in sorted(linearDict.iteritems(), key=lambda (k,v): (v,k)):
    print "%s: %s" % (key, value)

# COMMAND ----------

evaluator.evaluate(testStore)

# COMMAND ----------

BestModel = testStore.bestModel.stages[-1]
BestModel

# COMMAND ----------

BestModel.coefficients

# COMMAND ----------

indexDf['BestScaled'] = BestModel.coefficients

# COMMAND ----------

valuesDF['BestScaled'] = BestModel.coefficients

# COMMAND ----------

indexDf

# COMMAND ----------

plt.figure()
fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')

ax = fig.add_subplot(111) 
ax2 = ax.twinx() 

width = 0.4
indexDf.BestScaled.plot(kind='bar', color='red', ax=ax, width=width, position=0, legend = True)
indexDf.Scaled.plot(kind='bar', color='blue', ax=ax, width=width, position=1, legend = True)

display()

# COMMAND ----------

plt.figure()
fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
sns.barplot( y = 'BestScaled', x = 'feature', data = valuesDF)
plt.xticks(rotation = 60)
display()

# COMMAND ----------

BestModel.hasSummary

# COMMAND ----------

BestModel.intercept

# COMMAND ----------

BestModel.summary.meanSquaredError

# COMMAND ----------

BestModel.summary.r2

# COMMAND ----------

BestModel.summary.pValues

# COMMAND ----------

BestModel.summary.rootMeanSquaredError

# COMMAND ----------

BestModel.summary.residuals.count()

# COMMAND ----------

BestModel.params

# COMMAND ----------

BestModel.summary.predictions.selectExpr('cast(prediction as float) pre').collect()

# COMMAND ----------

BestModel.summary.predictions.select('prediction')

# COMMAND ----------

BestModel.summary.residuals.selectExpr('cast(residuals as float) res').show()

# COMMAND ----------

BestModel

# COMMAND ----------

BestModel.summary.predictionCol

# COMMAND ----------

pred = BestModel.summary.predictions.toPandas()

# COMMAND ----------

type(pred)

# COMMAND ----------

gridRandom = tune.ParamGridBuilder()

# COMMAND ----------

gridRandom = gridRandom.addGrid(rfRegression.numTrees, [2,4,5,8])

# COMMAND ----------

gridRandom = gridRandom.addGrid(rfRegression.maxDepth, [2,3,4,5,6])

# COMMAND ----------

gridRandom = gridRandom.build()

# COMMAND ----------

rfRegression = regression.RandomForestRegressor(featuresCol='sclaedFeatures', labelCol='ERBMI')

# COMMAND ----------

pipeRandom = Pipeline(stages = [vecScaled, scaled, rfRegression])

# COMMAND ----------

randomModel = pipeRandom.fit(training)

# COMMAND ----------

randomTested = randomModel.transform(test)

# COMMAND ----------

evaluatorRandom = RegressionEvaluator(labelCol= rfRegression.getLabelCol() , predictionCol= rfRegression.getPredictionCol())

# COMMAND ----------

evaluatorRandom.evaluate(randomTested)

# COMMAND ----------

randomStage = randomModel.stages[-1]

# COMMAND ----------

randomStage.featureImportances

# COMMAND ----------

featuresImportances

# COMMAND ----------

valuesDF['randomForestFeatures'] = randomStage.featureImportances

# COMMAND ----------

indexDf = valuesDF.set_index('feature')
indexDf

# COMMAND ----------

plt.figure()
fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')

ax = fig.add_subplot(111) 
ax2 = ax.twinx() 

width = 0.4
indexDf.abs().Scaled.plot(kind='bar', color='red', ax=ax, width=width, position=0, legend = True)
indexDf.randomForestFeatures.plot(kind='bar', color='blue', ax=ax, width=width, position=1, legend = True)

display()

# COMMAND ----------

cvVrRandom = CrossValidator(estimator=pipeRandom, estimatorParamMaps=gridRandom, evaluator=evaluatorRandom)

# COMMAND ----------

cvVrRandom.fit(training)

# COMMAND ----------

cvVdVerboseRandom = CrossValidatorVerbose(estimator=pipeRandom, estimatorParamMaps=gridRandom, evaluator=evaluatorRandom, numFolds=4)

# COMMAND ----------

a = list()
b = list()
c = list()

# COMMAND ----------

randomFit = cvVdVerboseRandom.fit(training)

# COMMAND ----------

randomDict = {}
for i in range(0, len(a), 2):
  randomDict[a[i] + " " + `b[i]` + " " + a[i + 1] + " " + `b[i + 1]`] = c[i]

# COMMAND ----------

for key, value in sorted(randomDict.iteritems(), key=lambda (k,v): (v,k)):
    print "%s: %s" % (key, value)

# COMMAND ----------

randomTransform = randomFit.transform(test)

# COMMAND ----------

randomForestBestModel = randomFit.bestModel

# COMMAND ----------

evaluatorRandom.evaluate(randomFit.bestModel.transform(test))

# COMMAND ----------

randomForestRegressionModel = randomForestBestModel.stages[-1]

# COMMAND ----------

bestModelFeatures = randomForestRegressionModel.featureImportances

# COMMAND ----------

valuesDF['randomForestBestFeatures'] = randomForestRegressionModel.featureImportances

# COMMAND ----------

indexDf = valuesDF.set_index('feature')
indexDf

# COMMAND ----------

plt.figure()
fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')

ax = fig.add_subplot(111) 
ax2 = ax.twinx() 

width = 0.4
indexDf.randomForestBestFeatures.plot(kind='bar', color='red', ax=ax, width=width, position=0, legend = True)
indexDf.randomForestFeatures.plot(kind='bar', color='blue', ax=ax, width=width, position=1, legend = True)

display()

# COMMAND ----------

plt.figure()
fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
sns.barplot( y = 'randomForestBestFeatures', x = 'feature', data = valuesDF)
plt.xticks(rotation = 60)
display()

# COMMAND ----------

i = 0
top5SortedDict = {}
for key, value in sorted(randomDict.iteritems(), key=lambda (k,v): (v,k)):
  if(i<5):
    #print(i)
    #print "%s: %s" % (key, value)
    top5SortedDict[key] = value
  i = i + 1
  #print "%s: %s" % (key, value)
top5SortedDict

# COMMAND ----------

plt.figure()
fig=plt.figure(figsize=(20, 9), dpi= 80, facecolor='w', edgecolor='k')
plt.bar(range(len(top5SortedDict)), list(top5SortedDict.values()), align='center')
plt.xticks(range(len(top5SortedDict)), list(top5SortedDict.keys()))
plt.xticks(rotation = 60)
display()

# COMMAND ----------

evaluatorRandom.evaluate(randomFit.bestModel.transform(test))/min(randomFit.avgMetrics)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

resd = BestModel.summary.residuals.toPandas()

# COMMAND ----------

type(resd)

# COMMAND ----------

b = resd.residuals

# COMMAND ----------

dfa = pd.DataFrame(data = a)

# COMMAND ----------

dfa.columns

# COMMAND ----------

dfa.dtypes

# COMMAND ----------

dfa['re'] = b

# COMMAND ----------

dfa.prediction

# COMMAND ----------

dfa.plot.scatter('prediction', 're')
display() 

# COMMAND ----------

dfa

# COMMAND ----------

dfa.describe()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

rmse = fn.sqrt(fn.avg((fn.col('ERBMI') - fn.col('prediction'))**2))

# COMMAND ----------

testModel.transform(test).select(rmse).show()

# COMMAND ----------

testModel.transform(test).show(500)

# COMMAND ----------

testModel.stages[1].coefficients

# COMMAND ----------

testModel.stages[1].intercept

# COMMAND ----------

testModel.stages[1].

# COMMAND ----------

healthCombinedCleaned = healthCombinedEdited.loc[~healthCombinedEdited['TRERNWA'].isin([-1])]
healthCombinedCleaned.TRERNWA.value_counts(dropna = False)

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(healthCombinedCleaned.drop('ERBMI', axis = 1), healthCombinedCleaned.ERBMI, test_size = 0.3, random_state = 21 )

# COMMAND ----------

reg = linear_model.LinearRegression()

# COMMAND ----------

reg.fit(X_train, y_train)

# COMMAND ----------

reg.score(X_test, y_test)

# COMMAND ----------

healthCombinedEdited.head()

# COMMAND ----------

healthCombinedEdited.ERINCOME.value_counts()

# COMMAND ----------

healthCombinedEdited[healthCombinedEdited.ERINCOME]

# COMMAND ----------

mostEating = healthCombinedEdited.ERTSEAT.value_counts(dropna = False)[0:25].index.tolist()
tryingEating = healthCombinedEdited.loc[healthCombinedEdited['ERTSEAT'].isin(mostEating)]

# COMMAND ----------

healthCombinedEdited.loc[healthCombinedEdited['ERTSEAT'].isin(mostEating)].ERTSEAT.value_counts(dropna = False)

# COMMAND ----------

fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
sns.regplot( x = tryingEating["ERTSEAT"], y = tryingEating["ERBMI"], fit_reg = False)
display()

# COMMAND ----------

fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
sns.boxplot( x = "ERTSEAT", y = "ERBMI", data = tryingEating)
display()

# COMMAND ----------

fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
tryingExer = healthCombinedEdited.loc[healthCombinedEdited['EUEXERCISE'].isin([1,2])]
sns.boxplot( x = "EUEXERCISE", y = "ERBMI", data = tryingExer)
display()

# COMMAND ----------

fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
sns.boxplot( x = "TESEX", y = "ERBMI", data = healthCombinedEdited)
display()

# COMMAND ----------

fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
sns.boxplot(x = 'EEINCOME1', y =  "ERBMI", data = healthCombinedEdited.loc[healthCombinedEdited['EEINCOME1'].isin([1,2, 3])])
display()

# COMMAND ----------

healthCombinedEdited.ERTPREAT.max()

# COMMAND ----------

labels = ['Very High', 'High', 'Med', 'Low', 'Very Low']
labels

# COMMAND ----------

healthCombinedEdited['TimeSecondaryEating'] = pd.cut(healthCombinedEdited.ERTPREAT, 5, right = False, labels = labels)

# COMMAND ----------

healthCombinedEdited['TimeSecondaryEating'].value_counts()

# COMMAND ----------

fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
sns.boxplot(x = 'TimeSecondaryEating', y =  "ERBMI", data = healthCombinedEdited)
display()

# COMMAND ----------

fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
sns.boxplot(x = 'EUDIETSODA', y =  "ERBMI", data = healthCombinedEdited.loc[healthCombinedEdited['EUDIETSODA'].isin([1,2, 3])])
display()

# COMMAND ----------

fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
sns.boxplot(x = 'EUDRINK', y =  "ERBMI", data = healthCombinedEdited.loc[healthCombinedEdited['EUDRINK'].isin([1,2])])
display()

# COMMAND ----------

healthCombinedEdited.EUFASTFD.value_counts()
fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
sns.boxplot(x = 'EUFASTFD', y =  "ERBMI", data = healthCombinedEdited.loc[healthCombinedEdited['EUFASTFD'].isin([1,2])])
display()

# COMMAND ----------

healthCombinedEdited.EUFASTFDFRQ.value_counts()
healthCombinedEdited['FastFoodFrequrency'] = pd.cut(healthCombinedEdited.EUFASTFDFRQ, 5, right = False, labels = labels)
fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
sns.boxplot(x = 'FastFoodFrequrency', y =  "ERBMI", data = healthCombinedEdited)
display()

# COMMAND ----------

healthCombinedEdited.EUGENHTH.value_counts(dropna = False)
fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
sns.boxplot(x = 'EUGENHTH', y =  "ERBMI", data = healthCombinedEdited.loc[healthCombinedEdited['EUGENHTH'].isin([1,2,3,4,5])])
display()

# COMMAND ----------

healthCombinedEdited.TESCHENR.value_counts(dropna = False)
fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
sns.boxplot(x = 'TESCHENR', y =  "ERBMI", data = healthCombinedEdited.loc[healthCombinedEdited['TESCHENR'].isin([1,2])])
display()

# COMMAND ----------

healthCombinedEdited.EUFASTFDFRQ.value_counts()
healthCombinedEdited['FastFoodFrequrency'] = pd.cut(healthCombinedEdited.EUFASTFDFRQ, 5, right = False, labels = labels)
fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
sns.boxplot(x = 'FastFoodFrequrency', y =  "ERBMI", data = healthCombinedEdited)
display()

# COMMAND ----------

tempDF = healthCombinedEdited.loc[healthCombinedEdited['TRERNWA'] != -1]
tempDF['Earnings'] = pd.cut(tempDF.TRERNWA, 5, right = False, labels = labels)
fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
sns.boxplot(x = 'Earnings', y =  "ERBMI", data = tempDF)
display()
