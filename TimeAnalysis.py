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

# COMMAND ----------

list(healthCombined)

# COMMAND ----------

healthCombined.ERBMI.value_counts(dropna = False), healthCombined.ERBMI.count()

# COMMAND ----------

np.float(healthCombined.loc[healthCombined.ERBMI.isin([-1])].ERBMI.count())/np.float(healthCombined.ERBMI.count())*100

# COMMAND ----------

# MAGIC %md
# MAGIC Less than 5 percent of -1(null) present, so removing those values

# COMMAND ----------

healthCombinedEdited = healthCombined.loc[healthCombined['ERBMI'] != -1]

# COMMAND ----------

healthCombined.EUPRPMEL.value_counts(dropna = False), healthCombined.EUPRPMEL.count()
#healthCombinedEdited.loc[healthCombinedEdited.tewhere == -3].tewhere

# COMMAND ----------

# MAGIC %md
# MAGIC Remove all nulls without replacement first. See the count of resulting output and decide for a different approach if the count is too less

# COMMAND ----------

healthCombinedEdited = healthCombinedEdited.loc[healthCombinedEdited['EUDIETSODA'].isin([1,2,3])]
healthCombinedEdited = healthCombinedEdited.loc[healthCombinedEdited['EUEXERCISE'].isin([1,2,])]
healthCombinedEdited = healthCombinedEdited.loc[healthCombinedEdited['EEINCOME1'].isin([1,2,3])]
healthCombinedEdited = healthCombinedEdited.loc[~healthCombinedEdited['EUEXFREQ'].isin([-1])]
healthCombinedEdited = healthCombinedEdited.loc[~healthCombinedEdited['EUFASTFD'].isin([-2, -3])]
healthCombinedEdited = healthCombinedEdited.loc[~healthCombinedEdited['EUFDSIT'].isin([-2])]
healthCombinedEdited = healthCombinedEdited.loc[~healthCombinedEdited['EUGENHTH'].isin([-2])]
healthCombinedEdited = healthCombinedEdited.loc[~healthCombinedEdited['EUMEAT'].isin([-1,-2])]
healthCombinedEdited = healthCombinedEdited.loc[~healthCombinedEdited['tewhere'].isin([-1])]

#'TRERNWA', 'EUFASTFDFRQ','EUMILK','EUEATSUM',
healthCombinedCleaned = healthCombinedEdited[['ERBMI', 'ERTPREAT', 'ERTSEAT', 'EUDIETSODA',  'EUEXERCISE', 'TEAGE',  'EEINCOME1', 'EUEXFREQ', 'EUFASTFD',  'EUFFYDAY', 'EUFDSIT', 'EUGENHTH'
                                             , 'EUGROSHP', 'EUMEAT',  'EUPRPMEL', 'TUACTIVITY_N',  'tuactdur24', 'tewhere', 'TESEX']]
healthCombinedCleaned.info()

# COMMAND ----------

# MAGIC %md
# MAGIC Enough data after removing null to continue with analysis

# COMMAND ----------

healthCombinedCleaned.describe()

# COMMAND ----------

# MAGIC %md 
# MAGIC Visualize to check if scaling required

# COMMAND ----------

plt.hist(healthCombinedCleaned.ERTPREAT)
display()

# COMMAND ----------

sns.jointplot( x = 'ERTPREAT', y = 'ERBMI', data = healthCombinedCleaned)
display()

# COMMAND ----------

plt.figure()
pair = sns.pairplot(healthCombinedCleaned[['ERBMI', 'ERTPREAT', 'ERTSEAT', 'EUDIETSODA',  'EUEXERCISE', 'TEAGE',  'EEINCOME1', 'EUEXFREQ', 'EUFASTFD',  'EUFFYDAY', 'EUFDSIT', 'EUGENHTH'
                                             , 'EUGROSHP', 'EUMEAT',  'EUPRPMEL', 'TUACTIVITY_N',  'tuactdur24', 'tewhere', 'TESEX']])
display()

# COMMAND ----------

# MAGIC %md
# MAGIC Convert pandas Dataframe to spark DataFrame for futher analysis

# COMMAND ----------

sqlContext = SQLContext(sc)

# COMMAND ----------

health_df = sqlContext.createDataFrame(healthCombinedCleaned)
display(health_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Split data in train and test for modeling. 70% train, 30% test

# COMMAND ----------

training, test = health_df.randomSplit([0.7,0.3],0)

# COMMAND ----------

training.count()

# COMMAND ----------

test.count()

# COMMAND ----------

# MAGIC %md
# MAGIC 2 Vector Assembler Pipeline stages   
# MAGIC 1 - With all features   
# MAGIC 2 - With just the intercept

# COMMAND ----------

vecScaled = feature.VectorAssembler(inputCols = [ 'ERTPREAT', 'ERTSEAT', 'EUDIETSODA',  'EUEXERCISE', 'TEAGE',  'EEINCOME1', 'EUEXFREQ', 'EUFASTFD',  'EUFFYDAY', 'EUFDSIT', 'EUGENHTH'
                                             , 'EUGROSHP', 'EUMEAT',  'EUPRPMEL', 'TUACTIVITY_N',  'tuactdur24', 'tewhere', 'TESEX'], outputCol = 'features')

# COMMAND ----------

vecIntercept = feature.VectorAssembler(inputCols=[], outputCol='emptyFeatures')

# COMMAND ----------

# MAGIC %md
# MAGIC Scaling stage to scale features from Vector Assembler

# COMMAND ----------

scaled = feature.StandardScaler(inputCol='features', outputCol='sclaedFeatures')

# COMMAND ----------

# MAGIC %md
# MAGIC Three Linear Regression Pipleline stage   
# MAGIC 1 - LR with just the intercept   
# MAGIC 2 - LR with all features unscaled   
# MAGIC 3 - LR with all features and scaled stage

# COMMAND ----------

regIntercept = regression.LinearRegression(labelCol= 'ERBMI', featuresCol= 'emptyFeatures')

# COMMAND ----------

regUnscaled = regression.LinearRegression(labelCol = 'ERBMI', featuresCol = 'features', regParam=0, elasticNetParam = 0)

# COMMAND ----------

regScaled = regression.LinearRegression(labelCol = 'ERBMI', featuresCol = 'sclaedFeatures', maxIter=5)

# COMMAND ----------

# MAGIC %md
# MAGIC 3 Piplelines for the different Linear Regression

# COMMAND ----------

pipeIntercept = Pipeline(stages = [vecIntercept, regIntercept])

# COMMAND ----------

PipeUnscaled = Pipeline(stages = [vecScaled, regUnscaled])

# COMMAND ----------

PipeScaled = Pipeline(stages = [vecScaled, scaled, regScaled])

# COMMAND ----------

# MAGIC %md
# MAGIC RMSE declartion for measuring model accuracy

# COMMAND ----------

rmse = fn.sqrt(fn.avg((fn.col('ERBMI') - fn.col('prediction'))**2))

# COMMAND ----------

# MAGIC %md
# MAGIC Starting with the Intercept model analysis

# COMMAND ----------

interceptModel = pipeIntercept.fit(training)

# COMMAND ----------

rmse = fn.sqrt(fn.avg((fn.col('ERBMI') - fn.col('prediction'))**2))

# COMMAND ----------

interceptModel.transform(test).select(rmse).show()

# COMMAND ----------

# MAGIC %md
# MAGIC RMSE of 5.71 with just the intercept   
# MAGIC Trying to LR with unscaled features

# COMMAND ----------

PipeUnscaled = Pipeline(stages = [vecScaled, regUnscaled])

# COMMAND ----------

unScaledModel = PipeUnscaled.fit(training)

# COMMAND ----------

unScaledModel.transform(test).select(rmse).show()

# COMMAND ----------

# MAGIC %md
# MAGIC Observed a reduced RMSE with the features   
# MAGIC Now, scaling the features before fitting to the model

# COMMAND ----------

PipeScaled = Pipeline(stages = [vecScaled, scaled, regScaled])

# COMMAND ----------

scaledModel = PipeScaled.fit(training)

# COMMAND ----------

scaledModel.transform(test).select(rmse).show()

# COMMAND ----------

# MAGIC %md
# MAGIC Scaled features resulted with a very small increase in the intercept

# COMMAND ----------

# MAGIC %md
# MAGIC Analysing coefficeints of scaled and unscaled model

# COMMAND ----------

linModelUnscaled = unScaledModel.stages[-1]

# COMMAND ----------

linModelUnscaled.coefficients

# COMMAND ----------

linModelScaled = scaledModel.stages[-1]

# COMMAND ----------

linModelScaled.coefficients

# COMMAND ----------

# MAGIC %md
# MAGIC Adding values to DataFrame for plotting the coefficients

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
valuesDF.columns

# COMMAND ----------

# MAGIC %md
# MAGIC Plotting scaled model with seaborn

# COMMAND ----------

plt.figure()
fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
sns.barplot( y = 'Scaled', x = 'feature', data = valuesDF)
plt.xticks(rotation = 60)
display()

# COMMAND ----------

# MAGIC %md
# MAGIC Plotting unscaled model with seaborn

# COMMAND ----------

plt.figure()
fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
sns.barplot( y = 'notScaled', x = 'feature', data = valuesDF)
plt.xticks(rotation = 60)
display()

# COMMAND ----------

# MAGIC %md
# MAGIC Plotting scaled and unsacled for comparison

# COMMAND ----------

indexDf = valuesDF.set_index('feature')
indexDf

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

# MAGIC %md
# MAGIC Applying regularization to further improve our RMSE   
# MAGIC First step to build a grid of two parameters - ElasticNetRegularization

# COMMAND ----------

grid = tune.ParamGridBuilder()

# COMMAND ----------

grid = grid.addGrid(regScaled.elasticNetParam, [0, 0.2, 0.4, 0.6, 0.8, 1])

# COMMAND ----------

grid = grid.addGrid(regScaled.regParam, np.arange(0,.1,.01))

# COMMAND ----------

grid = grid.build()

# COMMAND ----------

# MAGIC %md
# MAGIC Defining evalutor for Cross Validation

# COMMAND ----------

evaluator = RegressionEvaluator(labelCol=regScaled.getLabelCol(), predictionCol=regScaled.getPredictionCol())

# COMMAND ----------

# MAGIC %md
# MAGIC Import cross validator, cross validator model and rand to build a custome function CrossValidatorVerbose on top of CrossValidator

# COMMAND ----------

from pyspark.sql.functions import rand
from pyspark.ml.tuning import CrossValidator, CrossValidatorModel

# COMMAND ----------

a = list()
b = list()
c = list()

# COMMAND ----------

# MAGIC %md
# MAGIC CrossValidatorVerbose builds on top of Cross Validator by displaying and storing out from each fold and all the regularization parameters   
# MAGIC These values then can be used to compare the models with different regularizations

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

# MAGIC %md
# MAGIC Passing the pipeline, grid of HyperParameters, and evaluator to Cross Validation with three folds

# COMMAND ----------

cvVer = CrossValidatorVerbose(estimator = PipeScaled, estimatorParamMaps = grid, evaluator= evaluator, numFolds = 3)

# COMMAND ----------

varStore = cvVer.fit(training)

# COMMAND ----------

testStore = varStore.transform(test)

# COMMAND ----------

testStore.select(rmse).show()

# COMMAND ----------

# MAGIC %md
# MAGIC RMSE reduced by a very low value.   
# MAGIC Analyzing models from cross validation

# COMMAND ----------

# MAGIC %md
# MAGIC Extract all models from Cross Validation

# COMMAND ----------

linearDict = {}
for i in range(0, len(a), 2):
  linearDict[a[i] + " " + `b[i]` + " " + a[i + 1] + " " + `b[i + 1]`] = c[i]

# COMMAND ----------

# MAGIC %md 
# MAGIC Sort the models with lowest rmse first

# COMMAND ----------

for key, value in sorted(linearDict.iteritems(), key=lambda (k,v): (v,k)):
    print "%s: %s" % (key, value)

# COMMAND ----------

# MAGIC %md
# MAGIC Extract the Best Model from cross validation

# COMMAND ----------

BestModel = varStore.bestModel.stages[-1]
BestModel

# COMMAND ----------

BestModel.coefficients

# COMMAND ----------

# MAGIC %md
# MAGIC Add Coefficients from best model to the DataFrame for plotting

# COMMAND ----------

indexDf['BestScaled'] = BestModel.coefficients

# COMMAND ----------

valuesDF['BestScaled'] = BestModel.coefficients

# COMMAND ----------

# MAGIC %md
# MAGIC Comparing coefficients of Best Model vs Model with regularization

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

# MAGIC %md
# MAGIC Coefficients of best model

# COMMAND ----------

plt.figure()
fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
sns.barplot( y = 'BestScaled', x = 'feature', data = valuesDF)
plt.xticks(rotation = 60)
display()

# COMMAND ----------

BestModel.intercept

# COMMAND ----------

i = 0
top5SortedDict = {}
for key, value in sorted(linearDict.iteritems(), key=lambda (k,v): (v,k)):
  if(i<3):
    top5SortedDict[key] = value
  i = i + 1
top5SortedDict['Intercept'] = evaluator.evaluate(interceptModel.transform(test))
top5SortedDict['UnscaledModel'] = evaluator.evaluate(unScaledModel.transform(test))
top5SortedDict

# COMMAND ----------

cfplt.figure()
fig=plt.figure(figsize=(20, 9), dpi= 80, facecolor='w', edgecolor='k')
plt.bar(range(len(top5SortedDict)), list(top5SortedDict.values()), align='center')
plt.xticks(range(len(top5SortedDict)), list(top5SortedDict.keys()))
plt.xticks(rotation = 60)
plt.ylabel("RMSE")
plt.ylim([min(top5SortedDict.values()) - 0.2 , max(top5SortedDict.values()) + 0.2])
display()

# COMMAND ----------

# MAGIC %md
# MAGIC Running Regression with Random Forest Regression

# COMMAND ----------

# MAGIC %md
# MAGIC Pipleline stage for Random forest regression with scaled features

# COMMAND ----------

rfRegression = regression.RandomForestRegressor(featuresCol='sclaedFeatures', labelCol='ERBMI')

# COMMAND ----------

# MAGIC %md
# MAGIC Building grid with hyper-parameters for Randorm forest.   
# MAGIC Grid for chossing the number of trees and the maximum depth of each tree

# COMMAND ----------

gridRandom = tune.ParamGridBuilder()

# COMMAND ----------

gridRandom = gridRandom.addGrid(rfRegression.numTrees, [2,4,5,8])

# COMMAND ----------

gridRandom = gridRandom.addGrid(rfRegression.maxDepth, [2,3,4,5,6])

# COMMAND ----------

gridRandom = gridRandom.build()

# COMMAND ----------

pipeRandom = Pipeline(stages = [vecScaled, scaled, rfRegression])

# COMMAND ----------

randomModel = pipeRandom.fit(training)

# COMMAND ----------

randomTested = randomModel.transform(test)

# COMMAND ----------

# MAGIC %md
# MAGIC Defining evaluator for Random Forest

# COMMAND ----------

evaluatorRandom = RegressionEvaluator(labelCol= rfRegression.getLabelCol() , predictionCol= rfRegression.getPredictionCol())

# COMMAND ----------

evaluatorRandom.evaluate(randomTested)

# COMMAND ----------

# MAGIC %md
# MAGIC A much reduced rmse with Random Forest Regression

# COMMAND ----------

# MAGIC %md
# MAGIC Extracting feature impoartances for the variables passed on

# COMMAND ----------

randomStage = randomModel.stages[-1]

# COMMAND ----------

randomStage.featureImportances

# COMMAND ----------

# MAGIC %md 
# MAGIC Adding feature importances to DataFrame for plotting

# COMMAND ----------

valuesDF['randomForestFeatures'] = randomStage.featureImportances

# COMMAND ----------

indexDf = valuesDF.set_index('feature')
indexDf

# COMMAND ----------

# MAGIC %md
# MAGIC IS THERE A POINT TO THIS????

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

# MAGIC %md
# MAGIC Passing Random forest through cross validation with 4 folds

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

# MAGIC %md
# MAGIC Evalating best Random forest results in a lower rmse

# COMMAND ----------

# MAGIC %md
# MAGIC After hyper parametrizing we observe further reduction in the RMSE to 4.46

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
    top5SortedDict[key] = value
  i = i + 1
top5SortedDict

# COMMAND ----------

plt.figure()
fig=plt.figure(figsize=(20, 9), dpi= 80, facecolor='w', edgecolor='k')
plt.bar(range(len(top5SortedDict)), list(top5SortedDict.values()), align='center')
plt.xticks(range(len(top5SortedDict)), list(top5SortedDict.keys()))
plt.xticks(rotation = 60)
plt.ylim([min(top5SortedDict.values()) - 0.2 , max(top5SortedDict.values()) + 0.2])
display()

# COMMAND ----------

evaluatorRandom.evaluate(randomFit.bestModel.transform(test))/min(randomFit.avgMetrics)

# COMMAND ----------

# MAGIC %md 
# MAGIC Plotting top 5 Randform forest model

# COMMAND ----------

dfa.plot.scatter('prediction', 're')
display() 

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
