# Databricks notebook source
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pyspark.sql import functions as fn
from pyspark.ml import feature
from pyspark.sql import SQLContext

# COMMAND ----------

# MAGIC %sh wget https://www.dropbox.com/s/9xf7gwlal3dj4bv/HEALTHRESPONACTISUMMARYMERGED.csv?dl=1 -O combinedHealth.csv

# COMMAND ----------

healthCombined = pd.read_csv('combinedHealth.csv')
healthCombined.head()

# COMMAND ----------

healthCombined.tail()

# COMMAND ----------

sqlContext = SQLContext(sc)

# COMMAND ----------

health_df = sqlContext.createDataFrame(healthCombined)

# COMMAND ----------

display(health_df)

# COMMAND ----------

healthCombined.info()

# COMMAND ----------

healthCombined['year'] = (healthCombined.TUCASEID.apply(str).str)[:4]

# COMMAND ----------

healthCombined['month'] = (healthCombined.TUCASEID.apply(str).str)[4:6]

# COMMAND ----------

healthCombined.info()

# COMMAND ----------

healthCombined.columns

# COMMAND ----------

healthActivityDuration = healthCombined[[ 'year', 'EUEXFREQ', 'month']]

# COMMAND ----------

healthActivityDuration.head()

# COMMAND ----------

healthActivityDuration.tucumdur24.value_counts(dropna = False)

# COMMAND ----------

healthActivityDuration.month.value_counts(dropna = False)

# COMMAND ----------

healthActivityDuration14 = healthActivityDuration.loc[healthActivityDuration.year == '2014']

# COMMAND ----------

healthActivityDuration15 = healthActivityDuration.loc[healthActivityDuration.year == '2015']

# COMMAND ----------

healthActivityDuration14.tail()

# COMMAND ----------

healthActivityDuration14 = healthActivityDuration14.groupby(['month']).mean()

# COMMAND ----------

healthActivityDuration15 = healthActivityDuration15.groupby(['month']).mean()

# COMMAND ----------

healthActivityDurationCombinedYear = healthActivityDuration.groupby(['month']).mean()

# COMMAND ----------

healthActivityDuration15

# COMMAND ----------

healthActivityDuration14.info()

# COMMAND ----------

healthActivityDuration14.plot()
display()

# COMMAND ----------

plt.figure()
healthActivityDuration15.plot()
display()

# COMMAND ----------

plt.figure()
healthActivityDurationCombinedYear.plot()
display()

