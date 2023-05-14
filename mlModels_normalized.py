# imports and configure spark session

import matplotlib.pyplot as plt
# Import libraries
import numpy as np

np.set_printoptions(suppress=True)

# Import PySpark libraries
from pyspark.conf import SparkConf
import pyspark.sql.types as t
from sklearn.metrics import ConfusionMatrixDisplay
np.set_printoptions(suppress=True)
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier, \
    LinearSVC
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics

import warnings

warnings.filterwarnings('ignore')

# configure Spark to use GCP
conf = SparkConf()
conf.set('spark.jars.packages', 'com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.23.0')
spark = SparkSession.builder \
    .appName("Read CSV Files - CreditCard") \
    .config('spark.driver.extraClassPath', '/usr/share/java/google-cloud-sdk/lib/gcloud-java-all.jar') \
    .config('spark.driver.extraClassPath', '/usr/share/java/google-cloud-sdk/lib/bigtable/bundle/*') \
    .config(conf=conf) \
    .getOrCreate()

sc = spark.sparkContext

# Load file
df = spark.read.csv('gs://porthos/datasets/creditcard.csv', header=True, inferSchema=True, sep=",")
# Print Schema
df.printSchema()

# build a data split: 80/20
train, test = df.randomSplit(weights=[0.8, 0.2], seed=42)
print('Train shape: ', (train.count(), len(train.columns)))
print('Test shape: ', (test.count(), len(test.columns)))

# get feature columns names
feature_columns = [col for col in df.columns if col != 'Class']
print(feature_columns)
print(len(feature_columns))


# vectorize
vectorizer = VectorAssembler(inputCols=feature_columns, outputCol="features")
train_vec = vectorizer.transform(train)
test_vec = vectorizer.transform(test)

# instantiate Models

# regression
lr = LogisticRegression(
    featuresCol='features',
    labelCol='Class',
    predictionCol='Class_Prediction',
    maxIter=10,
    regParam=0.3,
    elasticNetParam=0.8
)

# decison tree
dt = DecisionTreeClassifier(featuresCol='features',
    labelCol='Class',
    predictionCol='Class_Prediction'
)

# random forest
rf = RandomForestClassifier(
    featuresCol='features',
    labelCol='Class',
    predictionCol='Class_Prediction'
)

# gradient - boosted tree
gbt = GBTClassifier(
    featuresCol='features',
    labelCol='Class',
    predictionCol='Class_Prediction'
)

# linear support vector machines
lsvc = LinearSVC(
    featuresCol='features',
    labelCol='Class',
    predictionCol='Class_Prediction'
)

# create list of models
list_of_models = [lr, dt, rf, gbt, lsvc]
list_of_model_names = ['Logistic Regression', 'Decision Tree', 'Random Forest',
                       'Gradient-Boosted Tree', 'Linear Support Vector Machines']

# go through list
for model, model_name in zip(list_of_models, list_of_model_names):

    # print current model
    print('Current model: ', model_name)

    # create a pipeline object
    pipeline = Pipeline(stages=[model])

    # fit pipeline
    pipeline_model = pipeline.fit(train_vec)

    # get scores on the training set
    train_pred = pipeline_model.transform(train_vec)

    # get scores on the test set
    test_pred = pipeline_model.transform(test_vec)

    # get accuracy on train and test set
    accuracy_evaluator = MulticlassClassificationEvaluator(predictionCol='Class_Prediction', labelCol='Class',
                                                           metricName='accuracy')
    accuracy_score_train = accuracy_evaluator.evaluate(train_pred)
    accuracy_score_test = accuracy_evaluator.evaluate(test_pred)
    print('Accuracy on Train: ', accuracy_score_train)
    print('Accuracy on Test: ', accuracy_score_test)

    # get precision on train and test set
    precision_evaluator = MulticlassClassificationEvaluator(predictionCol='Class_Prediction', labelCol='Class',
                                                            metricName='precisionByLabel')
    precision_score_train = precision_evaluator.evaluate(train_pred)
    precision_score_test = precision_evaluator.evaluate(test_pred)
    print('Precision on Train: ', precision_score_train)
    print('Precision on Test: ', precision_score_test)

    # get recall on train and test set
    recall_evaluator = MulticlassClassificationEvaluator(predictionCol='Class_Prediction',
                                                         labelCol='Class', metricName='recallByLabel')
    recall_score_train = recall_evaluator.evaluate(train_pred)
    recall_score_test = recall_evaluator.evaluate(test_pred)
    print('Recall on Train: ', recall_score_train)
    print('Recall on Test: ', recall_score_test)

    # get f1-score on train and test set
    f1_evaluator = MulticlassClassificationEvaluator(predictionCol='Class_Prediction', labelCol='Class', metricName='f1')
    f1_score_train = f1_evaluator.evaluate(train_pred)
    f1_score_test = f1_evaluator.evaluate(test_pred)
    print('F1-score on Train: ', f1_score_train)
    print('F1-score on Test: ', f1_score_test)

    # get confusion matrix on train set
    preds_and_labels_train = train_pred.withColumn("Class_Prediction", train_pred["Class_Prediction"]
                                                   .cast(t.DoubleType()))\
        .withColumn("Class", train_pred["Class"].cast(t.DoubleType()))
    preds_and_labels_train = preds_and_labels_train.select(['Class_Prediction', 'Class'])
    metrics_train = MulticlassMetrics(preds_and_labels_train.rdd)
    cm_arr_train = metrics_train.confusionMatrix().toArray().astype(float)
    cm_disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_arr_train)
    print('Confusion Matrix on Train set:')
    cm_disp_train.plot()
    plt.show()

