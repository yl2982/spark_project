# Sparkify project

### Table of Contents

1. [Installation](#installation)
2. [Spark Project Overview](#Overview)
3. [Load Data & Cleanse](#Load)
4. [Feature Engineering](#feature)
5. [Modeling & Evaluation](#model)
6. [Conclusion](#conclusion)

## Installation <a name="installation"></a>

This project uses the following software and Python libraries:

Python

Spark

Pyspark

pandas

Matplotlib

Seaborn

sklearn

Jupyter Notebook.

## Spark Project Overview<a name="Overview"></a>

`Sparkify` is a fictitious streaming service, Millions of users play their favorite songs through music streaming services on a daily basis, either through a free tier plan that plays advertisements, or by using a premium subscription model, which offers additional functionalities and is typically ad-free. Users can upgrade or downgrade their subscription plan any time, but also cancel it altogether, so it is very important to make sure they like the service.

The goal in this project was to help the up-and-coming `Sparkify` business by building and training a binary classifier that is able to accurately identify users who cancelled the Sparkify music streaming service, based on the patterns obtained from their past activity and interaction with the service. A successfully trained model could be deployed into `Sparkify` infrastructure to identify users who are likely to churn in advance.

The model development was performed on the medium-size dataset (243MB)

The model development process consists of the following steps:
- Load and Cleanse Data
- Exploratory Data & Analysis
- Feature Engineering
- Modeling

`Understanding data information`
- userId: user identifier
- auth: authentication level (Logged In, Logged Out, Cancelled, Guest)
- firstName: user's first name
- gender: user's gender
- itemInSession: log count in a given session
- lastName: user's last name
- length: song's length in seconds
- level: subscription level
- location: user's location
- method: http request method
- page: type of interaction
- registration: user's registration timestamp
- sessionId: session to which the log belongs to
- song: song name
- status: http status code
- ts: timestamp of a given log
- userAgent: agent used by the user to access the streaming service
- artist: name of the singer

## Load Data & Cleanse<a name="Load"></a>
- Check whether the important value (userId, sessionId) is missing or not.

## Feature Engineering<a name="feature"></a>
The following features are chosen to train the model
- num_Friend_perday
- num_Error_perday
- num_Help_perday
- num_NextSong_perday
- num_Advert_perday
- num_settings_perday
- num_Down_perday
- num_Up_perday
- num_NextSong_ratio
- positive_ratio
- negative_ratio
- AvgSessionTime
- MaxSessionTime
- MinSessionTime
- gender
- level
- downgraded
- upgraded

## Modeling & Evaluation<a name="model"></a>

The following three models will be chosen to train the data
- Logistic Classifier
- Random Forest Classifier
- Gradient-Boosted Tree Classifier

Since it is a binary classification problem, the model metrics will be `Accuracy`, `Precision`, `Recall` or `F1 score`. So what are the definitions of these metrics and how to decide?

The first thing need to be mentioned is the [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix) and the corresponding `True Positive(TP)`, `True Negative(TN)`, `False Positive(FP)`, `False Negative(FN)` which can be used to calculate `Accuracy`, `Precision`, `Recall` and `F1 score`.![confusion-matrix](https://github.com/yl2982/spark_project/blob/master/confusion-matrix.jpeg)

- **True Positive(TP) -** When the actual class is yes and the value of predicted class is also yes.


- **True Negative(TN) -** When the actual class is no and value of predicted class is also no.


- **False Positive(FP) -** When the actual class is no and the value of predicted class is also yes.


- **False Negative(FN) -** When the actual class is yes and the value of predicted class is also no.

The most familiar metric is probably `Accuracy` since it is just the portion of all accurate predicted. When the costs of having a mis-classified actual positive (or false negative) is very high, `Accuracy` metric may not be a good choice. This [blog](https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9) further explains why.
 
$Accuracy = \frac{TP+TN}{TP+TN+FP+FN}$

As a result, we need other metrics. Here comes the [Precision and Recall](https://en.wikipedia.org/wiki/Precision_and_recall#Definition_(classification_context))

$Precision = \frac{TP}{TP+FP} = \frac{TP}{Total Predicted Positive}$ 

`Precision` talks about how precise/accurate your model is out of those predicted positive, how many of them are actual positive. `Precision` is a good measure to determine, when the costs of False Positive is high. For instance, email spam detection. In email spam detection, a false positive means that an email that is non-spam (actual negative) has been identified as spam (predicted spam). The email user might lose important emails if the precision is not high for the spam detection model.

$Recall = \frac{TP}{TP+FN} = \frac{TP}{Total Actual Positive}$

`Recall` actually calculates how many of the Actual Positives our model capture through labeling it as Positive (True Positive). Applying the same understanding, we know that `Recall` shall be the model metric we use to select our best model when there is a high cost associated with False Negative. For instance, in fraud detection or sick patient detection. If a fraudulent transaction (Actual Positive) is predicted as non-fraudulent (Predicted Negative), the consequence can be very bad for the bank.

Finally, we will talk about [F1 score](https://en.wikipedia.org/wiki/F1_score). `F1 score` is the harmonic mean of the `Precision` and `Recall`, where an `F1 score` reaches its best value at 1 (perfect `Precision` and `Recall`) and worst at 0.

$F1 = 2 \times \frac{Precision \times Recall}{Precison + Recall}$

`F1 Score` is needed when you want to seek a balance between `Precision` and `Recall`. `F1 score` might be a better measure to use if we need to seek a balance between `Precision` and `Recall` and there is an uneven class distribution (large number of Actual Negatives).

In this project, `F1 score` will be chose as one of the model metric and will be used in the final model evaluation on the test set

**AUC** is a standard binary classification metric that works well for imbalanced datasets and will used when doing the cross-validation steps. It stands for **AreaUnderROC**. Therefore, we need first to know what is **ROC (Receiver Operating Characteristics)** and the corresponding **ROC-AUC Curve**.

The data is split to calculate **True Positive Rate** and **False Negative Rate**. See the image from `Udacity` video courses.
![ROC1](https://github.com/yl2982/spark_project/blob/master/ROC1.png)
Then the **ROC** curve is plotted based on these points and the areas under ROC curve (**AUC**) can be calculated.
![ROC2](https://github.com/yl2982/spark_project/blob/master/ROC2.png)
Different split ways may result in different AUC values.
![ROC3](https://github.com/yl2982/spark_project/blob/master/ROC3.png)
By using **AUC** one does not need to worry about where to set the probability threshold that translates the model output probabilities into positive and negative class predictions, since **AUC** summarizes model's performance over all possible thresholds.

### Build Pipelines
- Vectorize numeric features
- StandardScale numeric features
- Vectorize categorical features
- Total Assembler numeric and categorical features

`Logistic Classifier`

The following two prominent parameters will be tuned
- **regParam** (regularization parameter, default=0.0) : 0.0, 0.05, 0.1
- **elasticNetParam** (mixing parameter — 0 for L2 penalty, 1 for L1 penalty, default=0.0): 0.0, 0.5

`Random Forest Classifier`

The following two prominent parameters will be tuned
- **maxDepth** (maximum tree depth, default=5): 4,5,6,7
- **numTrees** (number of trees, default=20): 20,40

`Gradient-Boosted Tree Classifier`

The following two prominent parameters will be tuned
- **maxDepth** (maximum tree depth, default=5) : 4, 5
- **maxIter** (maximum number of iterations, default=20) : 20, 100

The grid search objects the performance of each parameter combination is measured by average **AUC** score (area under the ROC) obtained in **4-fold cross validation**.

Because of the relative small-size of the dataset, it is likely to meet with the [imbalanced classification](https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/) problem caused by the unevenly distributed classifier label in train and test data.

To avoid this problem, I will use the [sampleBy](http://spark.apache.org/docs/2.0.0/api/python/pyspark.sql.html#pyspark.sql.DataFrame.sampleBy) method to stratify the dataset based on the label.

```python
train = data.sampleBy('label', fractions={0:0.85, 1:0.5}, seed=100)
test = data.subtract(train)
```

### Results
The best results of each model are summarized in the table below.

|Classifier                        |AUC    |F1     |  Parameters                      |
|--------------------------------- |-------|-------|----------------------------------|
|Logistic Regression               |0.68   |0.36   | regParam=0.05, elasticNetParam=0.0|
|Random Forest Classifier          |0.66   |0.32   | maxDepth=4, maxTrees=40|
|Gradient-Boosted Tree Classifier  |0.61   |0.49   | maxIter=100, maxDepth=4|

These three models are trained via the 4-fold cross-validation methods to deal with the relative small-size of the dataset. It should be noted that the 4-fold cross-validation method has not followed the stratified sampling ways so it can be one problem for these models.

The **AUC** is relatively high for all three model but the **F1 scores** show a significantly weaker performance. It is interesting to recognize that the **AUC** score of GBT classifier model is the lowest but its **F1 score** is the highest. The reason for this needs to be considered in the future. However, since the **AUC** scores are higher it is very likely that we could substantially increase the **F1 score** on the test set by optimizing the probability threshold that splits positive and negative class predictions. The above **F1 scores** have been obtained using the default 0.5 threshold.

It is important to note that this data has only 448 unique users, which may result in the relatively low F1 score of these three models. If using the large dataset in Amazon EMR cluster, the score may be higher and can prevent the overlization problem and test the generalization ability.

### Feature Importance
![feature_importance](https://github.com/yl2982/spark_project/blob/master/feature_importance.png)
From the above feature importance figure, it can be concluded that `gender` and `level` are not so important. 

By comparison, the following is the top 5 important features:
- `num_Friend_perday` (number of friends on `Sparkify`)
- `AvgSessionTime` (users' average session time)
- `num_Advert_perday` (number of advertisements users see per day)
- `num_Down_perday` (number of users' thumb-down times)
- `num_NextSong_ratio` (number of songs listened among all the interaction times)

Therefore, the following advice may help `Sparkify` prevent losing customers
- Improve the social interaction on `Sparkify`
- Increase the average session time of users
- Increase the number of good ads and decrease the number of bad ads
- Find some ways to decrease the thumb-down times
- Improve the quality of songs on `Sparkify`

## Conclusion<a name="conclusion"></a>

**Reflection**

I have built three binary classifier models to identify churned users and it turned out that the `Gradient-Boosted Tree` model performed best. However, the average F1 score is still low because of the small-size dataset(only 448 unique users).

The most challenging part is the feature engineering part. It requires good intuition and creativity since there are thousands of aspects to extract the features from the data and only a few matter. I was adviced to go to this [website](https://elitedatascience.com/feature-engineering-best-practices) to see how to do feature engineering. At the same time, I was also encountered the `Data Leakage` problem in machine learning. I restarted doing the feature engineering process and solved this problem.

**Potential Improvements**

- build and test features that capture additional insights about user's activity patterns, e.g. average length of song listening sessions, ratios of skipped or partially listened songs, etc.
- utilize song-level features that have been ignored so far, e.g. calculate the user's listening diversity in terms of different songs/artists listened to in the specified observation period, etc.
- optimize data wrangling and feature engineering steps 
- perform the model on full Sparkify dataset, using the EMR cluster

## About Files

- Sparkify.ipynb is the jupyter notebook file that records the details of the project including analysis and codes.
- img is the directory that stores the image used in the Sparkify.ipynb

## Acknowledgement

Thank for [**Udacity**](udacity.com) for this wonderful project, providing the dataset.
