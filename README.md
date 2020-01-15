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

F1 score will be chosen as the evaluation score because of the relative small size of the dataset

### Results
The results obtained on the test set are summarized in the table below.

|Classifier                        |  F1   | Training Time |  
| :------------------------------- |-------|---------------|
|Logistic Regression               | 0.390 | 107.2s        |
|Random Forest Classifier          | 0.213 | 32.0s         |
|Gradient-Boosted Tree Classifier  | 0.394 | 499.2s        |

- The Gradient-Boosted Tree classifier model has the highest F1 score but its training time is the most. 
- The F1 score of Logistic Regression Classifier is close to the one of GBT model, but its training time is less.
- It is important to note that this data has only 448 unique users, which may result in the relative low F1 score of these three models. If using the large dataset in Amazon EMR cluster, the score may be higher

## Conclusion<a name="conclusion"></a>

**Reflection**

I have built three binary classifier models to identify churned users and it turned out that the `Gradient-Boosted Tree` model performed best. However, the average F1 score is still low because of the small-size dataset(only 448 unique users).

The most challenging part is the feature engineering part. It requires good intuition and creativity since there are thousands of aspects to extract the features from the data and only a few matter. I was adviced to go to this [website](https://elitedatascience.com/feature-engineering-best-practices) to see how to do feature engineering. At the same time, I was also encountered the `Data Leakage` problem in machine learning. I restarted doing the feature engineering process and solved this problem.

**Potential Improvements**

- build and test features that capture additional insights about user's activity patterns, e.g. average length of song listening sessions, ratios of skipped or partially listened songs, etc.
- utilize song-level features that have been ignored so far, e.g. calculate the user's listening diversity in terms of different songs/artists listened to in the specified observation period, etc.
- optimize data wrangling and feature engineering steps 
- perform the model on full Sparkify dataset, using the EMR cluster