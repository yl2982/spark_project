# Sparkify project

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [Result](#Result)

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

## Project Motivation<a name="motivation"></a>

This is udacity's capstone project, using spark to analyze user behavior data from music app Sparkify.

Sparkify is a music app, this dataset contains two months of sparkify user behavior log. The log contains some basic information about the user as well as information about a single action. A user can contain many entries. In the data, a part of the user is churned, through the cancellation of the account behavior can be distinguished.


## Result

- The best model is Random Forest model
- According to the Chi-Square feature selection method, Top features that affect the churn user is:
  Minimum Session Time, number of songs listened, number of different artists listened, total use days, number of thumbs-up times, numer of songs listened per session