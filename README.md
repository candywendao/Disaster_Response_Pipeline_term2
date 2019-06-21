# Disaster Response Pipeline Project

## Table of Contents

 1. [Project Overview](#overview)
 2. [File Structure](#structure)
						 	    - [ETL Pipeline - data](#data)
								- [ML Pipeline - models](#models)
								- [Flask Web App - app](#app)
 3. [Running Instructions](#instructions)
 4. [Acknowledgement](#acknowledgement)
***
<a id = 'overview'></a>
## Project Overview
This project processes messages data from [Figure Eight](https://www.figure-eight.com/) and classifies into 36 categories. It includes a web application that allows an emergency worker to input a message and get categories results and presents visualization charts of data. 

<a id = 'structure'></a>
## File Structure
<a id = 'data'></a>
### ETL Pipeline - data
File '/data/process_data.py' stores the ETL pipeline that loads, merges and clean 'categories' and 'messages' and stores in a SQLite database.


<a id = 'models'></a>
### ML Pipeline - models
File '/models/train_classifier.py' stores text processing and machine learning pipeline that:

 - loads data from SQLite database
 - split train and test datasets
 - train and tune models using Random Forest Classifier and GridSearchCV
 - predict on the test dataset
 - save model in a pickle file

<a id = 'app'></a>
### Flask Web App - app

Below is a screen-shot of the web app. It allows an emergency worker to input a message and get the classification results. Below the search bar, it presents several visualization charts of dataset.

![Web App Screenshot](https://github.com/candywendao/Disaster-Response-Pipeline/blob/master/app/WebApp.png)

<a id = 'instructions'></a>
## Running Instructions

 1. Run the following command in the project's root directory:
`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

2. Run the following command in the project's root directory:
`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Run the Flask web app in the app's directory 'python run.py'/; 
Check the web app from http://0.0.0.0:3001/


<a id = 'acknowledgement'></a>
## Acknowledgement
Thank [Figure Eight](https://www.figure-eight.com/) for providing  data and [Udacity](https://www.udacity.com/) for the instructions and advice.
