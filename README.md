# Disaster Response Pipeline Project
This project processes raw messages data and classifies into 36 categories. It includes a web app that allows users to input messages and get categories. 
Data source: [Figure Eight](https://www.figure-eight.com/)
### data
Two input csv files store messages and categories data;
'process_data.py' loads and transforms data and stores in a database;
Run the following command in the project's root directory:
`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

### models
'train_classifier.py' pulls data from database and applies Random Forest Classifier on the training data;
The model is saved in a pickle file; 
Run the following command in the project's root directory:
`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`


### app
Run the Flask web app in the app's directory 'python run.py'/; 
Check the web app from http://0.0.0.0:3001/

Below is a screen-shot of the web app. 
![Web App](https://github.com/candywendao/Disaster-Response-Pipeline/blob/master/app/WebApp.png)
