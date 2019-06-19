# model training script

import sys
import pandas as pd
import re
import sqlite3
import pickle

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import warnings
warnings.filterwarnings("ignore")



def load_data(database_filepath):
    '''
    This function loads data from database in the provided filepath and

    returns:
    
    X - messages
    Y - 36 categories
    category names

    '''
    conn = sqlite3.connect(database_filepath)
    df = pd.read_sql('SELECT * FROM messages', conn, index_col=None)
    X = df.message
    Y = df.drop(['index', 'id', 'message', 'genre'], axis = 1)
    return X, Y, Y.columns


def tokenize(text):
    '''
    This function tokenizes data - change to lower case,
    remove punctuations, stopwords and shortwords and returns a list of
    tokens
    
    '''
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stopwords.words('english')]
    tokens = [token for token in tokens if len(token) > 2]
    return tokens


def build_model():

    pipeline = Pipeline(
            [
                ('vect', CountVectorizer(tokenizer = tokenize)),
                ('tfidf', TfidfTransformer()),
                ('clf', MultiOutputClassifier(RandomForestClassifier(), n_jobs = 1))  
            ])

                        
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'tfidf__use_idf': (True, False),
        'tfidf__norm': ['l1', 'l2']
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv
            

def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = pd.DataFrame(model.predict(X_test))
    for i in range(len(category_names)):
        print(category_names[i],classification_report(Y_test.iloc[:,i], Y_pred.iloc[:,i]))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
