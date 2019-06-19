import json
import plotly
import re
import pandas as pd

from collections import Counter

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
import sqlite3


app = Flask(__name__)

def tokenize(text):
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
    tokens = word_tokenize(text)
    stopwords_ = stopwords.words("english")
    tokens = [word for word in tokens if word not in stopwords_]
    tokens = [WordNetLemmatizer().lemmatize(word, pos='v') for word in tokens]
    return tokens


# load data
database_filepath = '../data/DisasterResponse.db'
conn = sqlite3.connect(database_filepath)
df = pd.read_sql('SELECT * FROM messages', conn, index_col=None)


# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    df_cat = df[df.columns[4:]]
    cat_proportion = df_cat[df_cat != 0].sum()/len(df_cat)
    cat_proportion = cat_proportion.sort_values(ascending = False)
    cat_x = list(cat_proportion.index)

    words_list = []
    for text in df['message'].values:
        words_list.extend(tokenize(text))
    words_cnt = Counter(words_list)
    lst = words_cnt.most_common(10)
    common_words = pd.DataFrame(lst, columns = ['Word', 'Count'])
    words = common_words['Word'].values
    cnts = common_words['Count'].values
        
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=cat_x,
                    y=cat_proportion
                )
            ],

            'layout': {
                'title': 'Proportion of Messages <br> by Category',
                'yaxis': {
                    'title': "Proportion"
                    
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=words,
                    y=cnts
                )
            ],

            'layout': {
                'title': 'Proportion of Messages <br> by Category',
                'yaxis': {
                    'title': "Counts"
                    
                },
                'xaxis': {
                    'title': "Most Common Words"
                }
            }
        }
    ]
   
    

   
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
