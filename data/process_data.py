import sys
import pandas as pd
import sqlite3

def load_data(messages_filepath, categories_filepath):
    '''
    load two csv format files and merge on 'id'.
    - messages columns: id, messages, original, genre;
    - categories columns: id, categores.

    '''
    messages = pd.read_csv(messages_filepath)
    messages = messages.drop('original', axis = 1)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on = 'id')
    return df


def clean_data(df):
    
    '''split categories in to 36 columns and drop nan and duplicate values'''
    categories = df.categories.str.split(';', expand = True)
    col_names = categories.iloc[0,:].apply(lambda x: x[:-2])
    categories.columns = col_names
    for col in categories:
        categories[col] = categories[col].astype(str).str.split('-').str[1]
        categories[col] = categories[col].astype(int)
    
    df = df.drop('categories', axis = 1)
    df = pd.concat([df, categories], axis = 1)
    df = df.dropna().drop_duplicates()
    return df


def save_data(df, database_filename):
    #engine = create_engine('sqlite:///database_filename')
    #df.to_sql('df_table', engine, index = False, if_exists = 'replace')                       
    conn = sqlite3.connect(database_filename)
    df.to_sql('messages', con = conn, if_exists = 'replace')

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
