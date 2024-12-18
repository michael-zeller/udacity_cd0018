import sys
import pandas as pd
from sqlalchemy import create_engine

# load data from the source csv-files
def load_data(messages_filepath, categories_filepath):
    
    messages = pd.read_csv(messages_filepath)    
    categories = pd.read_csv(categories_filepath)
    
    df =   pd.merge(messages, categories, on='id')
    return df

# clean the data
def clean_data(df):
    # Split the categories column into separate category columns
    categories_split = df['categories'].str.split(';', expand=True)

    # Select the first row of the categories dataframe
    row = categories_split.iloc[0]

    # Extract a list of new column names for categories
    category_colnames = row.apply(lambda x: x.split('-')[0])

    # Rename the columns of `categories_split`
    categories_split.columns = category_colnames

    # Convert category values to just numbers 0 or 1
    for column in categories_split:
        # Set each value to be the last character of the string, convert to numeric, and then to boolean
        # in some cases related-2 is present and we need to convert it to true
        categories_split[column] = categories_split[column].str[-1].astype(int).astype(bool)

    # Concatenate the original dataframe with the new `categories_split` dataframe
    df = pd.concat([df.drop('categories', axis=1), categories_split], axis=1)
    
    df.drop_duplicates(inplace=True)
    
    return df

# save the data to a database
def save_data(df, database_filename):
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('Messages', engine, index=False)


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