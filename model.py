# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 01:03:46 2022

@author: SNEHA
"""

# Importing data analysis packages
import pandas as pd
import numpy as np



# Importing natural language processing packages

import string
from nltk.corpus import stopwords


# Importing model selection and feature extraction packages
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import make_column_transformer
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline

# Importing machine learning packages
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix





# Reading all the csv files and merging them into a single dataframe
def get_dataset():
    '''
    A helper method which reads the train and test data.

    Parameters
    ----------
    path : str
        Path of the data folder from which the separate files can be accessed

    Returns
    -------
    data : pandas.DataFrame
        Merged dataset.

    '''
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')
    
    return train_data, test_data



def transform_data(X, y):
    '''
    Function to extract features before model building

    Parameters
    ----------
    X : pandas.DataFrame
        train data
    y : pandas.DataFrame
        labels data

    Returns
    -------
    sparse matrix
        Fit all transformers, transform the data and concatenate results
    preprocessor : sklearn.compose.ColumnTransformer
        preprocessor object
        
    '''

    # Segregating features
    news_feature = 'text'
    title_feature = 'title_author'

    # Building a TF-IDF Vectorizer
    
    preprocessor = make_column_transformer(
    (TfidfVectorizer(
        stop_words= 'english',
        lowercase=True,
        min_df = 3
    ), news_feature),
    
    (TfidfVectorizer(
        stop_words= 'english',
        lowercase=True,
        min_df = 3
    ), title_feature))


    return preprocessor.fit_transform(X, y), preprocessor


def mean_std_cross_val_scores(model, X_train, y_train, **kwargs):
    '''
    Returns mean and std of cross validation

    Parameters
    ----------
    model :
        scikit-learn model
    X_train : numpy array or pandas DataFrame
        X in the training data
    y_train :
        y in the training data

    Returns
    ----------
        pandas Series with mean scores from cross_validation
    '''

    scores = cross_validate(model, X_train, y_train, **kwargs)

    mean_scores = pd.DataFrame(scores).mean()
    std_scores = pd.DataFrame(scores).std()
    out_col = []

    for i in range(len(mean_scores)):
        out_col.append(('%0.3f (+/- %0.3f)' % (mean_scores[i], std_scores[i])))

    return pd.Series(data=out_col, index=mean_scores.index)


def train_models(preprocessor, X, y):
    '''
    A Function to train multiple candidate models 
    
    Parameters
    ----------
    preprocessor : sklearn.compose.ColumnTransformer
        preprocessor objects
    X : pandas.DataFrame
        Train Data
    y : pandas.DataFrame
        Train Labels

    Returns
    -------
    models : List
        List of dictionaries with sklearn.pipeline.Pipeline objects
    results : dict
        Stores results with the scores of all models

    '''
    results = {}

    # Defining the scoring metrics
    scoring = ['accuracy', 'precision']

    # Building Logistic Regression
    logistic_regression_pipeline = make_pipeline(
        preprocessor,
        LogisticRegression(max_iter=2000)
    )

    results['Logistic Regression'] = mean_std_cross_val_scores(
        logistic_regression_pipeline,
        X,
        y,
        scoring=scoring,
        return_train_score=True
    )

    # Building Random Forest Classifier
    random_forest_pipeline = make_pipeline(
        preprocessor,
        RandomForestClassifier()
    )

    results['Random Forest'] = mean_std_cross_val_scores(
        random_forest_pipeline,
        X,
        y,
        scoring=scoring,
        return_train_score=True
    )

    models = [
        ('Random Forest', random_forest_pipeline),
        ('Logistic Regression', logistic_regression_pipeline)
    ]

    return random_forest_pipeline, results




def train_best_model(best_pipeline):
    '''A function to train the best model'''

    best_pipeline.fit(X_train, y_train)

    predictions = best_pipeline.predict(X_test)

    predictions = pd.DataFrame(
        predictions,
        index=X_test.index
    )

    conf_mat = confusion_matrix(y_test, predictions)

    return (
        accuracy_score(y_test, predictions),
        precision_score(y_test, predictions),
        conf_mat
    )


def predict(trained_model, X, y):
    '''A function to generate the final predictions on test set and evaluate the results'''
    predictions = trained_model.predict(X)

    predictions = pd.DataFrame(
        predictions,
        index=X_test.index
    )

    result_dataframe = pd.concat(
        [X['text'], pd.Series(y), predictions],
        axis=1,
        ignore_index=True
    )

    result_dataframe.columns = ['Text', 'True Class', 'Predicted Class']

    return result_dataframe


if __name__ == "__main__":
    
    train_data, test_data = get_dataset()
    
    train_data['title_author'] = train_data['title'] + ' '+ train_data['author']
    train_data = train_data.fillna("")

    train_data = train_data.drop(['id', 'title', 'author'], axis=1)

    # Splitting the data and the labels
    
    news = train_data.drop(columns=['label'])
    labels = train_data['label']

    # Splitting data to test and train before we do the EDA
    X_train, X_test, y_train, y_test = train_test_split(
        news,
        labels,
        test_size=0.2,
        shuffle=True,
        random_state=0,
        stratify=labels
    )

    transformed_data, preprocessor = transform_data(X_train, y_train)

    best_pipeline, results = train_models(preprocessor, X_train, y_train)

    accuracy, precision, conf_mat = train_best_model(
        best_pipeline
    )

    print('The accuracy of the model based on the test set: {}'.format(
        accuracy
    ))

    print('The precision of the model based on the test set: {}'.format(
        precision
    ))

    print('The confusion matrix on the test set:\n{}'.format(
        conf_mat
    ))

#     preds_df = predict(best_model, X_test, y_test)

#     print(preds_df)

