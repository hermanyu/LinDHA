# -*- coding: utf-8 -*-
"""
Herman's Helper Function library.
Contains functions to streamline the model building and model testing pipeline.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

import matplotlib.pyplot as plt
import itertools

import joblib

import PaulBettany as jarvis


def boxplots(df, columns, y, figsize = (25,15)):
    n = len(columns)
    q = n//5
    r = n%5
    
    fig, ax = plt.subplots(q+1, 5, figsize=figsize)
    
    for i in range(q):
        for j in range(5):
            sns.boxplot(x=columns[j+5*i], y=y, ax=ax[i][j], data=df)
            
    if r != 0:
        for j in range(r):
            sns.boxplot(x=columns[j+5*q], y=y, ax=ax[q][j], data=df)
            
def bargraphs(df, columns, figsize=(25,15)):
    n = len(columns)
    q = n//5
    r = n%5
    
    fig, ax = plt.subplots(q+1, 5, figsize=figsize)
    
    for i in range(q):
        for j in range(5):
            sns.countplot(x=columns[j+5*i], ax=ax[i][j], data=df)
            
    if r != 0:
        for j in range(r):
            sns.countplot(x=columns[j+5*q], ax=ax[q][j], data=df)


def num_cat_split(df, read_as_cat=[]):
    """
    creates 2 lists of column names of a dataframe: 
    1 list for numerical columns and 1 list for categorical columns

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to extract from.
    read_as_cat : str or list, optional
        column(s) to convert to categoricals. The default is [].

    Returns
    -------
    num_features : list
        list of numerical columns
    cat_features : list
        list of categorical columns

    """
    if type(read_as_cat) == str:
        read_as_cat = [read_as_cat]
        
    for col in read_as_cat:
        df[col] = df[col].map(str)
        
    num_features = [df.dtypes.index[i] for i in range(len(df.dtypes)) if df.dtypes[i] != 'object']
    cat_features = [df.dtypes.index[i] for i in range(len(df.dtypes)) if df.dtypes[i] == 'object']

    
    return num_features, cat_features



def encode_single_ordinal(df, column, grades, start=1):
    
    copy_df = df.copy(deep=True)
    
    encoded_col = copy_df[column].map({x:grades.index(x)+start for x in grades})
    encoded_col = encoded_col.astype('int64')
    
    return encoded_col
        


def encode_ordinals(df, columns, gradesdict, start=1, drop_original=False):
    copy_df = df.copy(deep=True)
    for col in columns:
        copy_df[col+'_ord'] = encode_single_ordinal(copy_df, col, gradesdict[col], start=start)
        
        if drop_original == True:
            copy_df.drop(columns=col, inplace=True)
        
    return copy_df


def create_interaction(df, columns):
    copy_df=df.copy(deep=True)
    for col0, col1 in zip(columns[0], columns[1]):
        copy_df[col0+' * '+col1] = copy_df[col0] * copy_df[col1]
    
    return copy_df



def sqrt_features(df, columns, drop_original=False):
    
    copy_df = df.copy(deep=True)
    for col in columns:
        if drop_original==True:
            copy_df.drop(columns=col,inplace=True)
        
        copy_df[col+'_sqrt'] = np.sqrt(copy_df[col])
    
    
    return copy_df
        




def model_parameters(model, display=True):
    """
    retrieves and displays model weights (coefficients) and bias (intercept).

    Parameters
    ----------
    model : LinearRegression() object
        model whose parameters are to be retrieved.
    display : bool, optional
        console print out of coefficients. The default is True.

    Returns
    -------
    pd.Series
        the vector of coefficients.

    """
    if display==True:
        print(" ")
        print("weights: ", model.coef_)
        print("bias: ", model.intercept_)
        print(" ")
        
    return pd.Series([model.intercept_]+[beta for beta in model.coef_])


def support(model):
    supp=0
    for c in model.coef_:
        if c != 0:
            supp+=1
            
    return supp



def grade_model(model, X, y, xval=False, cv=5, display=False):
    """
    grades the model performance on a set of data using common metrics: 
    R2 the coefficient of determination, MSE mean squared error, 
    RMSE root mean squared error, and MAE mean absolute error.

    Parameters
    ----------
    model : LinearRegression() object
        model we wish to evaluate.
    X : pd.DataFrame
        Data to evaluate performance on.
    y : pd.Series
        labels to training data.
    cv: int, optional
        number of cross-validation folds to use. Default is 5.
    display: bool, optional
        prints out the scores to console. Default is False.
    Returns
    -------
    pd.DataFrame
        Table of common evaluation metrics: R^2, MSE, RMSE, and MAE

    """
    y_hat = model.predict(X)
    R2 = metrics.r2_score(y, y_hat)
    x_val = 0
    if xval == True:
        x_val = cross_val_score(model, X, y, cv=cv).mean()
    mse = metrics.mean_squared_error(y, y_hat)
    rmse = metrics.mean_squared_error(y, y_hat, squared=False)
    mae = metrics.mean_absolute_error(y, y_hat)
    
    if display == True:
        print(" ")
        print(f"R2 :{R2}")
        if xval == True:
            print(f"k-folds Cross Val: {x_val}")
        print(f"MSE: {mse}")
        print(f"RMSE: {rmse}")
        print(f"MAE: {mae}")
        print(" ")
    
        
    return pd.Series({   'R2': R2,
                             'k-fold cross-val: ': x_val,
                            'MSE': mse,
                            'RMSE': rmse,
                            'MAE': mae
                            })

def plot_residuals(models, X, y, save=None):
    """
    plots residual/errors against the predicted output values y_pred.

    Parameters
    ----------
    model : LinearRegression() object
        model to plot residuals for
    X : pd.DataFrame
        The data to predict on.
    y : pd.Series
        The correct y values.
    save: str
        Relative path to save figure to. Default is None and will not save anything.

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots(1,1, figsize=(20,8));
    y_pred = models.predict(X)
    ax.scatter(x=y_pred, y=y-y_pred, s=1, color='tab:blue');
    ax.axhline(y=0, linestyle='--', color='tab:red');
    ax.set_ylabel('errors');
    ax.set_xlabel('y (predicted)');


def prepare_data(df, features,y = None, dummies=None, rel_freq=None, mean_target=None):
    """
    Takes a dataframe and prepares it for linear regression model fitting. 
    May pass lists of variable names to be encoded in any of 3 three different methods.

    Parameters
    ----------
    df : pd.DataFrame
        Data to be prepared.
    features : list
        list of features to use for training.
    y: str, optional
        the label of the response/target/dependent variable. Must be passed if encode = 'mean'.
        Default value is None.
    dummies: list, optional
        list of features to encode as dummy variables. Default is the empty list.
    rel_freq: list, optional
        list of features to encode using class relative frequency method. Will encode each
        class using the relative frequency of class. Default is the empty list.
    mean_target: list, optional
        list of features to encode using 'mean' method. Also known as 'target' encoding. 
        Will encode each class using the mean of the response variable calculated within class.
        Default is the empty list.

    Returns
    -------
    pd.DataFrame
        returns a dataframe with the specified features and dummy variables.

    """
    prepdf = df[features].copy(deep=True)
    
    if type(dummies)==list:
        prepdf = pd.get_dummies(prepdf, columns=dummies, drop_first=True)
    
    elif dummies == None:
        pass
    
    else:
        raise TypeError('dummies must be a list')
        
    
    if type(rel_freq)==list:
        for col in rel_freq:
            freq = prepdf.value_counts(subset=col, normalize=True)
            prepdf[col] = prepdf[col].map({c:freq[c] for c in freq.index})
    
    elif rel_freq == None:
        pass
    
    else:
        raise TypeError('rel_freq must be a list')
    
    
    if type(mean_target)==list:
        for col in mean_target:
            avg = df.groupby(col)[y].mean()
            prepdf[col] = prepdf[col].map({c:avg[c] for c in avg.index})
    
    elif mean_target==None:
        pass
    
    else:
        raise TypeError('mean_target must be a list')
    
    
    return prepdf


def build_model(X,y):
    """
    Builds a linear regression model and prints coefficients and test scores.

    Parameters
    ----------
    X : pd.DataFrame
        data to train model on.
    y : TYPE
        data labels.

    Returns
    -------
    model : LinearRegression() object
        trained regression model.

    """
    model = LinearRegression()
    model.fit(X,y)
    print(" ")
    print(list(zip(X.columns, model.coef_)))
    print("bias: ", model.intercept_)
    print(" ")
    grade_model(model, X, y, xval=True, display=True)
    
    return model


    
##############################################################################
##                           R&D Experimental Area                          ##
##############################################################################



class Project:
    
    def __init__(self, train, test, target, name=''):
        
        self.name = name
        training = train.copy(deep=True)
        unknown = test.copy(deep=True)
        self.training_index = training.index
        self.unknown_index = unknown.index
        self.training_size = len(training.index)
        self.data = pd.concat([training,unknown])
        self.training = self.data.iloc[:self.training_size,]
        self.y = self.training[target]
        self.unknown = self.data.iloc[self.training_size:,]
        self.target = target
        self.features = [col for col in self.data.columns if col != target]
        self.numericals = [col for col in self.features if self.data[col].dtypes != 'object']
        self.categoricals = [col for col in self.features if self.data[col].dtypes == 'object']
        self.predictions = self.unknown[target]
        self.submissions = self.unknown[[target]]
        self.preparedinputs = self.data[self.features]                
        self.X = self.preparedinputs.iloc[0:len(self.training.index)]
        self.X_unknown = self.preparedinputs.iloc[len(self.training.index):]
        self.X_train = self.X
        self.X_test = None
        self.Xs_unknown = None
        self.Xs_train = None
        self.Xs_test = None
        self.y_train = self.y
        self.y_test = None
        self.y_trainpred = None
        self.y_testpred = None
        self.train_errors = None
        self.test_errors = None
        self.scaler = StandardScaler(copy=True, with_mean=False, with_std=False)
        self.poly = None
        self.model = None
        self.parameters = None
        self.testscores = None
        self.trainscores = None
        self.seed = None
        
        
    def nulls(self, col=None, ascending=False):
        if col == None:
            return self.data[self.features].isnull().sum().sort_values(ascending=ascending)
        
        else:
            return self.data[col].isnull().sum()
        
        
        
        
    def num_to_cat(self, columns=[]):
        for col in columns:
            self.data[col] = self.data[col].astype('object')
            self.categoricals.append(self.numericals.pop(self.numericals.index(col)))
            
    
    
    
    
    def cat_to_num(self, columns=[]):
        for col in columns:
            self.numericals.append(self.categoricals.pop(self.categoricals.index(col)))
    
    
    
    
    
    def save(self, barebones = True, csv_path=None, pkl_path = None, csv=True, pkl = True):
        
        if csv == True and barebones!=True:
            if csv_path==None:
                self.data.to_csv('DataSet')
            
            else:
                self.data.to_csv(csv_path)
        
        if pkl == True and barebones!=True:
            if pkl_path == None:
                joblib.dump(self, self.name+'.pkl')
                
            else:
                joblib.dump(self, pkl_path)
        
        
        elif barebones==True:
            self.data.to_csv(csv_path)
            joblib.dump(self.model, pkl_path)
                  
            
      
##### Data Preparation 

    def update(self):
        self.features = [col for col in self.data.columns if col != self.target]
        self.numericals = [col for col in self.features if self.data[col].dtypes != 'object']
        self.categoricals = [col for col in self.features if self.data[col].dtypes == 'object']
        self.training = self.data.iloc[:self.training_size,]
        self.y = self.training[self.target]
        self.unknown = self.data.iloc[self.training_size:,]
        self.preparedinputs = self.data[self.features]
        self.X = self.preparedinputs.iloc[0:len(self.training.index)]
        self.X_unknown = self.preparedinputs.iloc[len(self.training.index):]
        
    def update_types(self):
        self.features = [col for col in self.data.columns if col != self.target]
        self.numericals = [col for col in self.features if self.data[col].dtypes != 'object']
        self.categoricals = [col for col in self.features if self.data[col].dtypes == 'object']
        


    def get_dummies(self, columns = []):
        if len(columns) == 0:
            columns = self.categoricals
        self.preparedinputs = pd.get_dummies(self.data[self.features], columns = columns, drop_first = True)
        self.X = self.preparedinputs.iloc[0:len(self.training.index)]
        self.X_unknown = self.preparedinputs.iloc[len(self.training.index):]
    

    def split(self, test_size=0.2, seed=None):
        
        if seed == None:
            seed = self.seed
            
            
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=seed)
        
            
            
    def prepare_data(self, dummies = [], test_size=0.2, seed=None):
        
        if seed == None:
            seed = self.seed
        
        if type(dummies) == str:
            dummies = [dummies]
            
        if dummies == []:
            dummies = self.categoricals
        
            
        self.get_dummies(columns = dummies)
        
        self.split(test_size=test_size, seed=seed)
        
        
        self.X_train = pd.DataFrame(data=self.scaler.fit_transform(self.X_train), 
                                    index = self.X_train.index,
                                    columns = self.scaler.get_feature_names_out(self.X_train.columns)
                                    )
        
        self.X_test = pd.DataFrame(data=self.scaler.transform(self.X_test), 
                                   index = self.X_test.index,
                                   columns = self.scaler.get_feature_names_out(self.X_test.columns)
                                   )
        self.X_unknown = pd.DataFrame(data=self.scaler.transform(self.X_unknown), 
                                      index = self.X_unknown.index,
                                      columns = self.scaler.get_feature_names_out(self.X_unknown.columns)
                                      )



#### Build Model

    def prototype(self):
        
        self.model.fit(self.X_train, self.y_train)
        grade_model(self.model, self.X_train, self.y_train, display=True)
        self.y_trainpred = self.model.predict(self.X_train)
        self.y_testpred = self.model.predict(self.X_test)
            

        
        self.train_errors = self.y_train - self.y_trainpred
        self.test_errors = self.y_test - self.y_testpred
        self.parameters = list(zip(self.features+['bias'], self.model.coef_+[self.model.intercept_]))
        
        
        
        
    def build_model(self):
                 
        self.X = self.scaler.fit_transform(self.X)
        self.model.fit(self.X, self.y)
        
        
    def support(self):
        supp = 0
        for i in self.model.coef_:
            if i != 0:
                supp+=1
                
        return supp
        
    
    def load_model(self, filepath):
        self.model = joblib.load(filepath)
        self.y_trainpred = self.model.predict(self.X_train)
        self.y_testpred = self.model.predict(self.X_test)
                  
        self.train_errors = self.y_train - self.y_trainpred
        self.test_errors = self.y_test - self.y_testpred
        self.parameters = list(zip(self.features+['bias'], self.model.coef_+[self.model.intercept_]))
        
         
##### Model Scoring         
            
    def cross_val(self, cv=5, mean=True):

            
        if mean == False:
            return cross_val_score(self.model, self.X_train, self.y_train, cv=cv)
        
        else:
            return cross_val_score(self.model, self.X_train, self.y_train, cv=cv).mean()
        
    
    def display_grades(self, slice='both'):
        
        if slice == 'test':
            print(f"R2:   {self.testscores['R2']}")
            print(f"MSE:  {self.testscores['MSE']}")
            print(f"RMSE: {self.testscores['RMSE']}")
            print(f"MAE:  {self.testscores['MAE']}")
            
        if slice == 'train':
            print(f"R2:   {self.trainscores['R2']}")
            print(f"MSE:  {self.trainscores['MSE']}")
            print(f"RMSE: {self.trainscores['RMSE']}")
            print(f"MAE:  {self.trainscores['MAE']}")
    
        if slice == 'both':
            print("          Train        ", "           Test       ")
            print(f"R2:   {self.trainscores['R2']},   {self.testscores['R2']}")
            print(f"MSE:  {self.trainscores['MSE']},  {self.testscores['MSE']}")
            print(f"RMSE: {self.trainscores['RMSE']},   {self.testscores['RMSE']}")
            print(f"MAE:  {self.trainscores['MAE']},   {self.testscores['MAE']}")
            
    def grade(self, display=True):
                  
        self.testscores = pd.Series([
                
                                    metrics.r2_score(self.y_test, self.y_testpred),
                                    metrics.mean_squared_error(self.y_test, self.y_testpred),
                                    metrics.mean_squared_error(self.y_test, self.y_testpred, squared=False),
                                    metrics.mean_absolute_error(self.y_test, self.y_testpred)
                
                                ],
                                index = ['R2', 'MSE', 'RMSE', 'MAE']
                )
            
        
                
        self.trainscores = pd.Series([
                
                                    metrics.r2_score(self.y_train, self.y_trainpred),
                                    metrics.mean_squared_error(self.y_train, self.y_trainpred),
                                    metrics.mean_squared_error(self.y_train, self.y_trainpred, squared=False),
                                    metrics.mean_absolute_error(self.y_train, self.y_trainpred)
                
                                ],
                                index = ['R2', 'MSE', 'RMSE', 'MAE']
                )

        if display == True:
            self.display_grades(slice='both')
        
    def plot_residuals(self, slice='test', scaled=False, figsize=(25, 6), s=1.5, color='tab:blue', linecolor='tab:red'):
        
        if slice == 'both':
            
            fig, ax = plt.subplots(1,2, figsize=figsize)
    
            ax[0].scatter(x=self.y_trainpred, y=self.train_errors, s=s, c=color);
            ax[0].set_xlabel('Train Set Predictions');
            ax[0].set_ylabel('Error/Residual');
            ax[0].set_title('Residual Plot (Train Set)', fontsize=16);
            ax[0].axhline(y=0, linestyle='--', c=linecolor)
            
            ax[1].scatter(x=self.y_testpred, y=self.test_errors, s=s, c=color);
            ax[1].set_xlabel('Test Set Predictions');
            ax[1].set_ylabel('Error/Residual');
            ax[1].set_title('Residual Plot (Test Set)', fontsize=16);
            ax[1].axhline(y=0, linestyle='--', c=linecolor);
        
        
        if slice == 'test':
            plt.figure(figsize=figsize);
            plt.scatter(x=self.y_testpred, y=self.test_errors, s=s, c=color);
            plt.xlabel('Test Set Predictions');
            plt.ylabel('Error/Residual');
            plt.title('Residual Plot (Test Set)', fontsize=16);
            plt.axhline(y=0, linestyle='--', c=linecolor);

        if slice == 'train':
            plt.figure(figsize=figsize);
            plt.scatter(x=self.y_trainpred, y=self.train_errors, s=s, c=color);
            plt.xlabel('Train Set Predictions');
            plt.ylabel('Error/Residual');
            plt.title('Residual Plot (Train Set)', fontsize=16);
            plt.axhline(y=0, linestyle='--', c=linecolor);
            
        
##### Model Predictions

    def prepare_submissions(self):
        self.submissions[self.target] = self.model.predict(self.X_unknown)
        
## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++