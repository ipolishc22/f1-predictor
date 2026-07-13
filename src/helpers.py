import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from pandas.api.types import is_string_dtype, is_numeric_dtype,is_categorical_dtype
from sklearn.metrics import roc_auc_score
from IPython.display import display
from concurrent.futures import ProcessPoolExecutor
import math
import re
import importlib

RANDOM_STATE = 42

def proc_df(df, y_fld=None, skip_flds=None, ignore_flds=None, do_scale=False, na_dict=None,
            preproc_fn=None, max_n_cat=None, subset=None, mapper=None):

    if not skip_flds: skip_flds=[]
    if na_dict is None: na_dict={}

    # get a random sample of the original data and save it into df
    if subset: df = get_sample(df,subset)
    else: df = df.copy()

    # save the y values in a dataframe and drop it from original df 
    if y_fld is None: y = None
    else:
        if not is_numeric_dtype(df[y_fld]):
            df[y_fld] = pd.Categorical(df[y_fld]).codes
        y = df[y_fld].values
        skip_flds += [y_fld]
    df.drop(skip_flds, axis=1, inplace=True)

    # we call the fix_missing on all column names and values in the df
    # they get saved into na_dict containing all the replacement values
    for name, vals in df.items(): na_dict=fix_missing(df,vals,name,na_dict)

    # apply numericalize() to all the columns in the dataframe
    for name, vals in df.items(): numericalize(df, vals, name, max_n_cat)

    # replace your current get_dummies line with these two lines:
    # this is a new line I added during Lecture 4. Might be removed later
    df = df.astype({col: str for col in df.select_dtypes('category').columns})
    df = pd.get_dummies(df, dummy_na=False, dtype=int, drop_first=True)
    return df, y, na_dict

def train_cats(df):
    for name, vals in df.items():
        if is_string_dtype(vals):
            df[name] = vals.astype("category").cat.as_ordered()

def is_date(x):
    return np.issubdtype(x.dtype, np.datetime64)


def apply_cats(validation_df,train_df):
    for name, vals in validation_df.items():
        if (name in train_df.columns) and (train_df[name].dtype.name=="category"):
            validation_df[name] = vals.astype("category").cat.as_ordered()
            validation_df[name] = validation_df[name].cat.set_categories(train_df[name].cat.categories, ordered=True)


def fix_missing(df,col,name,na_dict):
    if is_numeric_dtype(col):
        if pd.isnull(col).sum() or (name in na_dict):
            df[name+"_na"] = pd.isnull(col)
            filler = na_dict[name] if name in na_dict else col.median()
            df[name] = col.fillna(filler)
            na_dict[name] = filler
    return na_dict

def numericalize(df,col,name,max_n_cat):
    if (not is_numeric_dtype(col)) and (max_n_cat is None or len(col.cat.categories)>max_n_cat):
        df[name] = pd.Categorical(col).codes+1


def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 100):
        display(df)


def add_datepart(df, fldnames, drop=True, time=False, errors="raise"):
    # errors defaults to "raise" which means that if the function 
    # encounters a variable that is not in datetime format,
    # it will riase and error. 
    if isinstance(fldnames, str):
        fldnames = [fldnames]
    for fldname in fldnames:
        fld = df[fldname]
        fld_dtype = fld.dtype
        if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
            fld_dtype = np.datetime64
        
        if not is_date(fld):
            df[fldname] = fld = pd.to_datetime(fld, errors=errors)
        
        targ_pre = re.sub('[Dd]ate$', '', fldname)
        attr = ['Year', 'Month', 'Day', 'Dayofweek', 'Dayofyear',
                'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start',
                 'Is_year_end', 'Is_year_start']
        
        if time: 
            attr = attr + ['Hour', 'Minute', 'Second']
        
        df[targ_pre + 'Week'] = fld.dt.isocalendar().week.astype(int)
        
        for n in attr:
            df[targ_pre + n] = getattr(fld.dt, n.lower())
        
        df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
        if drop:
            df.drop(fldname, axis=1, inplace=True)


def rf_feat_importance(m, df):
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}).sort_values('imp', ascending=False)

def rmse(x,y):
    return np.sqrt(((x-y)**2).mean())

def get_sample(df,n):
    idxs = sorted(np.random.permutation(len(df))[:n])
    return df.iloc[idxs].copy()

def split_vals(a,n):
    return a[:n].copy(), a[n:].copy()

def print_score(m, X_train, y_train, X_valid, y_valid):
    #rmse(m.predict(X_train), y_train),
    #rmse(m.predict(X_valid), y_valid),
    train_score = m.score(X_train, y_train)
    valid_score = m.score(X_valid, y_valid)
    roc_score = roc_auc_score(y_valid, m.predict_proba(X_valid)[:,1])
    
    print(f"train: {train_score}")
    print(f"valid: {valid_score}")
    print(f"roc  : {roc_score}")
    if hasattr(m, "oob_score_"): print(f"oob  : {m.oob_score_}")


def parallel_trees(m, fn, n_jobs=8):
    return list(ProcessPoolExecutor(n_jobs).map(fn, m.estimators_))


