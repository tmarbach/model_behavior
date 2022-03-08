#!/usr/bin/env python3

import os
from re import fullmatch
import pandas as pd




def accel_data_csv_cleaner(accel_data_csv):
    df = pd.read_csv(accel_data_csv)
    if 'Behavior' not in df.columns:
        raise ValueError("'Behavior' column is missing")
    if 'accX' not in df.columns:
        raise ValueError("'accX' column is missing")
    if 'accY' not in df.columns:
        raise ValueError("'accY' column is missing")
    if 'accZ' not in df.columns:
        raise ValueError("'accZ' column is missing")
    df['input_index'] = df.index
    cols_at_front = ['Behavior',
                     'accX', 
                     'accY', 
                     'accZ']
    df = df[[c for c in cols_at_front if c in df]+
            [c for c in df if c not in cols_at_front]]
                   # check for correct number of columns, then check for correct column titles
    # need to check if the first 1 or 2 time signatures (sampling) have 25 entries, if not, kick an error
    #df['Behavior'] = df['Behavior'].fillna('u')
    #df['Behavior'] = df['Behavior'].replace(['n'],'u')
    df= df.dropna(subset=['Behavior'])
    df = df.loc[df['Behavior'] != 'n']
    
    #CURRENTLY removing "no video class" class
    #SET to removing unlabeled data and no video data. keeping labeled class data
    return df


def accel_data_dir_cleaner(accel_data_csv):
    files = [f for f in os.listdir(accel_data_csv) if f.endswith('.csv')]
    fulldf = []
    for csv in files:
        adf = pd.read_csv(csv, low_memory=False)
        if 'Behavior' not in adf.columns:
            raise ValueError("'Behavior' column is missing")
        if 'accX' not in adf.columns:
            raise ValueError("'accX' column is missing")
        if 'accY' not in adf.columns:
            raise ValueError("'accY' column is missing")
        if 'accZ' not in adf.columns:
            raise ValueError("'accZ' column is missing")
        fulldf.append(adf)
    df = pd.concat(fulldf)    
    df['input_index'] = df.index
    cols_at_front = ['Behavior',
                     'accX', 
                     'accY', 
                     'accZ']
    df = df[[c for c in cols_at_front if c in df]+
            [c for c in df if c not in cols_at_front]]
    df= df.dropna(subset=['Behavior'])
    df = df.loc[df['Behavior'] != 'n']
    #CURRENTLY removing "no video class" class
    #SET to removing unlabeled data and no video data. keeping labeled class data
    return df


def output_prepped_data(original_data, clean_csv_df):
    filename = os.path.basename(original_data)
    clean_csv_df.to_csv('prepped_'+filename, index=False)
    




def main(input_csv):
    clean_data = accel_data_csv_cleaner(input_csv)    
    output_prepped_data(input_csv, clean_data)
    return clean_data


if __name__ == "__main__":
    main()
