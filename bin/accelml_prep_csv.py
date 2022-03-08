#!/usr/bin/env python3

import os
import pandas as pd




def accel_data_csv_cleaner(accel_data_csv):
    df = pd.read_csv(accel_data_csv)
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


def output_prepped_data(original_data, clean_csv_df):
    filename = os.path.basename(original_data)
    clean_csv_df.to_csv('prepped_'+filename, index=False)
    




def main(input_csv):
    clean_data = accel_data_csv_cleaner(input_csv)    
    output_prepped_data(input_csv, clean_data)
    return clean_data


if __name__ == "__main__":
    main()
