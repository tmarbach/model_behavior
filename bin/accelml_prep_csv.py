#!/usr/bin/env python3

import os
import pandas as pd
#import argparse



# def arguments():
#     parser = argparse.ArgumentParser(
#             prog='accelml_prep_csv', 
#             description="Clean and prepare accelerometer csv data for CNN input by rounding\
#                         to 3 decimal places and removing blank timestamps",\
#             epilog="Columns of accelerometer data must be arranged:'tag_id', 'date', 'time',\
#                     'camera_date', 'camera_time', 'behavior', 'acc_x', 'acc_y', 'acc_z', 'temp_c',\
#                     'battery_voltage', 'metadata'"
#                  )
#     parser.add_argument(
#             "csv_file",
#             type=str,
#             help = "input the path to the csv file of accelerometer data that requires cleaning")
#     parser.add_argument(
#             "-o",
#             "--output",
#             help="Directs the output to a name of your choice",
#             default=False)
#     return parser.parse_args()



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
    df['Behavior'] = df['Behavior'].fillna('u')
    df['Behavior'] = df['Behavior'].replace(['n'],'u')
    #df= df.dropna(subset=['Behavior'])
    #df = df.loc[df['Behavior'] != 'n']
    
    df = df.loc[df['Behavior'] != 'h']
    #CURRENTLY removing handling class, makes no sense to train with it.
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
