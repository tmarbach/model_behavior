#!/usr/bin/env python3

import os
import pandas as pd
import argparse
#from datetime import datetime
#from datetime import timedelta

# TODO: inspect timestamps for the number of entries per second. 
    # if there are fewer than the median, then remove that second of data.
    # likely trims the first and last seconds of data. 
    # the input_index column must be put in after all csv files have been concatentated
        # but before rows are ignored for incomplete/faulty data
        # currently, concat doesnt occur
    
    # Ultimately, write an output file for the data removed by this script. 
    #   this would be for troubleshooting and verification. 

def arguments():
    parser = argparse.ArgumentParser(
            prog='accelml_prep_csv', 
            description="Clean and prepare accelerometer csv data for CNN input by rounding\
                        to 3 decimal places and removing blank timestamps",\
            epilog="Columns of accelerometer data must be arranged:'tag_id', 'date', 'time',\
                    'camera_date', 'camera_time', 'behavior', 'acc_x', 'acc_y', 'acc_z', 'temp_c',\
                    'battery_voltage', 'metadata'"
                 )
    parser.add_argument(
            "csv_file",
            type=str,
            help = "input the path to the csv file of accelerometer data that requires cleaning")
    parser.add_argument(
            "-o",
            "--output",
            help="Directs the output to a name of your choice",
            default=False)
    return parser.parse_args()



def accel_data_csv_cleaner(accel_data_csv):
    df = pd.read_feather(accel_data_csv)
    #df = pd.read_csv(accel_data_csv,low_memory=False)
    #check column names if they fit correctly and add an error if they don't
    # df = df.rename(columns={'TagID':'tag_id',
    #                         'Date':'date',
    #                         'Time':'time',
    #                         'Camera date':'camera_date',
    #                         'Camera time':'camera_time',
    #                         'Behavior':'behavior',
    #                         'accX':'acc_x',
    #                         'accY':'acc_y',
    #                         'accZ':'acc_z',
    #                         'Temp. (?C)':'temp_c',
    #                         'Battery Voltage (V)':'battery_voltage',
    #                         'Metadata':'metadata'},
    #                         errors="raise")
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


def output_data(original_data, clean_csv_df, output_location):
    filename = os.path.basename(original_data)
    if output_location == False:
        clean_csv_df.to_csv('clean_'+filename, index=False)
    else:
        clean_csv_df.to_csv(output_location, index=False)






def main():
    args = arguments()
    clean_data = accel_data_csv_cleaner(args.csv_file)
    output_data(args.csv_file, clean_data, args.output)


if __name__ == "__main__":
    main()
