#!/usr/bin/env python3

import os
import re
from datetime import datetime
import argparse
from accelml_prep_csv import accel_data_csv_cleaner 
from accelml_prep_csv import output_prepped_data
from rf import forester
from sliding_window import leaping_window
from sliding_window import reduce_dim_strat
from sliding_window import construct_xy
import pandas as pd
import numpy as np



def arguments():
    parser = argparse.ArgumentParser(
            prog='model_select', 
            description="Select a ML model to apply to acceleration data",
            epilog=""
                 ) 
    parser.add_argument(
            "-i",
            "--raw-accel-csv",
            type=str,
            help = "input the path to the csv file of accelerometer data that requires cleaning")
    parser.add_argument(
            "-m",
            "--model",
            help = "Choose a ML model of: svm, rf, or kmeans",
            default=False, 
            type=str
            )
    parser.add_argument(
            "-w",
            "--window-size",
            help="Number of rows to include in each data point (25 rows per second)",
            default=False, 
            type=int
            )
    parser.add_argument(
            "-c",
            "--classes-of-interest",
            help="Define the classes of interest",
            default=False, 
            type=str
            )
    parser.add_argument(
            "-d",
            "--data-output-file",
            help="Directs the data output to a filename of your choice",
            default=False
            )
    parser.add_argument(
            "-p",
            "--param-output-file",
            help="Directs the output of parameters to a filename of your choice",
            default=False
            )
    parser.add_argument(
            "-l",
            "--label-output-file",
            help="Directs the output of label data to a filename of your choice",
            default=False
            )
    return parser.parse_args()


# def run_a_model(model, X_train, X_test, y_train, y_test, n_classes):
#     if model == 'svm':
#         svmreport, parameter_list = svm(X_train, X_test, y_train, y_test)
#         return svmreport, parameter_list
#     elif model == 'rf':
#         rfreport, parameter_list = forester(X_train, X_test, y_train, y_test, n_classes)
#         return rfreport, parameter_list
#     elif model == 'nb':
#         kmreport, parameter_list = naive_bayes(X_train, X_test, y_train, y_test)
#                 #returns something different than the other models
#                 #because its unsupervised.
#     return kmreport, parameter_list


def output_data(reportdf, model, key, output_filename = False):
    #Add another file to output additional info to
    """
    Input:
    score_tuple -- various accuracy scores in tuple form
    modwinpar_list -- list of modelname and windowsize
    output_filename -- title for output filename

    Output:
    summary_stats -- print out summary stats after running
    recorded stats -- record stats in output file
    """
    reportdf.index.name = key
    if output_filename == False:
        reportdf.to_csv('summary_scores_'+str(key)+'.csv')
    elif os.path.exists(output_filename):
        reportdf.to_csv(output_filename, mode='a')# append if already exists
    else:
        reportdf.to_csv(output_filename)



def output_params(model_params, model, key, param_filename = False):
    paramdf = pd.DataFrame(columns=["model", "parameters", "key"])
    paramdf.loc['0','model'] = model
    paramdf.loc['0','parameters'] = [model_params]
    paramdf.loc['0','key'] = key
    if param_filename == False:
                #paramdf = pd.DataFrame(columns=["model", "parameters", "key"], data = param_data)
        paramdf.to_csv(model + '_params.csv', index=False)
    elif os.path.exists(param_filename):
               # paramdf = pd.DataFrame(param_data)
        paramdf.to_csv(param_filename, mode='a')
    else:
               # paramdf = pd.DataFrame(columns=["model", "parameters", "key"], data = param_data)
        paramdf.to_csv(param_filename, index=False)


def label_output(model, window_size, n_samples, n_features, n_classes, key, label_outputfile = False):
    labeldf = pd.DataFrame(columns=[
                    "model", 
                    "window_size", 
                    "sample#", 
                    "feature#",
                    "class#", 
                    "key"
                    ])
    labeldf.loc['0','model'] = model
    labeldf.loc['0','window_size'] = window_size
    labeldf.loc['0','sample#'] = n_samples
    labeldf.loc['0','feature#'] = n_features
    labeldf.loc['0','class#'] = n_classes
    labeldf.loc['0','key'] = key
    if label_outputfile == False:
            labeldf.to_csv(str(key) + '_details.csv', index=False)
    elif os.path.exists(label_outputfile):
            labeldf.to_csv(label_outputfile, mode='a')
    else:
            labeldf.to_csv(label_outputfile, index=False)


def construct_key(model, window_size):
    list_key = ['raw',model, str(window_size)]
    dt_string = str(datetime.now())
    numbers = re.sub("[^0-9]", "", dt_string)
    list_key.append(numbers)
    truekey = '-'.join(list_key)
    return truekey


def class_identifier(df, c_o_i):
    class_labels = []
    blist = list(df.behavior.unique().sum())
    if len(c_o_i) == len(blist):
        class_labels = blist
    class_labels = ["all other classes", oneclass for oneclass in c_o_i]
    bdict = {x: 0 for x in blist}
    count = 0
    for bclass in c_o_i:
        count +=1
        bdict[bclass] = count
    return bdict, class_labels


def main():
    args = arguments()
    df = accel_data_csv_cleaner(args.raw_accel_csv)
    output_prepped_data(args.raw_accel_csv,df)
#TODO: allow for multiple csv inputs, allow for "prepped_" data to be input and 
    # bypass the cleaning phase. 
    df = df.rename(columns={'Behavior':'behavior'})
    key = construct_key(args.model, args.window_size)
    classdict, class_labels = class_identifier(df, list(args.classes_of_interest))
    windows = leaping_window(df, int(args.window_size))
    Xdata, ydata = construct_xy(windows, classdict)
    n_samples, n_features, n_classes = Xdata.shape[0], Xdata.shape[1]*Xdata.shape[2], len(classdict)
    X_train, X_test, y_train, y_test = reduce_dim_strat(Xdata,ydata)
    report, parameters = forester(X_train, X_test, y_train, y_test, (len(args.classes_of_interest) + 1), class_labels)
    reportdf = pd.DataFrame(report).transpose()
    output_params(parameters, args.model, key, args.param_output_file)
    label_output(
                args.model,
                args.window_size,
                n_samples,
                n_features,
                n_classes,
                key,
                args.label_output_file
                )
    output_data(reportdf,args.model, key, args.data_output_file)


if __name__ == "__main__":
    main()

