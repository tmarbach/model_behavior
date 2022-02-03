#!/usr/bin/env python3

import os
import re
from datetime import datetime
import argparse
from turtle import window_height
from rf import forester
from window_maker import reduce_dimensions
from window_maker import pull_window
from window_maker import construct_xy
from svm import svm
from kmeans import kmeans
import pandas as pd



def arguments():
    parser = argparse.ArgumentParser(
            prog='model_select', 
            description="Select a ML model to apply to acceleration data",
            epilog=""
                 )
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



def run_a_model(model, Xdata, X_train, X_test, y_train, y_test, classes):
        if model == 'svm':
                svmreport, parameter_list = svm(X_train, X_test, y_train, y_test, classes)
                return svmreport, parameter_list
        elif model == 'rf':
                rfreport, parameter_list = forester(X_train, X_test, y_train, y_test, classes)
                return rfreport, parameter_list
        elif model == 'kmeans':
                kmreport, parameter_list = kmeans(Xdata,classes)
                #returns something different than the other models
                #because its unsupervised.
                return kmreport, parameter_list
        

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
        list_key = [model, str(window_size)]
        dt_string = str(datetime.now())
        numbers = re.sub("[^0-9]", "", dt_string)
        list_key.append(numbers)
        truekey = '-'.join(list_key)
        return truekey


def main():
    args = arguments()
    bdict = {'s': 0, 'l': 1, 't': 2, 'c': 3, 'a': 4, 'd': 5, 'i': 6, 'w': 7}
    df = pd.read_csv("~/CNNworkspace/raterdata/dec21_cleanPennf1.csv")
    key = construct_key(args.model, args.window_size) 
    windows, classes = pull_window(df, int(args.window_size))
    Xdata, ydata = construct_xy(windows, bdict)
    n_samples, n_features, n_classes = Xdata.shape[0], Xdata.shape[1]*Xdata.shape[2], len(classes)
    X_train, X_test, y_train, y_test = reduce_dimensions(Xdata,ydata)
    report, parameters = run_a_model(args.model, Xdata, X_train, X_test, y_train, y_test, classes)
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
    # return output_file (csv of report statistics of the model.)


if __name__ == "__main__":
    main()