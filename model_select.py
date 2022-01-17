#!/usr/bin/env python3

import os
from datetime import datetime
import argparse
from rf import forester
from window_maker import reduce_dimesions
from window_maker import pull_window
from window_maker import construct_xy
from svm import svm
from kmeans import kmeans
import pandas as pd



def arguments():
    parser = argparse.ArgumentParser(
            prog='model_selector', 
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
            "--window_size",
            help="Number of rows to include in each data point (25 rows per second)",
            default=False, 
            type=int
            )
    parser.add_argument(
            "-o",
            "--output_file",
            help="Directs the output to a name of your choice",
            default=False
            )
    return parser.parse_args()



def output_data(reportdf, model, output_filename, n_samples, n_features, n_classes, class_map):
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
        data_record = []
        dt_string = str(datetime.now())
        samfeaclas = f"# classes: {n_classes}; # samples: {n_samples}; # features {n_features}"
        data_record.append(model)
        data_record.append(dt_string)
        data_record.append(samfeaclas)
        data_record.append(class_map)
        data_series = pd.Series(data_record)
        # ^ instead need to be recorded in separate file
        rdf = reportdf.append(data_series, ignore_index=True)
        if os.path.exists(output_filename):
                rdf.to_csv(output_filename, mode='a')# append if already exists
        elif output_filename == False:
                rdf.to_csv('summary_scores_'+ model + '_'+str(n_features), index=False)
        else:
                rdf.to_csv(output_filename, index=False)


def run_a_model(model, Xdata, X_train, X_test, y_train, y_test, classes):
        if model == 'svm':
                svmreport = svm(X_train, X_test, y_train, y_test, classes)
                return svmreport
        elif model == 'rf':
                rfreport = forester(X_train, X_test, y_train, y_test, classes)
                return rfreport
        elif model == 'kmeans':
                kmreport = kmeans(Xdata,classes)
                #returns something different than the other models
                #because its unsupervised.
                return kmreport
        


def main():
    args = arguments()
    bdict = {'s': 0, 'l': 1, 't': 2, 'c': 3, 'a': 4, 'd': 5, 'i': 6, 'w': 7}
    df = pd.read_csv("~/CNNworkspace/raterdata/dec21_cleanPennf1.csv")
    windows, classes = pull_window(df, int(args.window_size))
    Xdata, ydata = construct_xy(windows, bdict)
    n_samples, n_features, n_classes = Xdata.shape[0], Xdata.shape[1]*Xdata.shape[2], len(classes)
    X_train, X_test, y_train, y_test = reduce_dimesions(Xdata,ydata)
    report, parameters = run_a_model(args.model, Xdata, X_train, X_test, y_train, y_test, classes)
    reportdf = pd.DataFrame(report).transpose()
    output_data(reportdf,args.model,args.output_file,n_samples, n_features, n_classes)
    # return output_file (csv of report statistics of the model.)

if __name__ == "__main__":
    main()