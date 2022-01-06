import os
from datetime import datetime
import argparse
from rf import forester
from window_maker import reduce_dimesions
from window_maker import pull_window
from window_maker import construct_xy
import pandas as pd
#import window_maker
#import svm


def arguments():
    parser = argparse.ArgumentParser(
            prog='model_selector', 
            description="Select a ML model to apply to acceleration data",\
            epilog=""
                 )
    parser.add_argument(
            "-m"
            "--model",
            help = "input the path to the csv file of accelerometer data that requires cleaning",
            default=False)
    parser.add_argument(
            "-w",
            "--window_size",
            help="Directs the output to a name of your choice",
            default=False, type=int)
            # check for int, throw error if not int
    parser.add_argument(
            "-o",
            "--output_file",
            help="Directs the output to a name of your choice",
            default=False)
    return parser.parse_args()


def output_data(reportdf, modwinpar_list, output_filename):

        """
        Input:
        score_tuple -- various accuracy scores in tuple form
        modwinpar_list -- list of modelname, windowsize, parameters
        output_filename -- title for output filename

        Output:
        summary_stats -- print out summary stats after running
        recorded stats -- record stats in output file
        """
        df = reportdf.append(modwinpar_list, ignore_index=True, header=False)
        df.append(pd.Series(), ignore_index=True)
        if os.path.exists(output_filename):
                df.to_csv(output_filename, mode='a')# append if already exists
        elif output_filename == False:
                df.to_csv('summary_scores_'+str(modwinpar_list[0]), index=False)
        else:
                df.to_csv(output_filename, index=False)


def model_flow(model, Xdata, X_train, X_test, y_train, y_test, classes):
        model = str(model)
        if model == 'svm':
                svmreport = svm(X_train, X_test, y_train, y_test)
                return svmreport
        elif model == 'rf':
                rfreport = forester(X_train, X_test, y_train, y_test)
                return rfreport
        elif model == 'kmeans':
                kmreport = kmeans(Xdata,classes)
                return kmreport
        


def main():
    args = arguments()
    df = pd.read_csv("~/CNNworkspace/raterdata/dec21_cleanPennf1.csv")
    windows, classes = pull_window(df, int(args.window_size))
    Xdata, ydata = construct_xy(windows)
    X_train, X_test, y_train, y_test = reduce_dimesions(Xdata,ydata)
    report = model_flow(args.model, X_train, X_test, y_train, y_test, classes)
    #report = forester(X_train, X_test, y_train, y_test, classes)
    reportdf = pd.DataFrame(report).transpose()
    #reportdf.to_csv('svm_stats.csv')
    output_data(reportdf,BLANK,args.output_file)
    print()
    #output_data(args.csv_file, clean_data, args.output)
    # return output_data which will be a csv file of the cleaned
    # and reorganized data, other scripts will work with it from there.

if __name__ == "__main__":
    main()