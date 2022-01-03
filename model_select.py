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
            default=False)

    parser.add_argument(
            "-o",
            "--output_file",
            help="Directs the output to a name of your choice",
            default=False)
    return parser.parse_args()


def output_data(original_data, clean_csv_df, output_location):
        """
        Input:
        score_tuple -- various accuracy scores in tuple form
        output_file_name -- title for output filename
        Output:
        summary_stats -- print out summary stats after running
        recorded stats -- record stats in output file
        """
        if os.path.exists(filename):
        append_write = 'a' # append if already exists
        else:
        append_write = 'w'
    filename = os.path.basename(original_data)
    if output_location == False:
        clean_csv_df.to_csv('summary_scores_'+filename, index=False)
    else:
        clean_csv_df.to_csv(output_location, index=False)


def main():
    args = arguments()
    df = pd.read_csv("~/CNNworkspace/raterdata/dec21_cleanPennf1.csv")
    windows = pull_window(df, int(args.window_size))
    Xdata, ydata = construct_xy(windows)
    X_train, X_test, y_train, y_test = reduce_dimesions(Xdata,ydata)
    print(forester(X_train, X_test, y_train, y_test))
    #output_data(args.csv_file, clean_data, args.output)
    # return output_data which will be a csv file of the cleaned
    # and reorganized data, other scripts will work with it from there.

if __name__ == "__main__":
    main()