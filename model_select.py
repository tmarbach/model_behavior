import argparse
import window_maker
import svm


def arguments():
    parser = argparse.ArgumentParser(
            prog='model_selector', 
            description="Select a ML model to apply to acceleration data",\
            epilog="Columns of accelerometer data must be arranged:'tag_id', 'date', 'time',\
                    'camera_date', 'camera_time', 'behavior', 'acc_x', 'acc_y', 'acc_z', 'temp_c',\
                    'battery_voltage', 'metadata'"
                 )
    parser.add_argument(
            "-m"
            "--model",
            type=str,
            help = "input the path to the csv file of accelerometer data that requires cleaning")
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


def main():
    args = arguments()
    windows = window_maker.pull_window(args.csv_file)
    Xdata, ydata = window_maker.construct_xy(windows)
    output_data(args.csv_file, clean_data, args.output)
    # return output_data which will be a csv file of the cleaned
    # and reorganized data, other scripts will work with it from there.

if __name__ == "__main__":
    main()