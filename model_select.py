import argparse
import window_maker
import svm


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
