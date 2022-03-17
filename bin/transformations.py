import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def transform_xy(windows, classdict):
    """
    Purpose:
        Converts list of single class dataframes to two arrays
        Xdata (all raw/transformed datapoints) & ydata (class label).
        transformations included per axis :
            mean, std, min, max, kurtosis, skew, corr (xy, yz, xz)
    Input:
        windows -- list of dataframes of all one class
    Output:
        Xdata -- arrays of xyz data of each window stacked together
        ydata -- integer class labels for each window
    """
    positions = ['accX', 'accY', 'accZ']
    Xdata, ydata = [], []
    for window in windows:
        alldata = window[positions].to_numpy()
        alldata = np.append(alldata, np.float32([window[positions].mean(axis = 0)]), 0)
        alldata = np.append(alldata, np.float32([window[positions].std(axis = 0)]), 0)
        alldata = np.append(alldata, np.float32([window[positions].min(axis = 0)]), 0)
        alldata = np.append(alldata, np.float32([window[positions].max(axis = 0)]), 0)
        alldata = np.append(alldata, np.float32([window[positions].kurtosis(axis = 0)]), 0)
        alldata = np.append(alldata, np.float32([window[positions].skew(axis = 0)]), 0)
        correlation = np.array([[window[positions[0]].corr(window[positions[1]]),
            window[positions[1]].corr(window[positions[2]]),
            window[positions[0]].corr(window[positions[2]])]])
        alldata = np.append(alldata, np.float32(correlation), 0)
                
        Xdata.append(alldata)

        ydata.append(classdict[window['behavior'].iloc[0]])
    return np.stack(Xdata), np.asarray(ydata)



def recall_heatmapper(conf_matrix_df, output_fig_name):
    df_norm_col=(conf_matrix_df-conf_matrix_df.mean())/conf_matrix_df.std()
    sns.heatmap(df_norm_col, annot=True, fmt='g')
    # sns.set(font_scale=1.2) # for label size
    plt.savefig('recall' + str(output_fig_name))
    plt.close


def precision_heatmapper(conf_matrix_df, output_fig_name):
    df_norm_row = conf_matrix_df.apply(lambda x: (x-x.mean())/x.std(), axis = 1)
    sns.heatmap(df_norm_row, annot=True, fmt='g')
    # plt.figure(figsize=(15,15))
    # sns.set(font_scale=1.2) # for label size
    plt.savefig('precision-' + str(output_fig_name))
    plt.close
