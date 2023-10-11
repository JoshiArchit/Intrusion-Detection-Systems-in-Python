"""
Filename : parseData.py
Author : Archit Joshi
Description :
Language : python3
"""
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def scaleData(data):
    scaler = MinMaxScaler()

    for col in data.columns:
        if data[col].dtype in ['int64', 'float64']:
            # Fit and transform the data
            data[[col]] = scaler.fit_transform(data[[col]])
    return data


def main():
    filepath = os.getcwd() + '//data//dummy'
    columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
               'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
               'num_failed_logins', 'logged_in', 'num_compromised',
               'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
               'num_shells', 'num_access_files', 'num_outbound_cmds',
               'is_host_login', 'is_guest_login', 'count', 'srv_count',
               'serror_rate', 'srv_serror_rate', 'rerror_rate',
               'srv_rerror_rate', 'same_srv_rate', 'dif_srv_rate',
               'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
               'dst_host_same_srv_rate', 'dst_hist_diff_srv_rate',
               'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
               'dst_host_serror_rate', 'dst_host_srv_serror_rate',
               'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label']
    print(len(columns))
    dataframe = pd.read_csv(filepath, names=columns)
    print(dataframe)
    dataframe = scaleData(dataframe)
    print(dataframe)


if __name__ == "__main__":
    main()
