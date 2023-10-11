"""
Filename : parseData.py
Author : Archit Joshi
Description :
Language : python3
"""
import os
import pandas as pd
import dpkt
import socket
from sklearn.preprocessing import MinMaxScaler


def scaleData(data):
    scaler = MinMaxScaler()

    for col in data.columns:
        if data[col].dtype in ['int64', 'float64']:
            # Fit and transform the data
            data[[col]] = scaler.fit_transform(data[[col]])
    return data


def create_pcap(data, pcap_file='pcap_data'):
    with open(pcap_file, 'wb') as f:
        pcap_writer = dpkt.pcap.Writer(f)
        for _, row in data.iterrows():
            eth = dpkt.ethernet.Ethernet()

            eth.data = dpkt.ip.IP()
            eth.data.data = dpkt.tcp.TCP()

            eth.data.src = socket.inet_aton('127.0.0.1')
            eth.data.dst = socket.inet_aton('127.0.0.1')
            eth.data.data.sport = 12345
            eth.data.data.dport = 80

            payload = ','.join(f'{attr}:{value}' for attr, value in row.items())
            eth.data.data.data = payload.encode('utf-8')

            # eth.data.data.data = str(row['label']).encode()

            pcap_writer.writepkt(eth)


def main():
    filepath = os.getcwd() + '//data//dummy_data'
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

    dataframe = pd.read_csv(filepath, names=columns)
    dataframe = scaleData(dataframe)
    create_pcap(dataframe)


if __name__ == "__main__":
    main()
