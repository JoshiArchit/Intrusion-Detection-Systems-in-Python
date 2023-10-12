"""
Filename : parseData.py
Author : Archit Joshi
Description :
Language : python3
"""
import os
import dpkt
import socket
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, \
    confusion_matrix


def parseData():
    filepath = os.getcwd() + '\\data\\corrected'
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
    data = pd.read_csv(filepath, names=columns)
    data = data[data['label'] != "imap."]
    return data


def scaleData(data):
    scaler = MinMaxScaler()

    for col in data.columns:
        if data[col].dtype in ['int64', 'float64']:
            # Fit and transform the data
            data[[col]] = scaler.fit_transform(data[[col]])
    return data


def create_pcap(data, pcap_file="pcap_data"):
    with open(pcap_file, 'wb') as f:
        pcap_writer = dpkt.pcap.Writer(f)
        for _, row in data.iterrows():
            eth = dpkt.ethernet.Ethernet()

            eth.data = dpkt.ip.IP()
            eth.data.data = dpkt.tcp.TCP()

            eth.data.src = socket.inet_aton('127.0.0.1')  # Loopback IP
            eth.data.dst = socket.inet_aton('127.0.0.1')  # Loopback IP
            eth.data.data.sport = 12345  # Replace with your source port
            eth.data.data.dport = 80  # Replace with your destination port
            eth.data.data.data = b''  # Empty payload for simplicity

            pcap_writer.writepkt(eth)


def classifier(data):
    X = data.iloc[:, :-1]  # Features
    y = data.iloc[:, -1]  # Labels

    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(include=['object']).columns

    X_num = X[num_cols]
    X_cat = X[cat_cols]

    # Apply one-hot encoding to categorical variables
    encoder = OneHotEncoder(drop='first', sparse=False)
    X_cat_encoded = encoder.fit_transform(X_cat)

    # Combine numerical and encoded categorical variables
    X_processed = np.hstack((X_num, X_cat_encoded))

    X_train, X_test, y_train, y_test = train_test_split(X_processed, y,
                                                        test_size=0.6,
                                                        random_state=42, stratify=y)

    svm = SVC(kernel='linear',
              C=1.0)  # You can choose different kernels (linear, rbf, etc.) and adjust C as needed
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    print(report)
    conf_matrix = confusion_matrix(y_test, y_pred)
    true_negatives = conf_matrix[1, 1]
    true_positives = conf_matrix[1, 0]
    false_positives = conf_matrix[0, 1]
    false_negatives = conf_matrix[1, 0]

    # Print the number of false positives
    print(f'Number of True Positives: {true_positives}')
    print(f'Number of True Negatives: {true_negatives}')
    print(f'Number of False Positives: {false_positives}')
    print(f'Number of False Negatives: {false_negatives}')


def main():
    dataframe = parseData()
    dataframe = scaleData(dataframe)
    # label_frequencies = dataframe['label'].value_counts()
    # for label, frequency in label_frequencies.items():
    #     print(f"Label {label} occurs {frequency} times.")
    classifier(dataframe)


if __name__ == "__main__":
    main()
