"""
Filename : idsClassifier.py
Author : Archit Joshi & Parijat kawale
Description : Designing an Intrusion detection system using classification
algorithms.
Language : python3
"""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import IsolationForest
from sklearn.metrics import *
import matplotlib.pyplot as plt
import seaborn as sns


def parseData():
    """
    Parsing data from the KDD dataset.

    :return: pandas dataframe with data
    """
    filepath = os.getcwd() + '/data/corrected'
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

    # Clean data and remove outliers with minimal occurrence
    label_counts = data['label'].value_counts()
    valid_labels = label_counts[label_counts >= 2].index
    data = data[data['label'].isin(valid_labels)]

    return data


def scaleData(data):
    scaler = MinMaxScaler()

    for col in data.columns:
        if data[col].dtype in ['int64', 'float64']:
            # Fit and transform the data
            data[[col]] = scaler.fit_transform(data[[col]])
    return data


def misuseClassifier(data):
    """
    Simulating a misuse-based IDS using a support vector machine model.

    :param data: labelled network dataframe data.
    :return: None
    """

    X = data.iloc[:, :-1]  # Features
    y = data.iloc[:, -1]  # Labels

    # Hot coding categorical data
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(include=['object']).columns

    X_num = X[num_cols]
    X_cat = X[cat_cols]

    # Apply one-hot encoding to categorical variables
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    X_cat_encoded = encoder.fit_transform(X_cat)

    # Combine numerical and encoded categorical variables
    X_processed = np.hstack((X_num, X_cat_encoded))

    # Split data into testing and training data
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y,
                                                        test_size=0.6,
                                                        random_state=42,
                                                        stratify=y)

    # SVM Classifier model and training
    svm = SVC(kernel='rbf', C=10, gamma=1.0)
    svm.fit(X_train, y_train)

    # Classify test data using the svm model
    y_pred = svm.predict(X_test)

    attack_labels = data['label'].unique().tolist()

    for label in y_pred:
        if label in attack_labels and "normal." not in label:
            print(f"Detected Attack: {label}")

    # Calculate and display metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=data[
        'label'].unique().tolist())
    print(f'Accuracy: {accuracy}')
    # print(report)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Calculate ratios for each label using the confusion matrix
    for i, label in enumerate(attack_labels):
        false_positives = conf_matrix[:, i].sum() - conf_matrix[i, i]
        false_negatives = conf_matrix[i, :].sum() - conf_matrix[i, i]

        print(f'Label: {label}')
        print(f'False Positives: {false_positives}')
        print(f'False Negatives: {false_negatives}')
        print()


def anomalyClassifier(data):
    """
    Anomaly based IDS using Isolation forest classifier.

    :param data: labelled network dataframe data.
    :return: None
    """

    X = data.iloc[:, :-1]  # Features
    y = data.iloc[:, -1]  # Labels

    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(include=['object']).columns

    X_num = X[num_cols]
    X_cat = X[cat_cols]

    # Apply one-hot encoding to categorical variables
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    X_cat_encoded = encoder.fit_transform(X_cat)

    # Combine numerical and encoded categorical variables
    X_processed = np.hstack((X_num, X_cat_encoded))

    X_train, X_test, y_train, y_test = train_test_split(X_processed, y,
                                                        test_size=0.4,
                                                        random_state=42,
                                                        stratify=y)
    # Adjust label to suit anomaly based detection.
    # -1 == anomaly, 1 == normal
    y_test_numeric = np.where(y_test == "normal.", 1, -1)
    model = IsolationForest(
        contamination=0.4)  # Adjust the contamination parameter

    model.fit(X_train)

    y_pred = model.predict(X_test)
    y_pred = np.where(y_pred == "normal.", 1, -1)
    # Evaluate the model (you might need a different metric depending on your use case)
    accuracy = accuracy_score(y_test_numeric, y_pred)
    conf_matrix = confusion_matrix(y_test_numeric, y_pred)

    # Assuming -1 is anomaly and 1 is normal
    true_positives = conf_matrix[0, 0]
    false_negatives = conf_matrix[0, 1]
    false_positives = conf_matrix[1, 0]
    true_negatives = conf_matrix[1, 1]

    # Calculate other metrics as needed (e.g., accuracy, precision, recall, F1-score)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)

    # Print the results
    print(f'True Positives: {true_positives}')
    print(f'False Negatives: {false_negatives}')
    print(f'False Positives: {false_positives}')
    print(f'True Negatives: {true_negatives}')
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1_score}')


def main():
    dataframe = parseData()
    dataframe = scaleData(dataframe)
    print("+++++++++ Running misuse based IDS +++++++++")
    misuseClassifier(dataframe)
    print("\n\n=====================================================")
    print("+++++++++ Running anomaly based IDS +++++++++")
    anomalyClassifier(dataframe)


if __name__ == "__main__":
    main()
