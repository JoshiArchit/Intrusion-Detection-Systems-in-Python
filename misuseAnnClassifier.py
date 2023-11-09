"""
Filename : workingFileA.py
Author : Archit Joshi
Description :
Language : python3
"""
import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
# Use tensorflow.keras for integrated Keras API
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Sequential
from tensorflow.keras.activations import sigmoid
from sklearn.model_selection import train_test_split


def parseData():
    """
    Parsing data from the KDD dataset.

    :return: pandas dataframe with data
    """
    filepath = os.getcwd() + '/corrected'
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


def annClassifierA(data):
    X = data.iloc[:, :-1]  # Features
    y = data.iloc[:, -1]  # Labels
    y = np.where(y == "normal.", 0, 1)

    # Data preprocessing for categorical variables
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(include=['object']).columns

    X_num = X[num_cols]
    X_cat = X[cat_cols]

    # Apply one-hot encoding to categorical variables
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    X_cat_encoded = encoder.fit_transform(X_cat)

    # Combine numerical and encoded categorical variables
    X_processed = np.hstack((X_num, X_cat_encoded))
    num_features = X_processed.shape[1]     # One-hot encoding increases our features

    # Data split
    # Step 1: Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y,
                                                        test_size=0.4,
                                                        random_state=42,
                                                        stratify=y)

    # Step 2: Split testing set into validation and test sets
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test,
                                                    test_size=0.5,
                                                    random_state=42)

    model = Sequential([
        Dense(64, activation='relu', input_shape=(num_features,)),
        # Input layer with ReLU activation
        Dense(32, activation='relu'),
        # Hidden layer with ReLU activation
        Dense(1, activation='sigmoid')
        # Output layer with sigmoid activation for binary classification
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Train the model (assuming you have training data in X_train and corresponding labels in y_train)
    model.fit(X_train, y_train, epochs=10, batch_size=32,
              validation_data=(X_val, y_val))

    # Evaluate the model on a test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {test_accuracy}')


def main():
    dataframe = parseData()
    dataframe = scaleData(dataframe)
    print("========= RUNNING ANN IDS =========")
    annClassifierA(dataframe)


if __name__ == "__main__":
    main()
