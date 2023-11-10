"""
Filename : annClassifier.py
Author : Archit Joshi, Parijat Kawale
Description : Implementing Misuse based and anomaly based IDS using Artificial
Neural Networks.
Language : python3
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


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
    """
    Helper function to scale input data using MinMaxScaler().

    :param data: training data with labels
    :return: normalized/scaled data
    """
    scaler = MinMaxScaler()

    for col in data.columns:
        if data[col].dtype in ['int64', 'float64']:
            # Fit and transform the data
            data[[col]] = scaler.fit_transform(data[[col]])
    return data


def annAnomalyClassifier(data):
    """
    Anomaly based IDS using ANN with RelU activation.

    :param data: scaled data with labels
    :return: None
    """
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

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

    # One-hot encode the target variable
    encoder = OneHotEncoder(sparse=False)
    y_encoded = encoder.fit_transform(y.values.reshape(-1, 1))

    # Splitting data into train & test
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y_encoded,
                                                        test_size=0.6,
                                                        random_state=42,
                                                        stratify=y)

    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    # model.add(Dense(labels.shape[1], activation='softmax'))  # For misuse-based IDS

    # Step 4: Training the ANN
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    model.add(
        Dense(y_encoded.shape[1],
              activation='softmax'))  # Adjust the number of neurons to match the number of classes

    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

    # Step 5: Evaluation
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    print(classification_report(y_test_classes, y_pred_classes))

    # Step 6: Confusion matrix and ratios
    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)

    # Extract values from the confusion matrix
    FP = conf_matrix.sum(axis=0) - np.diag(conf_matrix)
    FN = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
    TP = np.diag(conf_matrix)
    TN = conf_matrix.sum() - (FP + FN + TP)

    # Calculate false positive and false negative ratios
    false_positive_ratio = FP.sum() / (FP.sum() + TN.sum())
    false_negative_ratio = FN.sum() / (FN.sum() + TP.sum())

    print(f'False Positive Ratio: {false_positive_ratio}')
    print(f'False Negative Ratio: {false_negative_ratio}')


def annMisuseClassifier(data):
    """
    Misuse based IDS using ANN with softmax activation.

    :param data: scaled data with labels
    :return: None
    """
    X = data.iloc[:, :-1]  # Features
    y = data.iloc[:, -1]  # Labels

    # Encode labels for logits and multi-class classification
    y_encoded = LabelEncoder().fit_transform(y)
    y = pd.DataFrame({'Encoded_Labels': y_encoded})

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
    num_features = X_processed.shape[
        1]  # One-hot encoding increases our features

    # Data split (We use training, test and validation sets)
    # Step 1: Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y,
                                                        test_size=0.4,
                                                        random_state=42,
                                                        stratify=y)

    # Step 2: Split testing set into validation and test sets
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test,
                                                    test_size=0.5,
                                                    random_state=42)

    # Define the number of unique attack types
    num_classes = len(np.unique(y))

    # Define ANN model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(num_features,)),
        # Input layer with ReLU activation
        Dense(32, activation='relu'),
        # Hidden layer with ReLU activation
        Dense(num_classes, activation='softmax')
        # Output layer with sigmoid activation for binary classification
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32,
              validation_data=(X_val, y_val))

    # Evaluate the model on a test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {test_accuracy}')
    # Generate classification report
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    print(classification_report(y_test, y_pred))

    # Calculating confusion matrix values
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Extract values from the confusion matrix
    FP = conf_matrix.sum(axis=0) - np.diag(conf_matrix)
    FN = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
    TP = np.diag(conf_matrix)
    TN = conf_matrix.sum() - (FP + FN + TP)

    # Calculate overall false positive and false negative ratios
    overall_false_positive_ratio = FP.sum() / (FP.sum() + TN.sum())
    overall_false_negative_ratio = FN.sum() / (FN.sum() + TP.sum())

    print(f'Overall False Positive Ratio: {overall_false_positive_ratio}')
    print(f'Overall False Negative Ratio: {overall_false_negative_ratio}')


def main():
    dataframe = parseData()
    dataframe = scaleData(dataframe)
    print("\n======== Running Anomaly based ANN Classifier ========")
    annAnomalyClassifier(dataframe)
    print("\n======== Running Misuse based ANN Classifier ========")
    annMisuseClassifier(dataframe)


if __name__ == "__main__":
    main()
