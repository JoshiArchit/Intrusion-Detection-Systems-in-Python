"""
Filename : annClassifier.py
Author : Archit Joshi, Parijat Kawale
Description : Implementing Misuse based and anomaly based IDS using Artificial
Neural Networks.
Language : python3
"""
import os
import random
import warnings
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import *


def parseData():
    """
    Parsing data from the KDD dataset.

    :return: pandas dataframe with data
    """
    filepath = os.getcwd() + '/data/datasetKDD'
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
    label_encoder = LabelEncoder()
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
    history = model.fit(X_train, y_train, epochs=10, batch_size=32,
                        validation_data=(X_val, y_val))

    accuracy_dict = dict()
    for epoch, acc in enumerate(history.history['accuracy']):
        accuracy_dict[epoch] = acc

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

    return accuracy_dict


def modifiedDataMisuseIDS(data):
    """
    Misuse based IDS trained after removing 2 labels from training data to
    gauge generalization.

    :param data: dataset
    :return: None
    """
    random.seed(42)
    attacks = list(data['label'].unique())
    attacks_to_remove = random.sample(attacks, 2)

    print(
        f"Attacks removed from training data : {attacks_to_remove}\nThey are "
        f"still present in the testing data.Accuracy might be affected.")
    # Filter the DataFrame based on the selected labels
    filtered_data = data[data['label'].isin(attacks_to_remove)]

    # Get the count of data points with labels in attacks_to_remove
    count_of_points = len(filtered_data)

    # Calculate the total number of data points
    total_data_points = len(data)

    # Calculate the proportion
    proportion = count_of_points / total_data_points

    print(
        f"Proportion of data points in the dataset with labels in attacks_to_remove: {proportion:.2%}")

    X = data.iloc[:, :-1]  # Features
    y = data.iloc[:, -1]  # Labels
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.6,
                                                        random_state=42,
                                                        stratify=y)

    # Removed labels and corresponding  from X_train, y_train
    mask_train = ~y_train.isin(attacks_to_remove)
    X_train = X_train[mask_train]
    y_train = y_train[mask_train]

    # Process the data and prepare for models.
    # Data preprocessing for categorical variables
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(include=['object']).columns

    X_train_num = X_train[num_cols]
    X_test_num = X_test[num_cols]
    X_train_cat = X_train[cat_cols]
    X_test_cat = X_test[cat_cols]

    # Apply one-hot encoding to categorical variables
    encoder = OneHotEncoder(drop='first', sparse_output=False,
                            handle_unknown='ignore')
    X_train_cat_encoded = encoder.fit_transform(X_train_cat)
    X_test_cat_encoded = encoder.transform(X_test_cat)

    # Combine numerical and encoded categorical variables for training and test
    X_train_processed = np.hstack((X_train_num, X_train_cat_encoded))
    X_test_processed = np.hstack((X_test_num, X_test_cat_encoded))

    # SVM Classifier model and training
    svm = SVC(kernel='rbf', C=10, gamma=1.0)
    svm.fit(X_train_processed, y_train)
    print("MODEL TRAINED"
          )
    y_pred = svm.predict(X_test_processed)
    print("Prediction completed for test data")
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=data[
        'label'].unique().tolist())
    print(f'Accuracy: {accuracy}')
    # print(report)
    conf_matrix = confusion_matrix(y_test, y_pred)
    attack_labels = data['label'].unique().tolist()

    # Calculate ratios for each label using the confusion matrix
    for i, label in enumerate(attack_labels):
        false_positives = conf_matrix[:, i].sum() - conf_matrix[i, i]
        false_negatives = conf_matrix[i, :].sum() - conf_matrix[i, i]

        print(f'Label: {label}')
        print(f'False Positives: {false_positives}')
        print(f'False Negatives: {false_negatives}')
        print()


def modifiedDataAnomalyIDS(data):
    """
    Removing 2 attack labels from the training data to gauge generalization of
    the model
    :param data: dataset
    :return: None
    """
    attack_counts = data['label'].value_counts()
    attacks_to_remove = attack_counts.head(3).index.tolist()
    attacks_to_remove.remove('normal.')
    print(
        f"Attacks removed from training data : {attacks_to_remove}\nThey are "
        f"still present in the testing data.Accuracy might be affected.")

    X = data.iloc[:, :-1]  # Features
    y = data.iloc[:, -1]  # Labels
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.4,
                                                        random_state=42,
                                                        stratify=y)
    mask_train = ~y_train.isin(attacks_to_remove)
    X_train = X_train[mask_train]
    y_train = y_train[mask_train]

    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X_train.select_dtypes(include=['object']).columns

    X_train_num = X_train[num_cols]
    X_train_cat = X_train[cat_cols]
    X_test_num = X_test[num_cols]
    X_test_cat = X_test[cat_cols]

    # Apply one-hot encoding to categorical variables
    encoder = OneHotEncoder(drop='first', sparse_output=False,
                            handle_unknown='ignore')
    X_train_cat_encoded = encoder.fit_transform(X_train_cat)
    X_test_cat_encoded = encoder.transform(X_test_cat)

    # Combine numerical and encoded categorical variables
    X_train_processed = np.hstack((X_train_num, X_train_cat_encoded))
    X_test_processed = np.hstack((X_test_num, X_test_cat_encoded))

    y_test_numeric = np.where(y_test == "normal.", 1, -1)
    model = IsolationForest(
        contamination=0.4)  # Adjust the contamination parameter

    model.fit(X_train_processed)
    print("Model TRAINED")

    y_pred = model.predict(X_test_processed)
    y_pred = np.where(y_pred == "normal.", 1, -1)
    print("Model TESTED")

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


def timeSeriesGraph(accuracy_dict):
    """
    Time series plot for misuse based IDS's epochs vs accuracy.

    :param accuracy_dict: accuracy statistics per epoch.
    :return: None
    """
    epochs = list(accuracy_dict.keys())
    accuracies = list(accuracy_dict.values())

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, accuracies, marker='o', linestyle='-', color='b')
    plt.title('Training Accuracy Over Epochs ANN based misuse IDS')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()


def main():
    warnings.filterwarnings('ignore')

    # Parse and clean data
    dataframe = parseData()

    # Scale data
    dataframe = scaleData(dataframe)

    # Run anomaly based classifier
    print("\n======== Running Anomaly based ANN Classifier ========")
    annAnomalyClassifier(dataframe)

    # Run misuse based classifier
    print("\n======== Running Misuse based ANN Classifier ========")
    accuracy_data = annMisuseClassifier(dataframe)
    timeSeriesGraph(accuracy_data)

    # # ================== MODIFIED IDS using classifiers ==================# #
    # The idea is to remove attacks from training and introduce them directly #
    # to the model during testing. This would depict the models weakness to   #
    # adapt to new and generalized data and thus highlighting the need for    #
    # unsupervised learning.                                                  #
    # # ====================================================================# #
    print(
        "\n======== Running IDS with 2 attacks removed from the training data ========")
    print("\n======== Misuse based IDS with modified data")
    modifiedDataMisuseIDS(dataframe)
    print("\n======== Anomaly based IDS with modified data")
    modifiedDataAnomalyIDS(dataframe)
    warnings.filterwarnings('default')


if __name__ == "__main__":
    main()
