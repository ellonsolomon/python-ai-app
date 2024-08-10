import tensorflow as tf  
from tensorflow.keras import layers  
import pandas as pd  
import numpy as np  
from tensorflow.keras import datasets, layers, models  
from tensorflow.keras.utils import to_categorical  
from sklearn.preprocessing import LabelEncoder  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import confusion_matrix, precision_score, recall_score  
import io  
  
# Print a startup message  
print('Starting up iris model service')  
  
# Declare global variables to store models, datasets, and metrics  
global models, datasets, metrics  
models = []  
datasets = []  
metrics = []  
  
def build():  
    global models  
    # A Sequential model with two dense layers and an output layer  
    model = tf.keras.Sequential([  
        tf.keras.layers.Dense(64, activation='relu', input_shape=(20,)),  
        tf.keras.layers.Dense(64, activation='relu'),  
        tf.keras.layers.Dense(3, activation='softmax')  
    ])  
    # Compile the model with RMSprop optimizer and categorical crossentropy loss  
    model.compile(optimizer='rmsprop',  
                  loss='categorical_crossentropy',  
                  metrics=['accuracy'])  
    # Append the model to the list of models and return its index  
    models.append(model)  
    model_ID = len(models) - 1  
    return model_ID 

  
# Function to load local dataset  
def load_local():  
    global datasets  
    print("load local default data")  
    dataFolder = './'  
    dataFile = dataFolder + "iris_data_encoded.csv"  
    # Append the dataset read from CSV to the list of datasets and return its index  
    datasets.append(pd.read_csv(dataFile))  
    return len(datasets) - 1  
  
# Function to add a dataset to the global list  
def add_dataset(df):  
    global datasets  
    datasets.append(df)  
    return len(datasets) - 1  
  
# Function to retrieve a dataset by its index  
def get_dataset(dataset_ID):  
    global datasets  
    return datasets[dataset_ID]  
  
# Function to train a model with a specified dataset  
def train(model_ID, dataset_ID):  
    global datasets, models  
    dataset = datasets[dataset_ID]  
    model = models[model_ID]  
    # Extract features and labels from the dataset  
    X = dataset.iloc[:, 1:21].values  # Features from the second to the twenty-first column  
    y = dataset.iloc[:, 0].values  # Label is in the first column  
    # Encode the labels into integers  
    encoder = LabelEncoder()  
    y1 = encoder.fit_transform(y)  
    Y = pd.get_dummies(y1).values  
    # Split the data into training and testing sets  
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)  
    # Train the model  
    history = model.fit(X_train, y_train, batch_size=1, epochs=10)  
    print(history.history)  
    # Evaluate the model on the test set  
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)  
    print('Test loss:', loss)  
    print('Test accuracy:', accuracy)  
    # Predict the labels for the test set  
    y_pred = model.predict(X_test)  
    actual = np.argmax(y_test, axis=1)  
    predicted = np.argmax(y_pred, axis=1)  
    print(f"Actual: {actual}")  
    print(f"Predicted: {predicted}")  
    # Calculate and print the confusion matrix, precision, and recall  
    conf_matrix = confusion_matrix(actual, predicted)  
    print('Confusion matrix on test data is {}'.format(conf_matrix))  
    print('Precision Score on test data is {}'.format(precision_score(actual, predicted, average=None)))  
    print('Recall Score on test data is {}'.format(recall_score(actual, predicted, average=None)))  
    # Return the training history  
    return history.history  

def test(model_ID, dataset_ID):  
    global datasets, models, metrics  
    model = models[model_ID]  
    dataset = datasets[dataset_ID]  
      
    # Extract features and labels from the dataset  
    X = dataset.iloc[:, 1:21].values  # Features from the second to the twenty-first column  
    y = dataset.iloc[:, 0].values  # Label from the first column  
      
    # Encode the labels into integers  
    encoder = LabelEncoder()  
    y_encoded = encoder.fit_transform(y)  
    Y = pd.get_dummies(y_encoded).values  
      
    # Evaluate the model on the entire dataset  
    loss, accuracy = model.evaluate(X, Y, verbose=0)  
      
    # Predict the labels for the dataset  
    y_pred = model.predict(X)  
    actual = np.argmax(Y, axis=1)  
    predicted = np.argmax(y_pred, axis=1)  
      
    # Calculate the confusion matrix, precision, and recall  
    conf_matrix = confusion_matrix(actual, predicted)  
    precision = precision_score(actual, predicted, average=None)  
    recall = recall_score(actual, predicted, average=None)  
      
    # Save the test metrics  
    test_metrics = {  
        'loss': loss,  
        'accuracy': accuracy,  
        'confusion_matrix': conf_matrix.tolist(),  # Convert numpy array to list for JSON serializable  
        'precision': precision.tolist(),  
        'recall': recall.tolist()  
    }  
    metrics.append(test_metrics)  
      
    # Return the test metrics  
    return test_metrics  
  
# Function to create a new model and train it with the provided dataset  
def new_model(d):  
    model_ID = build()  
    dataset_ID = add_dataset(d)  
    train(model_ID, dataset_ID)  
    return model_ID  
  
# Function to score the model with provided input features  
def score(model_ID, input_features):  
    global models  
    model = models[model_ID]  
    # Convert the input features into a 2D array as model.predict expects a 2D array  
    x_test2_np = np.array([input_features])  
    y_pred2 = model.predict(x_test2_np)  
    print(y_pred2)  
    # Find the class with the highest probability  
    iris_class = np.argmax(y_pred2, axis=1)[0]  
    print(iris_class)  
    # Map the class index to the actual class name  
    class_mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}  
    predicted_class_name = class_mapping[iris_class]  
    # Return the result as a string  
    return "Score done, class=" + str(predicted_class_name)