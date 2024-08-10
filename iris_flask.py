from flask import Flask, request, jsonify  
import pandas as pd  
import io  

# Import functions from the backend model  
from base_iris import build, load_local, datasets, add_dataset, train, new_model, score, test  


app = Flask(__name__)  
  
# Endpoint to upload training data  
@app.route('/iris/datasets', methods=['POST'])  
def upload_dataset():  
    # Check if the 'train' file is part of the uploaded files  
    if 'train' not in request.files:  
        return jsonify({"error": "train file is required"}), 400  
    file = request.files['train']  
    # Check if the file is not empty  
    if not file:  
        return jsonify({"error": "No file provided"}), 400  
    try:  
        # Read the file into a DataFrame  
        df = pd.read_csv(io.StringIO(file.read().decode('utf-8')))  
        # Add the dataset to the list of datasets and get its ID  
        dataset_ID = add_dataset(df)  
        return jsonify({"dataset_ID": dataset_ID}), 201  
    except Exception as e:  
        # If there's an error, return the error message  
        return jsonify({"error": str(e)}), 500  
  
# Endpoint to build and train a new model instance  
@app.route('/iris/model', methods=['POST'])  
def build_and_train_model():  
    data = request.form  
    # Check if the 'dataset' key is in the form data  
    if 'dataset' not in data:  
        return jsonify({"error": "dataset index is required"}), 400  
    try:  
        # Get the dataset ID from the form data and train a new model  
        dataset_ID = int(data['dataset'])  
        model_ID = new_model(datasets[dataset_ID])  
        return jsonify({"model_ID": model_ID}), 201  
    except Exception as e:  
        # If there's an error, return the error message  
        return jsonify({"error": str(e)}), 500  
  
# Endpoint to re-train model using specified dataset  
@app.route('/iris/model/<int:model_ID>', methods=['PUT'])  
def retrain_model(model_ID):  
    dataset_index = request.args.get('dataset')  
    # Check if the 'dataset' query parameter is provided  
    if dataset_index is None:  
        return jsonify({"error": "dataset query param is required"}), 400  
    try:  
        # Get the dataset ID from query parameters and re-train the model  
        dataset_ID = int(dataset_index)  
        history = train(model_ID, dataset_ID)  
        return jsonify(history), 200  
    except Exception as e:  
        # If there's an error, return the error message  
        return jsonify({"error": str(e)}), 500  
  
# Endpoint to score model with provided values  
@app.route('/iris/model/<int:model_ID>', methods=['GET'])  
def score_model(model_ID):  
    fields = request.args.get('fields')  
    # Check if the 'fields' query parameter is provided  
    if not fields:  
        return jsonify({"error": "fields query param is required"}), 400  
    try:  
        # Parse the input features from query parameters  
        input_features = [float(x) for x in fields.split(',')]  
        # Check if there are exactly 20 features  
        if len(input_features) != 20:  
            return jsonify({"error": "Exactly 20 features are required"}), 400  
        # Score the model with the input features  
        result = score(model_ID, input_features)  
        return jsonify({"result": result}), 200  
    except Exception as e:  
        # If there's an error, return the error message  
        return jsonify({"error": str(e)}), 500  
    
    
# Endpoint to test a trained model using a specified dataset  
@app.route('/iris/model/<int:model_ID>/test', methods=['GET'])  
def test_model(model_ID):  
    dataset_index = request.args.get('dataset')  
    # Check if the 'dataset' query parameter is provided  
    if dataset_index is None:  
        return jsonify({"error": "dataset query param is required"}), 400  
    try:  
        # Get the dataset ID from query parameters and test the model  
        dataset_ID = int(dataset_index)  
        test_results = test(model_ID, dataset_ID)  
        return jsonify(test_results), 200  
    except Exception as e:  
        # If there's an error, return the error message  
        return jsonify({"error": str(e)}), 500  

  
# Main entry point for the Flask application  
if __name__ == '__main__':  
    # Run the app with debugging enabled, on host '0.0.0.0' and port '4000'  
    app.run(debug=True, host='0.0.0.0', port=4000)