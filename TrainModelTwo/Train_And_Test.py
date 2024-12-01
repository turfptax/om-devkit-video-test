import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle
import os
from sklearn.metrics import mean_squared_error, r2_score

def load_data(csv_file):
    """
    Loads the data from the CSV file.
    """
    data = pd.read_csv(csv_file)
    return data

def prepare_data(data):
    """
    Separates the features and labels.
    """
    # Features: Sensors 0-11
    feature_cols = [f"Sensor_{i}" for i in range(12)]
    X = data[feature_cols]

    # Labels: Labels 0-3
    label_cols = [f"Label_{i}" for i in range(4)]
    y = data[label_cols]

    return X, y

def train_model(X, y):
    """
    Trains the Random Forest Regressor model.
    """
    # Initialize the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Fit the model
    model.fit(X, y)

    return model

def save_model(model, filename):
    """
    Saves the trained model to a file using pickle.
    """
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {filename}")

def load_model(filename):
    """
    Loads the model from a file.
    """
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    print(f"Model loaded from {filename}")
    return model

def test_model_on_data(model_filename, csv_file):
    """
    Loads a trained model and tests it on new data.
    Saves the predictions and actual values to a CSV file.
    """
    # Load the model
    model = load_model(model_filename)
    
    # Check if the CSV file exists
    if not os.path.exists(csv_file):
        print(f"CSV file not found at {csv_file}")
        return
    
    # Load the data
    data = load_data(csv_file)
    print("Test data loaded successfully.")
    
    # Prepare the data
    X_test, y_test = prepare_data(data)
    print("Test data prepared.")
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error on test data: {mse}")
    print(f"R² Score on test data: {r2}")
    
    # Save predictions and actual values to CSV
    predictions_df = pd.DataFrame(y_pred, columns=[f"Predicted_Label_{i}" for i in range(y_pred.shape[1])])
    actuals_df = y_test.reset_index(drop=True)
    results_df = pd.concat([predictions_df, actuals_df], axis=1)
    results_df.to_csv('predictions_vs_actuals.csv', index=False)
    print("Predictions and actual values saved to 'predictions_vs_actuals.csv'")
    
    # Optionally, return predictions and metrics
    return y_pred, mse, r2

def main():
    # Paths to the files
    csv_file = 'filtered_output.csv'
    model_filename = 'random_forest_regressor.pkl'
    test_csv_file = 'filtered_output_TestData.csv'  # Replace with your actual file name

    # Check if the CSV file exists
    if not os.path.exists(csv_file):
        print(f"CSV file not found at {csv_file}")
        return

    # Load the data
    data = load_data(csv_file)
    print("Data loaded successfully.")

    # Prepare the data
    X, y = prepare_data(data)
    print("Data prepared for training.")

    # Train the model
    model = train_model(X, y)
    print("Model trained successfully.")

    # Save the model
    save_model(model, model_filename)

    # Test the model on the third data capture
    y_pred, mse, r2 = test_model_on_data(model_filename, test_csv_file)

    # Optionally, perform further analysis with y_pred and y_test
    # For example, plot the predictions vs. actual values

if __name__ == '__main__':
    main()
