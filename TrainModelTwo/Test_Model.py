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

def load_model(filename):
    """
    Loads the model from a file.
    """
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    print(f"Model loaded from {filename}")
    return model

def test_model(model, X_test, y_test):
    """
    Tests the model on the test data.
    """
    # Predict
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error on test data: {mse}")
    print(f"R² Score on test data: {r2}")

    return y_pred, mse, r2

def save_predictions(y_pred, y_test):
    """
    Saves the predictions and actual values to a CSV file.
    """
    predictions_df = pd.DataFrame(y_pred, columns=[f"Predicted_Label_{i}" for i in range(y_pred.shape[1])])
    actuals_df = y_test.reset_index(drop=True)
    results_df = pd.concat([predictions_df, actuals_df], axis=1)
    results_df.to_csv('predictions_vs_actuals.csv', index=False)
    print("Predictions and actual values saved to 'predictions_vs_actuals.csv'")

def main():
    # Path to the test CSV file (third data capture)
    test_csv_file = 'Test_Parsed_Data.csv'  # Replace with your actual test data file
    model_filename = 'random_forest_regressor.pkl'

    # Check if the model file exists
    if not os.path.exists(model_filename):
        print(f"Model file not found at {model_filename}. Please train the model first.")
        return

    # Load the model
    model = load_model(model_filename)

    # Check if the test CSV file exists
    if not os.path.exists(test_csv_file):
        print(f"CSV file not found at {test_csv_file}")
        return

    # Load the test data
    data = load_data(test_csv_file)
    print("Test data loaded successfully.")

    # Prepare the test data
    X_test, y_test = prepare_data(data)
    print("Test data prepared.")

    # Test the model
    y_pred, mse, r2 = test_model(model, X_test, y_test)

    # Save the predictions and actual values
    save_predictions(y_pred, y_test)

if __name__ == '__main__':
    main()
