import os
import sys
import pandas as pd
from flask import Flask, request, render_template, json
from src.components.data_transformation import DataTransformation
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.exception import CustomException

application = Flask(__name__)
app = application

# Global configuration for UI columns - Must match the DataFrame exactly
UI_COLS = [
    "Make", "Model", "Vehicle Class", "Engine Size(L)",
    "Cylinders", "Fuel Type", "Trans_Type", "Trans_Gears"
]


def load_and_clean_data():
    """
    Loads raw data from artifacts and applies custom cleaning logic 
    to ensure the UI has the same categories as the training data.
    """
    try:
        artifact_path = os.path.join('artifacts', 'data.csv')
        if not os.path.exists(artifact_path):
            print(
                f"Error: {artifact_path} not found. Run data_ingestion.py first.")
            return pd.DataFrame(columns=UI_COLS)

        df_raw = pd.read_csv(artifact_path)
        dt_obj = DataTransformation()
        df_cleaned = dt_obj.apply_custom_cleaning(df_raw)

        return df_cleaned
    except Exception as e:
        raise CustomException(e, sys)


# Initialize data at startup
df_cleaned = load_and_clean_data()


def get_json_payload():
    """ 
    Serializes the cleaned dataframe to a JS-safe JSON string.
    Missing values are filled to prevent JavaScript parsing errors.
    """
    if df_cleaned.empty:
        return "[]"

    # Filter only needed columns and convert to string for consistent UI rendering
    df_js = df_cleaned[UI_COLS].copy().fillna("Unknown").astype(str)
    return json.dumps(df_js.to_dict(orient='records'))


@app.route('/')
def index():
    return render_template('index.html', valid_configs=get_json_payload())


@app.route('/predictdata', methods=['POST'])
def predict_datapoint():
    try:
        # Constructing CustomData with separate Trans_Type and Trans_Gears
        data = CustomData(
            Make=request.form.get('Make'),
            Model=request.form.get('Model'),
            Vehicle_Class=request.form.get('Vehicle_Class'),
            Engine_Size=float(request.form.get('Engine_Size')),
            Cylinders=int(request.form.get('Cylinders')),
            Fuel_Type=request.form.get('Fuel_Type'),
            Trans_Type=request.form.get('Trans_Type'),
            Trans_Gears=int(request.form.get(
                'Trans_Gears')),  # Pass as integer
            Fuel_Consumption_Comb=float(
                request.form.get('Fuel_Consumption_Comb'))
        )

        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        return render_template('index.html',
                               results=round(results[0], 2),
                               valid_configs=get_json_payload())
    except Exception as e:
        # Logging the actual error for debugging
        print(f"Prediction error details: {str(e)}")
        return render_template('index.html',
                               results="Error in prediction",
                               valid_configs=get_json_payload())


if __name__ == "__main__":
    # Ensure terminal shows diagnostic info
    print(f"Starting Flask server. UI Data loaded: {len(df_cleaned)} rows.")
    app.run(host="0.0.0.0", port=5001, debug=True)
