import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object  # Ensure this function exists in utils.py
import os


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "proprocessor.pkl")

            # Loading the trained model and preprocessor
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # Transforming input features using the loaded preprocessor
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, Make: str, Model: str, Vehicle_Class: str, Engine_Size: float,
                 Cylinders: int, Fuel_Type: str, Trans_Type: str, Trans_Gears: int,
                 Fuel_Consumption_Comb: float):

        self.Make = Make
        self.Model = Model
        self.Vehicle_Class = Vehicle_Class
        self.Engine_Size = Engine_Size
        self.Cylinders = Cylinders
        self.Fuel_Type = Fuel_Type
        self.Trans_Type = Trans_Type
        self.Trans_Gears = Trans_Gears
        self.Fuel_Consumption_Comb = Fuel_Consumption_Comb

    def get_data_as_data_frame(self):
        """
        Converts the user input into a pandas DataFrame with the EXACT column names
        used during the model training/transformation phase.
        """
        try:
            custom_data_input_dict = {
                "Make": [self.Make],
                "Model": [self.Model],
                "Vehicle Class": [self.Vehicle_Class],
                "Engine Size(L)": [self.Engine_Size],
                "Cylinders": [self.Cylinders],
                "Fuel Type": [self.Fuel_Type],
                "Trans_Type": [self.Trans_Type],
                "Trans_Gears": [self.Trans_Gears],
                "Fuel Consumption Comb (L/100 km)": [self.Fuel_Consumption_Comb]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
