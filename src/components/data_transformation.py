import sys
import os
import re
import pandas as pd
import numpy as np
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    """
    Configuration for data transformation artifacts.
    """
    preprocessor_obj_file_path: str = os.path.join(
        'artifacts', "proprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def split_transmission(self, trans):
        """
        Regex to split transmission string (e.g., 'AS6') into Type ('AS') and Gears (6).
        """
        try:
            if trans == 'Others' or pd.isna(trans):
                return 'Others', 0

            letters = re.findall(r'[A-Za-z]+', str(trans))
            numbers = re.findall(r'\d+', str(trans))

            trans_type = letters[0] if letters else 'Unknown'
            gears = int(numbers[0]) if numbers else 0

            return trans_type, gears
        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transformer_object(self):
        """
        Creates a preprocessing pipeline for both numerical and categorical data.
        """
        try:
            # Define columns based on the dataset
            numerical_columns = [
                "Engine Size(L)", "Cylinders", "Fuel Consumption Comb (L/100 km)", "Trans_Gears"]
            categorical_columns = ["Make", "Model",
                                   "Vehicle Class", "Fuel Type", "Trans_Type"]

            # Numerical Pipeline: Impute missing values and scale
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Categorical Pipeline: Impute, One-Hot Encode (Dense), and Scale
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    # sparse_output=False is CRITICAL to avoid dimension errors with np.hstack/np.c_
                    ("one_hot_encoder", OneHotEncoder(
                        handle_unknown="ignore", sparse_output=False)),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Combine both pipelines into one preprocessor object
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def apply_custom_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Domain-specific cleaning and feature engineering.
        """
        try:
            df = df.copy()

            # 1. Standardize 'Make' and filter Top 10
            TOP_MAKES = ['FORD', 'CHEVROLET', 'BMW', 'MERCEDES-BENZ', 'PORSCHE',
                         'TOYOTA', 'GMC', 'AUDI', 'NISSAN', 'JEEP']
            df['Make'] = df['Make'].str.upper().str.strip()
            df['Make'] = df['Make'].where(df['Make'].isin(TOP_MAKES), 'Others')

            # 2. Extract keywords from 'Model'
            keywords_model = ['FFV', 'AWD', '4WD', '4X4']
            df['Model'] = df['Model'].apply(
                lambda x: next(
                    (word for word in keywords_model if word in str(x).upper()), 'Other_Models')
            )

            # 3. Standardize 'Vehicle Class'
            keywords_vehicle = ['SUV - SMALL', 'MID-SIZE', 'COMPACT', 'SUV - STANDARD',
                                'FULL-SIZE', 'SUBCOMPACT', 'PICKUP TRUCK - STANDARD']
            df['Vehicle Class'] = df['Vehicle Class'].str.upper().str.strip()
            df['Vehicle Class'] = df['Vehicle Class'].where(
                df['Vehicle Class'].isin(keywords_vehicle), 'Others')

            # 4. Process 'Transmission' using the split helper
            df[['Trans_Type', 'Trans_Gears']] = df['Transmission'].apply(
                lambda x: pd.Series(self.split_transmission(x))
            )

            # Ensure numeric types for calculation
            df["Engine Size(L)"] = pd.to_numeric(
                df["Engine Size(L)"], errors='coerce')
            df["Cylinders"] = pd.to_numeric(df["Cylinders"], errors='coerce')
            df["Fuel Consumption Comb (L/100 km)"] = pd.to_numeric(
                df["Fuel Consumption Comb (L/100 km)"], errors='coerce')

            return df
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Main entry point to transform CSV data into preprocessed numpy arrays.
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Applying custom cleaning logic...")
            train_df = self.apply_custom_cleaning(train_df)
            test_df = self.apply_custom_cleaning(test_df)

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "CO2 Emissions(g/km)"

            # Dropping unused columns and target
            drop_columns = [target_column_name, "Transmission", "Fuel Consumption City (L/100 km)",
                            "Fuel Consumption Hwy (L/100 km)", "CO2 Rating", "Smog Rating"]

            input_feature_train_df = train_df.drop(
                columns=drop_columns, errors='ignore')
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(
                columns=drop_columns, errors='ignore')
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing (Fit-Transform).")

            # These will be dense numpy arrays because sparse_output=False
            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(
                input_feature_test_df)

            # Reshape target to 2D for concatenation
            train_arr = np.hstack([input_feature_train_arr, np.array(
                target_feature_train_df).reshape(-1, 1)])
            test_arr = np.hstack([input_feature_test_arr, np.array(
                target_feature_test_df).reshape(-1, 1)])

            logging.info("Saving the preprocessor pickle file.")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
