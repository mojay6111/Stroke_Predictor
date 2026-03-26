"""
utils/preprocessing.py

Shared preprocessing logic used by both:
  - notebooks/03_model_training.ipynb  (training time)
  - app.py                             (prediction time)

This ensures the model always receives input in the exact
same format it was trained on.
"""
import numpy as np
import joblib
import os

MODELS_PATH = os.path.join(os.path.dirname(__file__), '../models/')

# Load saved encoders and imputer
encoders = joblib.load(MODELS_PATH + 'label_encoders.pkl')
imputer  = joblib.load(MODELS_PATH + 'bmi_imputer.pkl')

# Encoding maps (for reference and direct use in app.py)
GENDER_MAP        = {'Male': 1, 'Female': 0}
MARRIED_MAP       = {'Yes': 1, 'No': 0}
WORK_TYPE_MAP     = {'Govt_job': 0, 'Never_worked': 1, 'Private': 2,
                     'Self-employed': 3, 'children': 4}
RESIDENCE_MAP     = {'Rural': 0, 'Urban': 1}
SMOKING_MAP       = {'Unknown': 0, 'formerly smoked': 1,
                     'never smoked': 2, 'smokes': 3}


def preprocess_input(gender, age, hypertension, heart_disease,
                     ever_married, work_type, residence_type,
                     avg_glucose_level, bmi, smoking_status):
    """
    Accepts human-readable string/numeric inputs from the web form
    and returns a numpy array ready for model.predict_proba().

    Parameters
    ----------
    gender           : str  e.g. 'Male' or 'Female'
    age              : float
    hypertension     : int  0 or 1
    heart_disease    : int  0 or 1
    ever_married     : str  'Yes' or 'No'
    work_type        : str  e.g. 'Private'
    residence_type   : str  'Urban' or 'Rural'
    avg_glucose_level: float
    bmi              : float or None
    smoking_status   : str  e.g. 'never smoked'

    Returns
    -------
    np.ndarray of shape (1, 10)
    """
    features = np.array([[
        GENDER_MAP.get(gender, 0),
        float(age),
        int(hypertension),
        int(heart_disease),
        MARRIED_MAP.get(ever_married, 0),
        WORK_TYPE_MAP.get(work_type, 2),
        RESIDENCE_MAP.get(residence_type, 1),
        float(avg_glucose_level),
        float(bmi) if bmi else np.nan,
        SMOKING_MAP.get(smoking_status, 0),
    ]])

    # Impute BMI if missing
    features[:, 8] = imputer.transform(features[:, [8]]).ravel()

    return features