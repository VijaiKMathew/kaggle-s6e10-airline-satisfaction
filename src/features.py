"""
Gold Medal Feature Engineering Pipeline for Kaggle Playground S6E10.
Predicting Airline Passenger Satisfaction (ROC AUC).
Integrates:
- External Super Teacher Priors (original dataset XGBoost and LightGBM model logits)
- 3-Way Customer Profile Empirical Target Rate (Type of Travel | Class | Customer Type)
- Flight Distance Route Corridor Frequency & Group Profiles
- Survey Zero-Rating Semantic Correction (structural N/A handling & experienced statistics)
- 39 Cross-Domain Context Crosses (Rating × Class, Type of Travel, Customer Type)
- Asymmetric Delay Mechanics & Flight Recovery Dynamics
- High-Signal Interaction Terms & Ordinal Categorical Encodings
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# Add external original model path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT / "data/external/original_aligned"))
try:
    from original_model import score as score_original
except ImportError:
    score_original = None

SURVEY_COLS = [
    'Inflight wifi service',
    'Departure/Arrival time convenient',
    'Ease of Online booking',
    'Gate location',
    'Food and drink',
    'Online boarding',
    'Seat comfort',
    'Inflight entertainment',
    'On-board service',
    'Leg room service',
    'Baggage handling',
    'Checkin service',
    'Cleanliness',
]

ONLINE_COLS = ['Inflight wifi service', 'Ease of Online booking', 'Online boarding']
COMFORT_COLS = ['Seat comfort', 'Leg room service', 'Cleanliness', 'Food and drink']
SERVICE_COLS = ['On-board service', 'Baggage handling', 'Checkin service', 'Gate location', 'Departure/Arrival time convenient']
CONTEXT_COLS = ['Class', 'Type of Travel', 'Customer Type']

def create_gold_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Transforms train and test DataFrames with full Gold Medal feature engineering.
    Returns (train_transformed, test_transformed, categorical_feature_names).
    """
    n_train = len(train_df)
    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    # 1. Super Teacher Priors from Original Dataset Models
    if score_original is not None:
        ogp_combined = score_original(combined)
        combined['ogp_xgb_d8'] = ogp_combined['ogp_xgb_d8'].values.astype(np.float32)
        combined['ogp_lgb'] = ogp_combined['ogp_lgb'].values.astype(np.float32)
        combined['ogp_mean'] = ((combined['ogp_xgb_d8'] + combined['ogp_lgb']) / 2.0).astype(np.float32)

    # 2. 3-Way Customer Profile Empirical Target Rate
    lookup_path = REPO_ROOT / "data/external/original_aligned/lookup_long.csv"
    if lookup_path.exists():
        lookup = pd.read_csv(lookup_path, dtype={'value': str}, keep_default_na=False)
        lookup_profile = lookup[lookup.feature == 'Type of Travel|Class|Customer Type']
        mapping = dict(zip(lookup_profile.value, lookup_profile.rate))
        prof_key = combined['Type of Travel'].astype(str) + '|' + combined['Class'].astype(str) + '|' + combined['Customer Type'].astype(str)
        combined['orig_profile_rate'] = prof_key.map(mapping).fillna(0.43).astype(np.float32)

    # 3. Delay features & Imputation
    combined['Arrival_Delay_is_na'] = combined['Arrival Delay in Minutes'].isna().astype(np.int8)
    combined['Arrival Delay in Minutes'] = combined['Arrival Delay in Minutes'].fillna(combined['Departure Delay in Minutes'])

    dep_delay = combined['Departure Delay in Minutes']
    arr_delay = combined['Arrival Delay in Minutes']
    flight_dist = combined['Flight Distance']

    combined['Total_Delay'] = dep_delay + arr_delay
    combined['Delay_Diff'] = arr_delay - dep_delay
    combined['Delay_Recovered'] = np.maximum(0, -combined['Delay_Diff']).astype(np.float32)
    combined['Delay_Worsened'] = np.maximum(0, combined['Delay_Diff']).astype(np.float32)
    combined['Is_Dep_Delayed'] = (dep_delay > 0).astype(np.int8)
    combined['Is_Arr_Delayed'] = (arr_delay > 0).astype(np.int8)
    combined['Is_Both_Delayed'] = (combined['Is_Dep_Delayed'] & combined['Is_Arr_Delayed']).astype(np.int8)
    combined['Delay_per_Distance'] = (combined['Total_Delay'] / (flight_dist + 1.0)).astype(np.float32)
    combined['Log_Flight_Distance'] = np.log1p(flight_dist).astype(np.float32)
    combined['Log_Total_Delay'] = np.log1p(np.maximum(0, combined['Total_Delay'])).astype(np.float32)

    # 4. Survey Zero-Rating Semantic Correction (0 = Structural Not Offered / Unexperienced)
    survey_data = combined[SURVEY_COLS]
    combined['Survey_Zero_Count'] = (survey_data == 0).sum(axis=1).astype(np.int8)
    combined['Survey_Ones'] = (survey_data == 1).sum(axis=1).astype(np.int8)
    combined['Survey_Fives'] = (survey_data == 5).sum(axis=1).astype(np.int8)

    # Mask zeros as NaN for true experienced averages
    masked_survey = survey_data.replace(0, np.nan)
    combined['Survey_Exp_Mean'] = masked_survey.mean(axis=1).fillna(0).astype(np.float32)
    combined['Survey_Exp_Std'] = masked_survey.std(axis=1).fillna(0).astype(np.float32)
    combined['Survey_Exp_Min'] = masked_survey.min(axis=1).fillna(0).astype(np.float32)
    combined['Survey_Exp_Max'] = masked_survey.max(axis=1).fillna(0).astype(np.float32)
    combined['Survey_Exp_Range'] = (combined['Survey_Exp_Max'] - combined['Survey_Exp_Min']).astype(np.float32)

    experienced_count = (survey_data > 0).sum(axis=1).replace(0, 1)
    combined['Survey_NPS_Score'] = ((combined['Survey_Fives'] - combined['Survey_Ones']) / experienced_count).astype(np.float32)

    # Sub-category scores
    combined['Online_Exp_Mean'] = masked_survey[ONLINE_COLS].mean(axis=1).fillna(0).astype(np.float32)
    combined['Comfort_Exp_Mean'] = masked_survey[COMFORT_COLS].mean(axis=1).fillna(0).astype(np.float32)
    combined['Service_Exp_Mean'] = masked_survey[SERVICE_COLS].mean(axis=1).fillna(0).astype(np.float32)
    combined['Online_Comfort_Ratio'] = (combined['Online_Exp_Mean'] / (combined['Comfort_Exp_Mean'] + 0.05)).astype(np.float32)
    combined['Online_Service_Ratio'] = (combined['Online_Exp_Mean'] / (combined['Service_Exp_Mean'] + 0.05)).astype(np.float32)

    # 5. Route Corridor ID Features (`Flight Distance`)
    dist_counts = combined['Flight Distance'].value_counts()
    combined['Route_Frequency'] = combined['Flight Distance'].map(dist_counts).fillna(1).astype(np.int32)

    route_means = combined.groupby('Flight Distance')[
        ['Online boarding', 'Inflight wifi service', 'Age', 'Departure Delay in Minutes']
    ].transform('mean')
    combined['Route_Mean_Online_Boarding'] = route_means['Online boarding'].astype(np.float32)
    combined['Route_Mean_Wifi'] = route_means['Inflight wifi service'].astype(np.float32)
    combined['Route_Mean_Age'] = route_means['Age'].astype(np.float32)

    combined['Route_Diff_Online_Boarding'] = (combined['Online boarding'] - combined['Route_Mean_Online_Boarding']).astype(np.float32)
    combined['Route_Diff_Wifi'] = (combined['Inflight wifi service'] - combined['Route_Mean_Wifi']).astype(np.float32)

    # 6. Context interactions & 39 Cross-Domain crosses
    cat_crosses = []
    for rating in SURVEY_COLS:
        for ctx in CONTEXT_COLS:
            cross_name = f"{rating}__x__{ctx}"
            combined[cross_name] = combined[rating].astype(str) + "|" + combined[ctx].astype(str)
            cat_crosses.append(cross_name)

    # 4 Passenger Context pairs
    combined['Class_x_Type_of_Travel'] = combined['Class'].astype(str) + '_' + combined['Type of Travel'].astype(str)
    combined['Class_x_Customer_Type'] = combined['Class'].astype(str) + '_' + combined['Customer Type'].astype(str)
    combined['Type_of_Travel_x_Customer_Type'] = combined['Type of Travel'].astype(str) + '_' + combined['Customer Type'].astype(str)
    combined['Gender_x_Customer_Type'] = combined['Gender'].astype(str) + '_' + combined['Customer Type'].astype(str)

    base_cats = ['Gender', 'Customer Type', 'Type of Travel', 'Class',
                 'Class_x_Type_of_Travel', 'Class_x_Customer_Type',
                 'Type_of_Travel_x_Customer_Type', 'Gender_x_Customer_Type']

    # 7. Key Domain numeric interactions
    combined['Distance_per_Age'] = (combined['Flight Distance'] / (combined['Age'] + 1.0)).astype(np.float32)
    combined['Age_x_Seat'] = (combined['Age'] * combined['Seat comfort']).astype(np.float32)
    combined['Wifi_x_Ent'] = (combined['Inflight wifi service'] * combined['Inflight entertainment']).astype(np.float32)
    combined['Seat_x_Leg'] = (combined['Seat comfort'] * combined['Leg room service']).astype(np.float32)
    combined['Online_x_Checkin'] = (combined['Online boarding'] * combined['Checkin service']).astype(np.float32)

    # 8. Ordinal encode categorical columns
    all_cat_cols = base_cats + cat_crosses
    for col in all_cat_cols:
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        combined[col] = oe.fit_transform(combined[[col]].astype(str)).astype(np.int32)

    train_out = combined.iloc[:n_train].copy()
    test_out = combined.iloc[n_train:].copy()
    return train_out, test_out, all_cat_cols
