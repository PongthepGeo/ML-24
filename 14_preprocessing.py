#-----------------------------------------------------------------------------------------#
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
#-----------------------------------------------------------------------------------------#

df = pd.read_csv('../dataset/ch_04/feature_soilnutrients_VI.csv')

#-----------------------------------------------------------------------------------------#

# Step 1: Add encoded label column
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['feature_soil nutrients'])

# Step 2: Normalize the feature columns (shift mean to zero)
columns_to_normalize = ['NDVI', 'GNDVI', 'NDRE', 'OSAVI', 'BLUE', 'GREEN', 'RED', 'REDEDGE', 'NIR']
scaler = StandardScaler()

# Create new columns for normalized data with '_normalized' suffix
df_normalized = pd.DataFrame(scaler.fit_transform(df[columns_to_normalize]),
                             columns=[f'{col}_normalized' for col in columns_to_normalize])

# Step 3: Apply smoothing window (moving average with window size 3) separately for original and normalized data
# Original columns smoothed
df_smoothed_original = df[columns_to_normalize].rolling(window=3, min_periods=1).mean()
df_smoothed_original = df_smoothed_original.add_suffix('_smoothed')

# Normalized columns smoothed
df_smoothed_normalized = df_normalized.rolling(window=3, min_periods=1).mean()
df_smoothed_normalized = df_smoothed_normalized.add_suffix('_smoothed')

# Step 4: Concatenate the original, normalized, and smoothed dataframes
df = pd.concat([df, df_normalized, df_smoothed_original, df_smoothed_normalized], axis=1)

# Step 5: Save to CSV
output_path = 'data_out/preprocessed_data.csv'
df.to_csv(output_path, index=False)