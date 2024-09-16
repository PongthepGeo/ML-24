#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs')
import utilities as U
#-----------------------------------------------------------------------------------------#
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
#-----------------------------------------------------------------------------------------#

df = pd.read_csv('data_out/preprocessed_data.csv')

#-----------------------------------------------------------------------------------------#

class_counts = df['label'].value_counts()
lowest_class = class_counts.idxmin()
df_filtered = df[df['label'] != lowest_class]  # Remove the lowest class
features = ['NDVI', 'GNDVI', 'NDRE', 'OSAVI', 'BLUE', 'GREEN', 'RED', 'REDEDGE', 'NIR',
            'NDVI_normalized', 'GNDVI_normalized', 'NDRE_normalized', 'OSAVI_normalized',
            'BLUE_normalized', 'GREEN_normalized', 'RED_normalized', 'REDEDGE_normalized',
            'NIR_normalized', 'NDVI_smoothed', 'GNDVI_smoothed', 'NDRE_smoothed', 'OSAVI_smoothed',
            'BLUE_smoothed', 'GREEN_smoothed', 'RED_smoothed', 'REDEDGE_smoothed', 'NIR_smoothed',
            'NDVI_normalized_smoothed', 'GNDVI_normalized_smoothed', 'NDRE_normalized_smoothed',
            'OSAVI_normalized_smoothed', 'BLUE_normalized_smoothed', 'GREEN_normalized_smoothed',
            'RED_normalized_smoothed', 'REDEDGE_normalized_smoothed', 'NIR_normalized_smoothed']
target = 'label'
X = df_filtered[features]; y = df_filtered[target]

#-----------------------------------------------------------------------------------------#

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set class distribution: {Counter(y_train)}")
print(f"Test set class distribution: {Counter(y_test)}")

#-----------------------------------------------------------------------------------------#

xgb_clf = xgb.XGBClassifier(objective='multi:softmax',
                            num_class=len(y.unique()),  # Update number of classes
                            colsample_bytree=0.3,
                            learning_rate=0.0001,
                            max_depth=16,
                            alpha=1,
                            n_estimators=100,
                            eval_metric="mlogloss")  
eval_set = [(X_train, y_train), (X_test, y_test)]
xgb_clf.fit(X_train, y_train, eval_set=eval_set, verbose=False)

#-----------------------------------------------------------------------------------------#

# NOTE Compute feature importance scores
booster = xgb_clf.get_booster()
importance_scores = booster.get_score(importance_type='weight')  
importance_df = pd.DataFrame({
    'Feature': [features[int(k[1:])] if k.startswith('f') else k for k in importance_scores.keys()],
    'Importance': importance_scores.values()
})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
importance_df.to_csv('data_out/feature_importance.csv', index=False)
print(importance_df.head())

#-----------------------------------------------------------------------------------------#

eval_result = xgb_clf.evals_result()
y_pred = xgb_clf.predict(X_test)
accuracy = (y_pred == y_test).mean()
print(f"\nAccuracy: {accuracy:.8f}")

#-----------------------------------------------------------------------------------------#

U.history_plot(eval_result, output_dir='figure_out')
U.plot_confusion_matrix(y_test, y_pred, X_test, xgb_clf, output_dir='figure_out')

#-----------------------------------------------------------------------------------------#