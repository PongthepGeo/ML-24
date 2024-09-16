#-----------------------------------------------------------------------------------------#
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, f1_score, roc_auc_score, precision_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from collections import Counter
#-----------------------------------------------------------------------------------------#

df = pd.read_csv('data_out/preprocessed_data.csv')
output_path = 'data_out/evaluation_metrics_multiclass.csv'
param_grid = {'colsample_bytree': [0.3, 0.5, 0.7],
              'learning_rate': [1e-2, 1e-3, 1e-4],
              'max_depth': [6, 7, 11],
              'alpha': [0, 1, 2]}
n_estimators = 200

#-----------------------------------------------------------------------------------------#

# NOTE Check the class distribution and remove the class with the lowest number of samples
# NOTE Here, we remove the Slightly low class
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set class distribution: {Counter(y_train)}")
print(f"Test set class distribution: {Counter(y_test)}")

#-----------------------------------------------------------------------------------------#

num_classes = len(y.unique())  # Update the number of classes based on the remaining data
total_combinations = len(param_grid['colsample_bytree']) * len(param_grid['learning_rate']) * \
                     len(param_grid['max_depth']) * len(param_grid['alpha'])
progress_counter = 0

#-----------------------------------------------------------------------------------------#

results = []
for colsample_bytree in param_grid['colsample_bytree']:
    for learning_rate in param_grid['learning_rate']:
        for max_depth in param_grid['max_depth']:
            for alpha in param_grid['alpha']:
                progress_counter += 1
                print(f"Progress: {progress_counter}/{total_combinations} (Iteration {progress_counter})")
                params = {'colsample_bytree': colsample_bytree,
                          'learning_rate': learning_rate,
                          'max_depth': max_depth,
                          'alpha': alpha}
                
                xgb_clf = xgb.XGBClassifier(objective='multi:softprob',  # Multi-class classification
                                            num_class=num_classes,
                                            colsample_bytree=colsample_bytree,
                                            learning_rate=learning_rate,
                                            max_depth=max_depth,
                                            alpha=alpha,
                                            n_estimators=n_estimators,
                                            eval_metric="mlogloss")

                eval_set = [(X_train, y_train), (X_test, y_test)]
                xgb_clf.fit(X_train, y_train, eval_set=eval_set, verbose=False)
                y_pred = xgb_clf.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred, average='macro')
                precision = precision_score(y_test, y_pred, average='macro')
                f1 = f1_score(y_test, y_pred, average='macro')
                roc = roc_auc_score(y_test, xgb_clf.predict_proba(X_test), multi_class='ovr')

                result = {'colsample_bytree': colsample_bytree,
                          'learning_rate': learning_rate,
                          'max_depth': max_depth,
                          'alpha': alpha,
                          'accuracy': accuracy,
                          'recall': recall,
                          'precision': precision,
                          'f1_score': f1,
                          'roc_auc': roc}
                results.append(result)
                print(f'Results: {result}')

#-----------------------------------------------------------------------------------------#

results_df = pd.DataFrame(results)
results_df.to_csv(output_path, index=False)

#-----------------------------------------------------------------------------------------#