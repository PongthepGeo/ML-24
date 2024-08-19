#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs')
import utilities as U
#-----------------------------------------------------------------------------------------#
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
import os
#-----------------------------------------------------------------------------------------#

well_log = pd.read_csv('../dataset/ch_03/well_log_data.csv')
features = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE']
target = 'Facies'
test_well = 'CHURCHMAN BIBLE'  # Designate the exclusive test well
save_trained_model = 'trained_model' 
save_name = 'CHURCHMAN'

#-----------------------------------------------------------------------------------------#

os.makedirs(save_trained_model, exist_ok=True)  # Create the directory if it does not exist
well_log['Facies'] -= 1  # Adjust labels, index starts from 0
unique_wells = well_log['Well Name'].unique()
validation_wells = [well for well in unique_wells if well != test_well]  
best_accuracy = 0
save_model_path = os.path.join(save_trained_model, 'model.json')
print(f"Model will be saved at: {save_model_path}")

#-----------------------------------------------------------------------------------------#

for validation_well in validation_wells:
    print(f"Training with validation on {validation_well}")
    train_data = well_log[(well_log['Well Name'] != validation_well) & (well_log['Well Name'] != test_well)]
    validation_data = well_log[well_log['Well Name'] == validation_well]
    X_train, y_train = train_data[features], train_data[target]
    X_validation, y_validation = validation_data[features], validation_data[target]
    model = xgb.XGBClassifier(objective='multi:softmax',
                              num_class=len(pd.unique(well_log[target])),
                              colsample_bytree=0.3,
                              learning_rate=0.004,
                              max_depth=8,
                              alpha=10,
                              n_estimators=1000,
                              eval_metric="mlogloss")
    model.fit(X_train, y_train,
              eval_set=[(X_validation, y_validation)],
              verbose=False)
    validation_predictions = model.predict(X_validation)
    validation_accuracy = accuracy_score(y_validation, validation_predictions)
    if validation_accuracy > best_accuracy:
        best_accuracy = validation_accuracy
        model.save_model(save_model_path)  

#-----------------------------------------------------------------------------------------#

# Load the best model and test on the designated test well
best_model = xgb.XGBClassifier()
best_model.load_model(save_model_path)  # Load the best model
test_data = well_log[well_log['Well Name'] == test_well].copy()  # Explicitly create a copy here
X_test, y_test = test_data[features], test_data[target]
# Predict and evaluate
best_model_predictions = best_model.predict(X_test)
test_data.loc[:, 'Predicted_Facies'] = best_model_predictions  
test_accuracy = accuracy_score(y_test, best_model_predictions)
print(f'Test Accuracy using Best Model on {test_well}: {test_accuracy:.2f}')

#-----------------------------------------------------------------------------------------#

lithocolors = ['#F4D03F', '#F5B041', '#DC7633', '#6E2C00', '#1B4F72', '#2E86C1',
               '#AED6F1', '#A569BD', '#196F3D']
log_colors = ['green', 'red', 'blue', 'black', 'purple']
log_names = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE']
lithofacies = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS', 'WS', 'D', 'PS', 'BS']
U.plot_facies_comparison(test_data, lithocolors, log_colors, log_names, save_name, lithofacies)

#-----------------------------------------------------------------------------------------#