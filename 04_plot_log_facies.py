#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs')
import utilities as U
#-----------------------------------------------------------------------------------------#
import pandas as pd
#-----------------------------------------------------------------------------------------#

# Modify only this part
well_log = pd.read_csv('../dataset/ch_03/well_log_data.csv')
well_data = well_log[well_log['Well Name'] == 'CROSS H CATTLE']
lithocolors = ['#F4D03F', '#F5B041', '#DC7633', '#6E2C00', '#1B4F72', '#2E86C1',
               '#AED6F1', '#A569BD', '#196F3D']
log_colors = ['green', 'red', 'blue', 'black', 'purple']
log_names = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE']
save_name = 'well_log_data'
lithofacies = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS', 'WS', 'D', 'PS', 'BS']

#-----------------------------------------------------------------------------------------#
# Call customized function
U.log_9_facies(well_data, lithocolors, log_colors, log_names, save_name, lithofacies)
#-----------------------------------------------------------------------------------------#