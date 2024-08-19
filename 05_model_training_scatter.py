#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs')
import utilities as U
#-----------------------------------------------------------------------------------------#
import pandas as pd
#-----------------------------------------------------------------------------------------#

well_log = pd.read_csv('../dataset/ch_03/well_log_data.csv')
well_log['Facies'] -= 1
lithofacies = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS', 'WS', 'D', 'PS', 'BS']
well_log['Facies Label'] = well_log['Facies'].apply(lambda x: lithofacies[x])
test_well = 'NEWBY'  # Designate the exclusive test well
test_data = well_log[well_log['Well Name'] == test_well].copy()
marker_dict = {
    'SS': "o",  # Circle
    'CSiS': "s",  # Square
    'FSiS': "^",  # Triangle up
    'SiSh': "p",  # Pentagon
    'MS': "*",  # Star
    'WS': "h",  # Hexagon
    'D': "D",  # Diamond
    'PS': "x",  # X
    'BS': "+"  # Plus
}

#-----------------------------------------------------------------------------------------#

U.plot_facies(test_data, marker_dict)

#-----------------------------------------------------------------------------------------#