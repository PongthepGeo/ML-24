#-----------------------------------------------------------------------------------------#
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
#-----------------------------------------------------------------------------------------#
params = {
	'savefig.dpi': 300,  
	'figure.dpi' : 300,
	'axes.labelsize':12,  
	'axes.titlesize':12,
	'axes.titleweight': 'bold',
	'legend.fontsize': 10,
	'xtick.labelsize':10,
	'ytick.labelsize':10,
	'font.family': 'serif',
	'font.serif': 'Times New Roman'
}
matplotlib.rcParams.update(params)
#-----------------------------------------------------------------------------------------#

def log_9_facies(logs, facies_colors, log_colors, log_names, save_file, lithofacies):
	logs = logs.sort_values(by='Depth')
	cmap_facies = colors.ListedColormap(facies_colors[:len(facies_colors)], 'indexed')
	ztop = logs.Depth.min()
	zbot = logs.Depth.max()
	cluster = np.repeat(np.expand_dims(logs['Facies'].values, 1), 100, 1)
	f, ax = plt.subplots(nrows=1, ncols=len(log_names) + 1, figsize=(12, 8),
						 gridspec_kw={'width_ratios': [1]*len(log_names) + [0.5]})
	for count, item in enumerate(log_names):
		ax[count].plot(logs[item], logs.Depth, color=log_colors[count])
		ax[count].set_xlabel(item)
		ax[count].set_xlim(logs[item].min(), logs[item].max())
		ax[count].set_ylim(ztop, zbot)
		ax[count].invert_yaxis()
		ax[count].locator_params(axis='x', nbins=3)
		if count > 0:
			ax[count].set_yticklabels([])
		else:
			ax[count].set_ylabel('Depth')
	# Facies image
	im = ax[-1].imshow(cluster, interpolation='none', aspect='auto',
					   cmap=cmap_facies, vmin=1, vmax=len(facies_colors))
	ax[-1].set_xticklabels([])
	ax[-1].set_yticklabels([])
	ax[-1].set_xlabel('Facies')
	# Color bar
	divider = make_axes_locatable(ax[-1])
	cax = divider.append_axes('right', size='20%', pad=0.05)
	cbar = plt.colorbar(im, cax=cax)
	# cbar.set_label('Facies')
	cbar.set_ticks(np.arange(1, len(facies_colors) + 1))
	cbar.set_ticklabels(lithofacies)
	f.suptitle(f'Well: {logs.iloc[0]["Well Name"]}', y=0.93)
	plt.subplots_adjust(wspace=0.1)  # Adjust the spacing between subplots
	plt.savefig(f'figure_out/{save_file}.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
	plt.show()

#-----------------------------------------------------------------------------------------#

def plot_facies_comparison(logs, facies_colors, log_colors, log_names, save_file, lithofacies):
	# Sort logs by depth
	logs = logs.sort_values(by='Depth')
	# Define colormap with 0-based indexing
	cmap_facies = colors.ListedColormap(facies_colors[:len(facies_colors)], 'indexed')

	# Repeat facies values across the horizontal extent of the plot for visualization
	actual_cluster = np.repeat(np.expand_dims(logs['Facies'].values, 1), 100, 1)
	predicted_cluster = np.repeat(np.expand_dims(logs['Predicted_Facies'].values, 1), 100, 1)

	# Create subplot structure
	f, ax = plt.subplots(nrows=1, ncols=len(log_names) + 2, figsize=(15, 8),
						 gridspec_kw={'width_ratios': [1] * len(log_names) + [0.5, 0.5]})
	
	# Plot each log
	for count, item in enumerate(log_names):
		ax[count].plot(logs[item], logs['Depth'], color=log_colors[count])
		ax[count].set_xlabel(item)
		ax[count].set_xlim(logs[item].min(), logs[item].max())
		ax[count].set_ylim(logs['Depth'].min(), logs['Depth'].max())
		ax[count].invert_yaxis()
		if count > 0:
			ax[count].set_yticklabels([])
		else:
			ax[count].set_ylabel('Depth')

	# Display actual facies
	ax[-2].imshow(actual_cluster, interpolation='none', aspect='auto', cmap=cmap_facies, vmin=0, vmax=len(facies_colors)-1)
	ax[-2].set_xticklabels([])
	ax[-2].set_yticklabels([])
	
	# Display predicted facies
	ax[-1].imshow(predicted_cluster, interpolation='none', aspect='auto', cmap=cmap_facies, vmin=0, vmax=len(facies_colors)-1)
	ax[-1].set_xticklabels([])
	ax[-1].set_yticklabels([])

	# Configure color bar
	cbar = plt.colorbar(ax[-1].images[0], ax=ax[-1], orientation='vertical')
	cbar.set_ticks(np.arange(0.5, len(facies_colors) + 0.5))  # Set ticks in the middle of color regions
	cbar.set_ticklabels(lithofacies)  # Use lithofacies names as labels

	# Set the plot title and adjust layout
	f.suptitle(f'Well: {logs.iloc[0]["Well Name"]}', y=0.95)
	plt.subplots_adjust(wspace=0.1)
	plt.savefig(f'figure_out/{save_file}.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
	plt.show()

#-----------------------------------------------------------------------------------------#

def plot_facies(well_log, marker_dict):
	plt.figure()
	for facies, marker in marker_dict.items():
		subset = well_log[well_log['Facies Label'] == facies]
		plt.scatter(subset['GR'], subset['ILD_log10'], label=facies, 
					alpha=1, s=10, marker=marker)  # Use marker from the dictionary

	plt.xlabel('GR')
	plt.ylabel('ILD_log10')
	plt.title('Scatter Plot of GR vs ILD_log10 Colored by Facies')
	plt.legend(title='Lithofacies')
	plt.savefig(f'figure_out/featue_scatter.png', format='png', bbox_inches='tight', transparent=True, pad_inches=0)
	plt.show()

#-----------------------------------------------------------------------------------------#

def linear_model(x, omega_0, omega):
	return omega_0 + np.dot(omega, x)

#-----------------------------------------------------------------------------------------#

def plot_rock(df, predicted_density=None):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Rock'], df['Random Density'], color='orange', edgecolor='black', label='True Rock Density')
    if predicted_density is not None:
        plt.plot(df['Rock'], predicted_density, color='green', marker='x', markeredgecolor='red', linestyle='-',
				 label='Predicted Density')
    plt.ylabel(r'Density ($\mathrm{kg/m^3}$)')
    plt.xticks(rotation=90)
    plt.legend(loc='lower right')
    plt.savefig(f'figure_out/linear_model.png', format='png', bbox_inches='tight', transparent=True, pad_inches=0.1)
    plt.show()

#-----------------------------------------------------------------------------------------#

def normalize(value, original_min, original_max, new_min, new_max):
	return (value - original_min) / (original_max - original_min) * (new_max - new_min) + new_min

def vector(normalized_weights, normalized_bubble_teas, x, y, u, v, weights, bubble_teas):
	fig, ax = plt.subplots()
	plt.scatter(normalized_weights, normalized_bubble_teas, s=40, color='green', marker='o',
				linewidths=1, edgecolors='black')
	plt.quiver(x, y, u, v, color='orange')
	# for i in range(len(weights)):
	# 	plt.text(normalized_weights[i], normalized_bubble_teas[i]
		# f'({int(weights[i])}, {int(bubble_teas[i])})', ha='left', fontsize=8)
	ax.xaxis.set_ticks([])
	ax.yaxis.set_ticks([])
	ax.set_aspect('equal')
	plt.grid(True)
	plt.xlabel('Weight')
	plt.ylabel('Bubble Teas')
	plt.savefig(f'figure_out/bubble_tea.png', format='png', bbox_inches='tight',
				transparent=True, pad_inches=0.1)
	plt.show()

#-----------------------------------------------------------------------------------------#