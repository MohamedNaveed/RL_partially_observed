import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

params = {'axes.labelsize':14,
            'font.size':14,
            'legend.fontsize':14,
            'xtick.labelsize':12,
            'ytick.labelsize':12,
            'text.usetex':True,
            'figure.figsize':[12,8]}
plt.rcParams.update(params)
# Set font type to Type 42 (TrueType) for embedding in PDF
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

# Load the data from the CSV file
path = '../data/sac_cartpole/'
file_name = 'sac_cartpole_output_v4_buffer10_5_1M'
file_path = f"{path}/{file_name}.csv"  # Replace with your actual file path
data = pd.read_csv(file_path)

# Plotting
plt.figure()

cost_vec = []

for cost in data['cost']:
    cost_vec.append(float(cost.strip('[]')))

len_episodes = np.arange(1,len(data['step'])*100,100)

# Plot episode cost
plt.subplot(3, 1, 1)
plt.plot(len_episodes, cost_vec, color='blue', linewidth = 3, label='Episode Cost')
plt.xlabel('Episode')
plt.ylabel('Episode Cost')
plt.title('SAC Training Metrics Buffer size = 1e5')
plt.grid()
plt.legend()

# Plot Q-function loss (qf1_loss)
plt.subplot(3, 1, 2)
plt.plot(len_episodes, data['qf_loss'], color='red',linewidth = 3, label='Critic Loss')
#plt.yscale('log')
plt.xlabel('Episode')
plt.ylabel('QF Loss')
plt.grid()
plt.legend()

# Plot actor loss
plt.subplot(3, 1, 3)
plt.plot(len_episodes, data['actor_loss'], color='green',linewidth = 3, label='Actor Loss')
plt.xlabel('Episode')
plt.ylabel('Actor Loss')
plt.grid()
plt.legend()

# Adjust layout
plt.tight_layout()

# Save the plot as a PDF with embedded fonts
plt.savefig(f"../plots/SAC_training_metrics_{file_name}.pdf", format='pdf', bbox_inches='tight')

# Display the plot (optional)
#plt.show()
