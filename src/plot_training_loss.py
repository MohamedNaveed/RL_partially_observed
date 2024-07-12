import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Read the CSV file
csv_file = '../data/cartpole_po_nz_2_q_2.csv'  # Replace with your CSV file path
data = pd.read_csv(csv_file)

# Display the first few rows of the dataframe to verify
print(data.head())
idx = [0,data['qf1_loss'].size]
steps = np.arange(data['qf1_loss'].size) * (data['step'][1] - data['step'][0])
# Ensure 'global_step' and 'training_loss' (or equivalent) columns exist
if 'step' in data.columns and 'qf1_loss' in data.columns:
    # Plot the training loss
    plt.figure(figsize=(10, 5))
    plt.plot(steps, data['qf1_loss'][idx[0]:idx[1]], label='Training Loss')
    plt.yscale('log')  # Set y-axis scale to logarithmic
    plt.xlabel('Global Steps')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("Required columns ('step' and 'qf1_loss') not found in the CSV file.")


if 'step' in data.columns and 'actor_loss' in data.columns:
    # Plot the training loss
    plt.figure(figsize=(10, 5))
    plt.plot(steps, data['actor_loss'][idx[0]:idx[1]], label='Q value')
    plt.xlabel('Global Steps')
    plt.ylabel('Actor Loss')
    plt.title('Actor Loss Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("Required columns ('step' and 'actor_loss') not found in the CSV file.")
