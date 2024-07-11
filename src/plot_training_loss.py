import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
csv_file = '../data/cartpole_converged.csv'  # Replace with your CSV file path
data = pd.read_csv(csv_file)

# Display the first few rows of the dataframe to verify
print(data.head())

# Ensure 'global_step' and 'training_loss' (or equivalent) columns exist
if 'step' in data.columns and 'qf1_loss' in data.columns:
    # Plot the training loss
    plt.figure(figsize=(10, 5))
    plt.plot(data['step'], data['qf1_loss'], label='Training Loss')
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
    plt.plot(data['step'], data['actor_loss'], label='Q value')
    plt.xlabel('Global Steps')
    plt.ylabel('Actor Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("Required columns ('step' and 'actor_loss') not found in the CSV file.")
