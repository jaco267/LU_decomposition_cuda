import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data provided in the markdown table
data = {
    'n': [200, 300, 400, 500, 600, 1000, 1200, 1300, 1600, 2000, 4000, 8000, 16000, 20000, 25000, 30000, 31000],
    # Use np.nan for missing values
    'cpu': [7, 25, 59, 121, 201, 942, 1618, 2124, 3997, 7990, 70105, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    'block LU (cuda)': \
          [25, 31, 34, 39, 41, 59, 68, 72, 88, 104, 282, 1025, 4469, 7388, 12421, 19289, 20913]
}

df = pd.DataFrame(data)

# Rename the CUDA column for a shorter legend label
df = df.rename(columns={'block LU (cuda)': 'cuda'})

plt.figure(figsize=(10, 6))

# Plot CPU data
plt.plot(df['n'], df['cpu'], label='CPU', marker='o', linestyle='-')

# Plot CUDA data
plt.plot(df['n'], df['cuda'], label='Block LU (CUDA)', marker='x', linestyle='--')

# Use log-log scale to better visualize the performance data across a wide range of 'n' and time values
plt.xscale('log')
plt.yscale('log')

plt.title('Performance Comparison (Log-Log Scale)')
plt.xlabel('$n$')
plt.ylabel('Time (units, ms)') # Assuming the units are in ms
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.legend()

# Save the plot
plt.savefig('performance_plot.png')