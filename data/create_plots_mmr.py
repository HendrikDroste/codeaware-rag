import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# File paths
baseline_file = 'results_baseline.csv'
custom_split_file = 'results_custom_split.csv'
llm_embedding_file = 'results_llm_embedding.csv'

# Function to load and clean CSV data
def load_csv(file_path):
    # Now that files have headers, read with header=0
    df = pd.read_csv(file_path)

    # Remove huggingface/ and vendor name, keeping only the model name
    df['Name'] = df['Name'].apply(lambda x: x.split('/')[-1])

    return df

# Load data
baseline_df = load_csv(baseline_file)
custom_split_df = load_csv(custom_split_file)
llm_embedding_df = load_csv(llm_embedding_file)

# Filter out specific models from baseline data
baseline_df = baseline_df[~baseline_df['Name'].isin(['codet5p-110m-embedding', 'unixcoder-base'])]

# Create output directory if it doesn't exist
output_dir = 'plots'
os.makedirs(output_dir, exist_ok=True)

# Add annotation for all plots
annotation_text = "MRR: Higher values are better\nDataset size: n=18"

# Sort models by baseline MRR values in descending order - This will be our reference order
sorted_baseline_df = baseline_df.sort_values('MRR', ascending=False)
sorted_model_order = sorted_baseline_df['Name'].tolist()

# 1. Create baseline chart
plt.figure(figsize=(12, 6))
models = sorted_baseline_df['Name']
mmr_values = sorted_baseline_df['MRR']

bars = plt.bar(models, mmr_values, color='skyblue')
plt.xlabel('Models')
plt.ylabel('MRR')
#plt.title('Baseline MRR by Model')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 0.5)  # Set y-axis to end at 0.5
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add annotation about MRR and dataset size
plt.figtext(0.01, 0.01, annotation_text, fontsize=9)

# Display values above bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.3f}',
             ha='center', va='bottom', rotation=0)

plt.savefig(os.path.join(output_dir, 'baseline_mmr.png'), dpi=300)

# 2. Custom Split vs Baseline grouped bar chart
# Find common models by comparing the model parts (last part of path)
common_models = []
for model in sorted_model_order:  # Use the same order from baseline
    if model in custom_split_df['Name'].values:
        common_models.append(model)

# Prepare data for common models
custom_split_values = []
baseline_values = []

for model in common_models:
    custom_split_values.append(float(custom_split_df[custom_split_df['Name'] == model]['MRR'].values[0]))
    baseline_values.append(float(baseline_df[baseline_df['Name'] == model]['MRR'].values[0]))

# Create grouped bar chart
plt.figure(figsize=(12, 6))
x = np.arange(len(common_models))
width = 0.35

plt.bar(x - width/2, custom_split_values, width, label='Custom Split', color='coral')
plt.bar(x + width/2, baseline_values, width, label='Baseline', color='skyblue')

plt.xlabel('Models')
plt.ylabel('MRR')
#plt.title('Custom Split vs Baseline MRR Comparison')
plt.xticks(x, common_models, rotation=45, ha='right')
plt.ylim(0, 0.5)  # Set y-axis to end at 0.5
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Add annotation about MRR and dataset size
plt.figtext(0.01, 0.01, annotation_text, fontsize=9)

# Display values above bars
for i, v in enumerate(custom_split_values):
    plt.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

for i, v in enumerate(baseline_values):
    plt.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

plt.savefig(os.path.join(output_dir, 'custom_split_vs_baseline.png'), dpi=300)

# 3. LLM Embedding vs Baseline grouped bar chart
# Find common models using the same order from baseline
common_models = []
for model in sorted_model_order:  # Use the same order from baseline
    if model in llm_embedding_df['Name'].values:
        common_models.append(model)

# Prepare data for common models
llm_embedding_values = []
baseline_values = []

for model in common_models:
    llm_embedding_values.append(float(llm_embedding_df[llm_embedding_df['Name'] == model]['MRR'].values[0]))
    baseline_values.append(float(baseline_df[baseline_df['Name'] == model]['MRR'].values[0]))

# Create grouped bar chart
plt.figure(figsize=(12, 6))
x = np.arange(len(common_models))
width = 0.35

plt.bar(x - width/2, llm_embedding_values, width, label='LLM Embedding', color='lightgreen')
plt.bar(x + width/2, baseline_values, width, label='Baseline', color='skyblue')

plt.xlabel('Models')
plt.ylabel('MRR')
#plt.title('LLM Embedding vs Baseline MRR Comparison')
plt.xticks(x, common_models, rotation=45, ha='right')
plt.ylim(0, 0.5)  # Set y-axis to end at 0.5
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Add annotation about MRR and dataset size
plt.figtext(0.01, 0.01, annotation_text, fontsize=9)

# Display values above bars
for i, v in enumerate(llm_embedding_values):
    plt.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

for i, v in enumerate(baseline_values):
    plt.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

plt.savefig(os.path.join(output_dir, 'llm_embedding_vs_baseline.png'), dpi=300)

# Optional: Show plots
# plt.show()

print("Charts have been saved in the 'plots' directory.")
