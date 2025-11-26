import pandas as pd
import config
from huggingface_hub import hf_hub_download

# Download and load the TSV
path = hf_hub_download(repo_id=config.REPO_ID, filename=config.FILENAME_TSV, repo_type="dataset")
df = pd.read_csv(path, sep="\t")

# Print the unique values in the 'fold' column
print("Unique splits found in TSV:", df['fold'].unique())