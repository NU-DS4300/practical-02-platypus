import os
import pandas as pd
def merge_result_files():
    """Merge all batch CSVs into a single final results file."""
    all_files = [f for f in os.listdir("results") if f.startswith("batch_") and f.endswith(".csv")]
    if not all_files:
        print("No batch files found to merge.")
        return
    
    # Sort files by batch number to maintain order
    all_files.sort(key=lambda x: int(x.split("_")[1]))
    
    # Read and concatenate all dataframes
    all_dfs = []
    for file in all_files:
        print(f"Reading {file}")
        df = pd.read_csv(os.path.join("results", file), sep="|")
        all_dfs.append(df)
    
    # Combine into one dataframe
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df.to_csv("results.csv", index=False, sep="|")
        print(f"Successfully merged {len(all_dfs)} batch files into results.csv")

if __name__ == "__main__":
    print("Hi")
    merge_result_files()