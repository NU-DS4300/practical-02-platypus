import pandas as pd
import re
import matplotlib.pyplot as plt

# Load the CSV file (replace 'responses.csv' with the actual file path)
df = pd.read_csv("results.csv", sep='|')

# Define correct answers as regex patterns
correct_answers = {
    "When was Redis released?": r"\b2009\b",
    "How many databases does Redis support?": r"\b16\b",
    "What kind of imbalances in an AVL tree require multiple rotations?": r"^(?=.*LR)(?=.*RL)",
    "What is the EC2 lifecycle?": r"(?=.*\bLaunch\b)(?=.*\b(?:Start|Stop)\b)(?=.*\bTerminate\b)(?=.*\bReboot\b)",
    "When was neo4j's graph query language invented?": r"\b2011\b",
    "Name the data types supported by Redis for values.": r"(?=.*\bStrings\b)(?=.*\bLists\b)(?=.*\bSets\b)(?=.*\bSorted Sets\b)(?=.*\bHashes\b)(?=.*\bGeospatial Data\b)"
}

# Function to check if response contains the correct answer
def check_correctness(response, question):
    pattern = correct_answers.get(question, "")
    if pattern and re.search(pattern, response, re.IGNORECASE):
        return True
    return False

# Apply correctness check
df["Correct"] = df.apply(lambda row: check_correctness(row["response"], row["question"]), axis=1)

# Save results
df.to_csv("validated_responses.csv", index=False)

# Function to plot graphs
def plot_results(data_series, title):
    # Convert series to DataFrame for plotting
    df_plot = pd.DataFrame({'Correct (%)': data_series})
    
    # Create the bar plot
    ax = df_plot.plot(kind='bar', figsize=(10, 6), color='green')
    
    # Set the y-axis to go from 0 to 100
    plt.ylim(0, 100)
    
    # Add value labels on top of each bar
    for i, v in enumerate(data_series):
        ax.text(i, v + 2, f'{v:.1f}%', ha='center')
    
    # Set titles and labels
    plt.title(title)
    plt.ylabel("Percentage Correct (%)")
    plt.xlabel("Category")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Group by database and get only correct percentages
db_group = df.groupby("database")["Correct"].mean() * 100  # mean of True/False gives proportion of True
plot_results(db_group, "Correct Responses by Database (%)")

# Group by embedding model
embed_group = df.groupby("embedding_model")["Correct"].mean() * 100
plot_results(embed_group, "Correct Responses by Embedding Model (%)")

# Group by chunk size
chunk_group = df.groupby("chunk_size")["Correct"].mean() * 100
plot_results(chunk_group, "Correct Responses by Chunk Size (%)")

# Group by overlap
overlap_group = df.groupby("overlap")["Correct"].mean() * 100
plot_results(overlap_group, "Correct Responses by Overlap (%)")

# Group by topk
topk_group = df.groupby("topk")["Correct"].mean() * 100
plot_results(topk_group, "Correct Responses by Top-K (%)")

# Group by topk
topk_group = df.groupby("llm")["Correct"].mean() * 100
plot_results(topk_group, "Correct Responses by LLM (%)")

# Group by topk
topk_group = df.groupby("prompt")["Correct"].mean() * 100
plot_results(topk_group, "Correct Responses by System Prompt (%)")

print("Validation and graphing complete. Results saved to validated_responses.csv.")
