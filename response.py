import pandas as pd
import re
import matplotlib.pyplot as plt

# Load the CSV file (replace 'responses.csv' with the actual file path)
df = pd.read_csv("results.csv", sep='|')

# Define correct answers as regex patterns
correct_answers = {
    "When was Redis released?": r"\b2009\b",
    "How many databases does Redis support?": r"\b16\b",
    "What kind of imbalances in an AVL tree require multiple rotations?": r"\bLR\b|\bRL\b",
    "What is the EC2 lifecycle?": r"\bLaunch\b.*\bStart/stop\b.*\bTerminate\b.*\bReboot\b",
    "When was neo4j's graph query language invented?": r"\b2011\b",
    "Name the data types supported by Redis for values.": r"\bStrings\b.*\bLists\b.*\bSets\b.*\bSorted Sets\b.*\bHashes\b.*\bGeospatial Data\b"
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
def plot_results(grouped_df, title):
    grouped_df.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title(title)
    plt.ylabel("Number of Questions")
    plt.xlabel("Category")
    plt.legend(["Incorrect", "Correct"], loc="upper right")
    plt.xticks(rotation=45)
    plt.show()

# Group by database
db_group = df.groupby("database")["Correct"].value_counts().unstack(fill_value=0)
plot_results(db_group, "Performance by Database")

# Group by embedding model
embed_group = df.groupby("embedding_model")["Correct"].value_counts().unstack(fill_value=0)
plot_results(embed_group, "Performance by Embedding Model")

# Group by chunk size
chunk_group = df.groupby("chunk_size")["Correct"].value_counts().unstack(fill_value=0)
plot_results(chunk_group, "Performance by Chunk Size")

# Group by overlap
overlap_group = df.groupby("overlap")["Correct"].value_counts().unstack(fill_value=0)
plot_results(overlap_group, "Performance by Overlap")

# Group by topk
topk_group = df.groupby("topk")["Correct"].value_counts().unstack(fill_value=0)
plot_results(topk_group, "Performance by Top-K")

print("Validation and graphing complete. Results saved to validated_responses.csv.")
