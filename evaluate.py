import pandas as pd
import sys
from sklearn.metrics import accuracy_score

# Check command-line arguments
if len(sys.argv) < 3:
    print("Usage: python evaluate.py <submission_csv> <labels_csv>")
    sys.exit(1)

submission_csv = sys.argv[1]  # e.g., submission/team1.csv
labels_csv = sys.argv[2]      # e.g., private_data/test_labels.csv

# Load CSVs
submission = pd.read_csv(submission_csv)
labels = pd.read_csv(labels_csv)

# Merge on id
merged = pd.merge(labels, submission, on="id")

# Compute accuracy
acc = accuracy_score(merged["emotion"], merged["predicted_label"])
print(f"Accuracy: {acc*100:.2f}%")
