import pandas as pd

# Load the dataset
file_path = "attention_detection_dataset_v1.csv"
df = pd.read_csv(file_path)

# Display the first few rows to understand the structure
df.head()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report

# Separate features and label
X = df.drop(columns=["label"])
y = df["label"]

# Identify categorical and numerical columns
categorical_cols = ["pose"]
numerical_cols = X.columns.difference(categorical_cols)

# Preprocessing: One-hot encode categorical, passthrough numerical
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numerical_cols)
    ]
)

# Build pipeline with RandomForestClassifier
clf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
clf_pipeline.fit(X_train, y_train)

# Predict probabilities
y_proba = clf_pipeline.predict_proba(X_test)[:, 1]
y_pred = clf_pipeline.predict(X_test)

# Evaluate
roc_auc = roc_auc_score(y_test, y_proba)
report = classification_report(y_test, y_pred, output_dict=True)

print(roc_auc, report)
