import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report
)



# 1. Loading data.


df = pd.read_csv("products.csv")

print(f"Original number of rows: {len(df)}")
print(f"Original number of columns: {len(df.columns)}")



# 2. Cleaning column names.

print("2. CLEANING COLUMN NAMES")


df.columns = df.columns.str.strip()

print("Columns:")
print(df.columns.tolist())

# 3. Cleaning product title column.

print("3. CLEANING PRODUCT TITLES")

# Convert Product Title to string
df["Product Title"] = df["Product Title"].astype("string")

# Remove leading/trailing spaces
df["Product Title"] = df["Product Title"].str.strip()

# Replace multiple spaces with one space
df["Product Title"] = df["Product Title"].str.replace(
    r"\s+",
    " ",
    regex=True
)

# Remove missing titles
df = df.dropna(subset=["Product Title"])

# Remove empty titles
df = df[df["Product Title"] != ""]

print(f"Rows after title cleaning: {len(df)}")



# 4. Cleaning category labels column.

print("4. CLEANING CATEGORY LABELS")


# Convert category to string
df["Category Label"] = df["Category Label"].astype("string")

# Remove leading/trailing spaces
df["Category Label"] = df["Category Label"].str.strip()

# Remove missing categories
df = df.dropna(subset=["Category Label"])

# Remove empty categories
df = df[df["Category Label"] != ""]



# 5. Standardizing category labels.

print("5. STANDARDIZING CATEGORY LABELS")

category_mapping = {
    "CPU": "CPUs",
    "Mobile Phone": "Mobile Phones",
    "fridge": "Fridges"
}

df["Category Label"] = df["Category Label"].replace(
    category_mapping
)

print("\nFinal category distribution:")
print(df["Category Label"].value_counts())

# 6. Removing duplicates.

print("6. REMOVING DUPLICATES")

before_duplicates = len(df)

df = df.drop_duplicates()

after_duplicates = len(df)

print(f"Rows before duplicates removal: {before_duplicates}")
print(f"Rows after duplicates removal:  {after_duplicates}")
print(f"Duplicates removed:             {before_duplicates - after_duplicates}")



# 7. Preparing data for machine learning.

print("7. PREPARING DATA FOR MACHINE LEARNING")

# Input feature
X = df["Product Title"]

# Target variable
y = df["Category Label"]

print(f"Number of samples: {len(X)}")
print(f"Number of categories: {y.nunique()}")

print("\nCategories:")
print(sorted(y.unique()))



# 8. TRAIN / TEST SPLIT.


print("8. TRAIN / TEST SPLIT")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")



# 9. Creating linear svm pipeline.


print("9. CREATING LINEAR SVM MODEL")

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LinearSVC()
    )
])

print("Model: TF-IDF + Linear SVM")



# 10. Training the model.

print("10. TRAINING MODEL")

model.fit(X_train, y_train)

print("Training completed successfully!")



# 11. Evaluating the model.

print("11. MODEL EVALUATION")

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)



# 12. Saving the model.

print("12. SAVING MODEL")


model_filename = "product_category_model.pkl"

joblib.dump(
    model,
    model_filename
)

print(f"Model saved successfully as: {model_filename}")




