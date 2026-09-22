
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)



# 1. Loading Data.

print("1. LOADING DATA")


df = pd.read_csv("products.csv")

print(f"Number of rows: {len(df)}")
print(f"Number of columns: {len(df.columns)}")

print("\nFirst 5 rows:")
print(df.head())

print("\nOriginal columns:")
print(df.columns.tolist())



# 2. Cleaning column names.


print("2. CLEANING COLUMN NAMES")


# Remove unnecessary spaces from column names
df.columns = df.columns.str.strip()

print("Cleaned columns:")
print(df.columns.tolist())



# 3. Basic data information.


print("3. DATA INFORMATION")


print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nNumber of duplicate rows:")
print(df.duplicated().sum())



# 4. Cleaning product title column.



print("4. CLEANING PRODUCT TITLE")


# Convert Product Title to string
df["Product Title"] = df["Product Title"].astype("string")

# Remove leading/trailing spaces
df["Product Title"] = df["Product Title"].str.strip()

# Replace multiple spaces with a single space
df["Product Title"] = df["Product Title"].str.replace(
    r"\s+",
    " ",
    regex=True
)

# Remove rows where Product Title is missing
df = df.dropna(subset=["Product Title"])

# Remove empty titles
df = df[df["Product Title"] != ""]

print(f"Rows after Product Title cleaning: {len(df)}")



# 5. Cleaning category label column.



print("5. CLEANING CATEGORY LABEL")


# Convert category to string
df["Category Label"] = df["Category Label"].astype("string")

# Remove leading/trailing spaces
df["Category Label"] = df["Category Label"].str.strip()

# Remove rows where the target category is missing
df = df.dropna(subset=["Category Label"])

# Remove empty categories
df = df[df["Category Label"] != ""]

print("Categories before checking:")
print(df["Category Label"].value_counts())



# 6. Standardizing category labels.



print("6. STANDARDIZING CATEGORY LABELS")


category_mapping = {
    "CPU": "CPUs",
    "Mobile Phone": "Mobile Phones",
    "fridge": "Fridges"
}

df["Category Label"] = df["Category Label"].replace(category_mapping)

print("\nCategories after standardization:")
print(df["Category Label"].value_counts())



# 7. Removing duplicates.



print("7. REMOVING DUPLICATES")


before_duplicates = len(df)

df = df.drop_duplicates()

after_duplicates = len(df)

print(f"Rows before removing duplicates: {before_duplicates}")
print(f"Rows after removing duplicates:  {after_duplicates}")
print(f"Duplicates removed:              {before_duplicates - after_duplicates}")



# 8. Dataset summary after cleaning.



print("8. CLEAN DATASET SUMMARY")


print(f"Final number of rows: {len(df)}")
print(f"Final number of columns: {len(df.columns)}")

print("\nRemaining missing values:")
print(df.isnull().sum())

print("\nCategory distribution:")
print(df["Category Label"].value_counts())



# 9. Vizualizing category distribution.


plt.figure(figsize=(12, 6))

sns.countplot(
    data=df,
    y="Category Label",
    order=df["Category Label"].value_counts().index
)

plt.title("Product Category Distribution")
plt.xlabel("Number of products")
plt.ylabel("Category")

plt.tight_layout()
plt.show()



# 10. Feature engineering.

print("10. FEATURE ENGINEERING")


# Number of characters in the title
df["title_char_count"] = df["Product Title"].str.len()

# Number of words in the title
df["title_word_count"] = df["Product Title"].str.split().str.len()

# Number of digits in the title
df["title_digit_count"] = df["Product Title"].str.count(r"\d")

# Number of special characters
df["title_special_count"] = df["Product Title"].str.count(
    r"[^a-zA-Z0-9\s]"
)

# Longest word in the title
df["title_longest_word"] = (
    df["Product Title"]
    .str.split()
    .apply(
        lambda words: max(
            [len(word) for word in words],
            default=0
        )
    )
)

print("\nExample engineered features:")
print(
    df[
        [
            "Product Title",
            "title_char_count",
            "title_word_count",
            "title_digit_count",
            "title_special_count",
            "title_longest_word"
        ]
    ].head()
)



# 11. Preparing data for machine learning.



print("11. PREPARING DATA FOR MACHINE LEARNING")


X = df["Product Title"]
y = df["Category Label"]

print(f"Number of samples: {len(X)}")
print(f"Number of categories: {y.nunique()}")

print("\nCategories:")
print(sorted(y.unique()))



# 12. TRAIN / TEST SPLIT.


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))



# 13. Defining machine learning models.


models = {

    "Logistic Regression": Pipeline([
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
            LogisticRegression(
                max_iter=1000
            )
        )
    ]),

    "Naive Bayes": Pipeline([
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
            MultinomialNB()
        )
    ]),

    "Linear SVM": Pipeline([
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
    ]),

    "Random Forest": Pipeline([
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
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            )
        )
    ])
}



# 14. Training and comparing models.

print("14. TRAINING AND COMPARING MODELS")


results = {}

for name, model in models.items():


    print(f"Training: {name}")


    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    results[name] = accuracy

    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )



# 15. Model comparison.

print("15. MODEL COMPARISON")


results_df = pd.DataFrame(
    {
        "Model": list(results.keys()),
        "Accuracy": list(results.values())
    }
)

results_df = results_df.sort_values(
    "Accuracy",
    ascending=False
)

print(results_df.to_string(index=False))


# 16. Visualizing model performance.

plt.figure(figsize=(10, 6))

sns.barplot(
    data=results_df,
    x="Accuracy",
    y="Model"
)

plt.xlim(0, 1)
plt.title("Model Accuracy Comparison")
plt.xlabel("Accuracy")
plt.ylabel("Model")

plt.tight_layout()
plt.show()



# 17. Selecting best model.


best_model_name = results_df.iloc[0]["Model"]
best_accuracy = results_df.iloc[0]["Accuracy"]

best_model = models[best_model_name]

print("\n" + "=" * 60)
print("17. BEST MODEL")
print("=" * 60)

print(f"Best model: {best_model_name}")
print(f"Accuracy: {best_accuracy:.4f}")



# 18. Confusion matrix for the selected model.


best_predictions = best_model.predict(X_test)

cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=sorted(y_test.unique())
)

plt.figure(figsize=(12, 10))

sns.heatmap(
    cm,
    annot=False,
    cmap="Blues",
    xticklabels=sorted(y_test.unique()),
    yticklabels=sorted(y_test.unique())
)

plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted Category")
plt.ylabel("Actual Category")

plt.tight_layout()
plt.show()



# 19. Final classification report.

print("19. FINAL CLASSIFICATION REPORT")

print(
    classification_report(
        y_test,
        best_predictions,
        zero_division=0
    )
)




