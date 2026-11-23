from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

project_folder = Path(__file__).resolve().parents[1]
data_file = project_folder / "data" / "complaints.csv"
model_file = project_folder / "models" / "ticket_classifier.joblib"

# Load complaint text and its correct category.
data = pd.read_csv(data_file)
complaints = data["complaint"]
categories = data["category"]

# Keep some complaints aside so we can check the model on unseen examples.
x_train, x_test, y_train, y_test = train_test_split(
    complaints,
    categories,
    test_size=0.2,
    random_state=42,
    stratify=categories,
)

# Turn words into numbers, then learn which patterns match each category.
model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2)),
    MultinomialNB(),
)

model.fit(x_train, y_train)

# Check predictions against the examples kept aside.
predictions = model.predict(x_test)
results = pd.DataFrame({
    "complaint": x_test.to_numpy(),
    "correct category": y_test.to_numpy(),
    "model prediction": predictions,
})

print("\nThe six held-aside examples:")
print(results.to_string(index=False))
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
print(classification_report(y_test, predictions, zero_division=0))

# Save the trained model so the Flask app can use it later.
joblib.dump(model, model_file)
print(f"Model saved to: {model_file}")