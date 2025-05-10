import time
import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
import re

start_time = time.time()

def filter_location(location):
    result = location.split(",")
    if len(result) > 1:
        return result[1][1:]
    else:
        return location

# data = pd.read_excel("Week8/job_dataset.ods", engine="odf", dtype=str)
data = pd.read_excel("job_dataset.ods", engine="odf", dtype=str)

# profile = ProfileReport(df, title="Jobs DS Report")
# profile.to_file("Jobs DS Report.html")

# Drop NA and duplicate rows and filter "location"
data = data.dropna(axis=0)
data = data.drop_duplicates()
data["location"] = data["location"].apply(filter_location)

# Remove special characters (keep letters, numbers, spaces)
data["description"] = data["description"].str.replace(r"[^a-zA-Z0-9\s]", "", regex=True)

# Train_Test_Split
target = "career_level"
x = data.drop(target, axis=1)
y = data[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# Compute class weights
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, class_weights))

"""Bring out TfidfVectorizer for "description" from ColumnTransformer() reduce processing time from 56s to 17s"""
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=0.01, max_df=0.99)
# vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
processed_data = vectorizer.fit_transform(x_train["description"])
print(processed_data.shape)

# Preprocessor Transformer
preprocessor = ColumnTransformer(transformers=[
    ("title", TfidfVectorizer(stop_words="english", ngram_range=(1, 1)), "title"),
    ("location", OneHotEncoder(handle_unknown="ignore"), ["location"]),
    # ("description", TfidfVectorizer(stop_words="english", ngram_range=(1, 2)), "description"),
    ("function", OneHotEncoder(), ["function"]),
    ("industry", TfidfVectorizer(stop_words="english", ngram_range=(1, 1)), "industry"),
])

# Pipeline
pipeline = Pipeline(steps=[
    ("pre_processor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42, class_weight=class_weight_dict))
])

# Grid search parameters
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20],
    'classifier__min_samples_split': [2, 5]
}

# Grid search with timing
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1_weighted', verbose=1, n_jobs=-1)
grid_search.fit(x_train, y_train)

# Evaluation
y_pred = grid_search.predict(x_test)
print(classification_report(y_test, y_pred))
print("Best Parameters:", grid_search.best_params_)
print("--- %s seconds ---" % (time.time() - start_time))

"""Without min_df=0.01, max_df=0.99 in TfidfVectorizer, the size of vocabulary is 921105 and runtime is 16.46s below"""
# (6441, 921105)
#                                         precision    recall  f1-score   support
#
#                         bereichsleiter       0.45      0.55      0.50       192
#          director_business_unit_leader       0.67      0.71      0.69        14
#                    manager_team_leader       0.68      0.53      0.59       533
# managing_director_small_medium_company       0.00      0.00      0.00         1
#   senior_specialist_or_project_manager       0.81      0.87      0.84       865
#                             specialist       0.18      0.50      0.26         6
#
#                               accuracy                           0.71      1611
#                              macro avg       0.46      0.53      0.48      1611
#                           weighted avg       0.72      0.71      0.71      1611
#
# Best Parameters: {'classifier__max_depth': 20, 'classifier__min_samples_split': 2, 'classifier__n_estimators': 200}
# --- 16.465654134750366 seconds ---

"""With min_df=0.01, max_df=0.99 in TfidfVectorizer, the size of vocabulary is 4022 and runtime is 14.32s below"""
"""The performance do not significantly change."""
# (6441, 4022)
#                                         precision    recall  f1-score   support
#
#                         bereichsleiter       0.45      0.55      0.50       192
#          director_business_unit_leader       0.67      0.71      0.69        14
#                    manager_team_leader       0.68      0.53      0.59       533
# managing_director_small_medium_company       0.00      0.00      0.00         1
#   senior_specialist_or_project_manager       0.81      0.87      0.84       865
#                             specialist       0.18      0.50      0.26         6
#
#                               accuracy                           0.71      1611
#                              macro avg       0.46      0.53      0.48      1611
#                           weighted avg       0.72      0.71      0.71      1611
#
# Best Parameters: {'classifier__max_depth': 20, 'classifier__min_samples_split': 2, 'classifier__n_estimators': 200}
# --- 14.32004714012146 seconds ---