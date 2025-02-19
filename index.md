While participating in the four-month-long free online course in Machine Learning Zoomcamp delivered by DataTalks, I learned that as an ML practitioner, a significant portion of the work is not just about selecting the right model but about constructing datasets and performing feature engineering.

The process of exploring, describing, and analyzing datasets reveals inadequacies in data quality, which must be addressed before training a model. Nothing beats a high-quality dataset.

During the course, model selection depended on the nature of the target variable:
	•	Regression models predicted continuous values (e.g., house prices).
	•	Classification models determined categorical outcomes (e.g., whether a student gets admitted to college based on academic and extracurricular characteristics).
	•	Neural networks tackled more complex tasks, such as image classification.

We explored models such as logistic regression, decision trees, random forests, gradient boosting, and XGBoost, fine-tuning their parameters to improve performance.

However, two key aspects stood out to me beyond the core curriculum, which I want to highlight:

1. The Importance of Normalization in Model Performance

During training, I realized that not all models require feature scaling. However, for some models, failing to normalize the data can lead to poor performance or misleading coefficients.

When Should You Normalize Data?

Feature scaling (such as standardization or min-max scaling) is especially important when using:
✅ K-Nearest Neighbors (KNN) – Distance-based models can be skewed by unscaled features.

✅ Support Vector Machines (SVM) – A large range of feature values affects how the margin is calculated.

✅ Neural Networks – Can struggle with unnormalized inputs, leading to slower convergence.

✅ Linear & Logistic Regression (with Regularization) – Regularization techniques (e.g., Lasso, Ridge) assume features are on the same scale.

Meanwhile, models such as decision trees, random forests, and gradient boosting do not require normalization since they split data based on feature values, not distances.

Code Example: The Impact of Normalization on Logistic Regression

Below is a simple example demonstrating how failing to normalize data can impact logistic regression performance when regularization is applied.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Generate synthetic dataset
np.random.seed(42)
X = np.random.rand(1000, 2) * [100, 1]  # One feature with a large range, one with a small range
y = (X[:, 0] + X[:, 1] > 50).astype(int)  # Simple threshold-based target

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression without normalization
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy without normalization:", accuracy_score(y_test, y_pred))

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression with normalization
model_scaled = LogisticRegression()
model_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = model_scaled.predict(X_test_scaled)
print("Accuracy with normalization:", accuracy_score(y_test, y_pred_scaled))

Observations:
	•	Without normalization, the model struggles to assign balanced weights to features.
	•	After normalization, regularization techniques work as intended, leading to improved accuracy.

This experiment highlights why normalization is critical for models using regularization.

2. The Art of Feature Engineering & Feature Selection

Another major takeaway was feature selection—how do you determine which features to include in a model?

While it might seem logical to include as many features as possible, irrelevant or redundant features can degrade model performance. Instead, it’s useful to:
✅ Start with feature importance scores (e.g., from decision trees).
✅ Train a baseline model with the most important features.
✅ Gradually add features and track performance metrics (e.g., ROC AUC, Recall, Precision).

Mini Case Study: Feature Engineering in Action

Imagine we’re predicting whether a student will be placed in a job based on academic and extracurricular characteristics.

We have:
	•	Numerical features: GPA, internship hours, project count.
	•	Categorical features: Participation in extracurricular activities (yes/no).

A naive approach might involve feeding raw data into a model. However, domain knowledge suggests that combining certain features might improve predictive power.

Feature Interaction: Creating a New Feature

By crossing features, we can create new, meaningful variables. Here, we introduce a new feature:

df['gpa_project_ratio'] = df['gpa'] / (df['project_count'] + 1)  # Avoid division by zero

This GPA-to-Project ratio captures how well students balance academics with practical experience.

Evaluating Feature Importance

To determine if our new feature is valuable, we can use Random Forest feature importance:

from sklearn.ensemble import RandomForestClassifier

X = df.drop(columns=['placement_status'])  # Exclude target variable
y = df['placement_status']

model = RandomForestClassifier()
model.fit(X, y)

# Get feature importance scores
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print(importances)

If the gpa_project_ratio has high importance, we keep it. Otherwise, we remove it.

# Final Thoughts

Through this journey, I’ve come to appreciate data preprocessing, normalization, and feature engineering as critical steps in model performance. While model selection is important, garbage in, garbage out still applies—quality data leads to quality models.
