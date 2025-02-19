While participating in the four-month-long free online course in Machine Learning Zoomcamp delivered by DataTalks, I learned that as an ML practitioner, a significant portion of the work is not just about selecting the right model but about constructing datasets and performing feature engineering.

The process of exploring, describing, and analyzing datasets reveals inadequacies in data quality, which must be addressed before training a model. Nothing beats a high-quality dataset.

During the course, model selection depended on the nature of the target variable:
- Regression models predicted continuous values (e.g., house prices).
- Classification models determined categorical outcomes (e.g., whether a student gets admitted to college based on academic and extracurricular characteristics).
- Neural networks tackled more complex tasks, such as image classification.

We explored models such as logistic regression, decision trees, random forests, gradient boosting, and XGBoost, fine-tuning their parameters to improve performance.

However, two key aspects stood out to me beyond the core curriculum, which I want to highlight:

### 1. The Importance of Normalization in Model Performance

During training, I realized that not all models require feature scaling. However, for some models, failing to normalize the data can lead to poor performance or misleading coefficients.

#### When Should You Normalize Data?

Feature scaling (such as standardization or min-max scaling) is especially important when using:
✅ K-Nearest Neighbors (KNN) – Distance-based models can be skewed by unscaled features.
✅ Support Vector Machines (SVM) – A large range of feature values affects how the margin is calculated.
✅ Neural Networks – Can struggle with unnormalized inputs, leading to slower convergence.
✅ Linear & Logistic Regression (with Regularization) – Regularization techniques (e.g., Lasso, Ridge) assume features are on the same scale.

Meanwhile, models such as decision trees, random forests, and gradient boosting do not require normalization since they split data based on feature values, not distances.

#### Code Example: The Impact of Normalization on Logistic Regression

Below is a simple example demonstrating how failing to normalize data can impact logistic regression performance when regularization is applied.

