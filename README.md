
 # 🧠 Task 4: Classification with Logistic Regression
# 📋 Objective

Build a binary classifier using Logistic Regression to predict whether a tumor is Malignant (1) or Benign (0) based on various diagnostic features from the Breast Cancer dataset.

# 🧰 Tools and Libraries Used

Python 3.x

Pandas – Data manipulation and preprocessing

NumPy – Numerical computations

Matplotlib / Seaborn – Data visualization

Scikit-learn (sklearn) – Machine learning model building and evaluation

# 📂 Dataset

Dataset Name: Breast Cancer Diagnostic Data (CSV file)
Path (example):

C:\evulvate internship\task4 dataset\data.csv

Example Columns:

id, radius_mean, texture_mean, smoothness_mean, symmetry_mean, etc.

diagnosis — Target column (M = Malignant, B = Benign)

# ⚙️ Steps and Workflow
1. Import Libraries

All necessary Python libraries such as pandas, numpy, matplotlib, and sklearn are imported.

2. Load Dataset
df = pd.read_csv(r"C:\evulvate internship\task4 dataset\data.csv")


The dataset is loaded using Pandas and basic exploration (head(), info(), isnull().sum()) is performed.

3. Data Cleaning & Encoding

Dropped unnecessary columns like id or unnamed columns.

Encoded the diagnosis column:

M → 1 (Malignant)

B → 0 (Benign)

Converted all categorical features into numeric form using pd.get_dummies().

4. Feature Scaling

Used StandardScaler() to normalize features for better model performance.

5. Train-Test Split

Split data into training (80%) and testing (20%) subsets:

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

6. Logistic Regression Model

Trained a Logistic Regression model with:

log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

7. Model Evaluation

Confusion Matrix – to evaluate classification accuracy

Classification Report – shows precision, recall, F1-score

ROC Curve and AUC – measures overall classification performance

8. Threshold Tuning

Experimented with custom decision thresholds (default = 0.5, example = 0.4) to analyze trade-offs between precision and recall.

9. Sigmoid Function Visualization

Plotted the sigmoid function to explain how logistic regression converts linear outputs to probabilities:

𝜎
(
𝑧
)
=
1
1
+
𝑒
−
𝑧
σ(z)=
1+e
−z
1
	​

# 📈 Model Performance (Expected)
Metric	Typical Range
Accuracy	95–98%
ROC-AUC	0.98–0.99
Precision	High for both classes
Recall	High, especially for Malignant detection
🧩 Key Concepts
Logistic Regression

​


It classifies data points by comparing predicted probability against a threshold (commonly 0.5).

# ROC Curve

Plots True Positive Rate (Recall) vs. False Positive Rate.

The closer the curve is to the top-left corner, the better the model.

AUC (Area Under Curve) indicates the quality of classification (1.0 = perfect model).

🧾 Outputs and Visuals
# Confusion Matrix
Displays correct vs incorrect predictions.

ROC Curve
Demonstrates classifier’s ability to distinguish between classes.

Sigmoid Function Plot
Shows probability behavior of logistic regression.

# 💡 Conclusion

Logistic Regression effectively classifies tumors with high accuracy and AUC score.

Feature scaling and encoding were crucial for model performance.

Threshold tuning helps adjust the model for higher recall or higher precision depending on the clinical requirement.

# 🖥️ How to Run

Install dependencies:

pip install pandas numpy matplotlib seaborn scikit-learn


Update the dataset path in the code:

df = pd.read_csv(r"C:\evulvate internship\task4 dataset\data.csv")


Run the Python script in Jupyter Notebook or VS Code.
