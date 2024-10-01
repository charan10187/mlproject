from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

# Initialize Flask app
app = Flask(__name__)

# Load and preprocess the data
data = pd.read_csv('C:/Users/srich/OneDrive/Desktop/project/HR_comma_sep.csv')

# Clean the data: Drop missing values and duplicates
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)

# Encode categorical features
le_dept = LabelEncoder()
data['Department'] = le_dept.fit_transform(data['Department'])
le_salary = LabelEncoder()
data['salary'] = le_salary.fit_transform(data['salary'])

# Scale numerical features
scaler = StandardScaler()
numerical_features = ['satisfaction_level', 'last_evaluation', 'number_project', 
                      'average_montly_hours', 'time_spend_company']
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Feature engineering: Create interaction between satisfaction_level and number_project
data['satisfaction_project_interaction'] = data['satisfaction_level'] * data['number_project']

# Split data into features (X) and target (y)
X = data.drop('left', axis=1)
y = data['left']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the training and test data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression Model
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_scaled, y_train)

# XGBoost Classifier
xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

# Flask route to render a simple form for user input
@app.route('/')
def home():
    return render_template('index.html')

# Flask route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from form
    satisfaction_level = float(request.form['satisfaction_level'])
    last_evaluation = float(request.form['last_evaluation'])
    number_project = int(request.form['number_project'])
    average_montly_hours = int(request.form['average_montly_hours'])
    time_spend_company = int(request.form['time_spend_company'])
    work_accident = int(request.form['work_accident'])
    promotion_last_5years = int(request.form['promotion_last_5years'])
    department = int(request.form['department'])
    salary = int(request.form['salary'])
    
    # Calculate the interaction term
    satisfaction_project_interaction = satisfaction_level * number_project
    
    # Make the prediction using XGBoost
    prediction = xgb_model.predict([[satisfaction_level, last_evaluation, number_project, 
                                     average_montly_hours, time_spend_company, work_accident, 
                                     promotion_last_5years, department, salary, 
                                     satisfaction_project_interaction]])
    
    # Interpret the prediction result
    if prediction[0] == 1:
        result = "Employee left."
    else:
        result = "Employee did not leave."
    
    # Now plot the boxplot
    plt.figure(figsize=(10, 6))
    
    # Box plot to show the distribution of satisfaction_project_interaction for all employees
    sns.boxplot(x='left', y='satisfaction_project_interaction', data=data)
    plt.title('Satisfaction * Project Interaction Comparison (Left vs Stayed)')
    plt.xlabel('Employee Left (0 = No, 1 = Yes)')
    plt.ylabel('Satisfaction * Project Interaction')
    
    # Mark the user's satisfaction_project_interaction value on the plot
    plt.scatter(y=satisfaction_project_interaction, x=0.5, color='red', s=100, label="Specific Employee Interaction", zorder=5)
    plt.legend()
    
    # Save the plot to a BytesIO object
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    # Pass both the prediction result and plot to the template
    return render_template('result.html', prediction_text=result, plot_url=plot_url)

if __name__ == "__main__":
    app.run(debug=True)