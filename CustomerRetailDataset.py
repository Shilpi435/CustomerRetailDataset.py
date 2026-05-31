# Customer Retail Dataset Analysis
# Logistic Regression vs Decision Tree vs KNN
# =========================================================

# =========================================================
# Import Libraries
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# =========================================================
# Step 1: Create Customer Retail Dataset
# =========================================================

data = {

    'Quantity': [

        10, 15, 7, 20, 30,
        5, 12, 18, 25, 8,
        14, 28, 16, 22, 6

    ],

    'UnitPrice': [

        100, 150, 80, 200, 300,
        60, 120, 180, 250, 90,
        140, 280, 160, 220, 70

    ],

    'Country': [

        'India', 'USA', 'India', 'Canada', 'USA',
        'India', 'Canada', 'USA', 'India', 'Canada',
        'USA', 'India', 'Canada', 'USA', 'India'

    ],

    'Purchase': [

        'Yes', 'Yes', 'No', 'Yes', 'Yes',
        'No', 'Yes', 'Yes', 'Yes', 'No',
        'Yes', 'Yes', 'Yes', 'Yes', 'No'

    ]

}

# Create DataFrame
df = pd.DataFrame(data)

print("\nDataset:\n")

print(df)

# =========================================================
# Step 2: Handle Missing Values
# =========================================================

df.fillna(0, inplace=True)

# =========================================================
# Step 3: Encode Categorical Columns
# =========================================================

encoder = LabelEncoder()

df['Country'] = encoder.fit_transform(
    df['Country']
)

df['Purchase'] = encoder.fit_transform(
    df['Purchase']
)

# =========================================================
# Step 4: Define Features and Target
# =========================================================

X = df[['Quantity', 'UnitPrice', 'Country']]

y = df['Purchase']

# =========================================================
# Step 5: Split Dataset
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.3,

    random_state=42

)

# =========================================================
# Step 6: Logistic Regression Model
# =========================================================

logistic_model = LogisticRegression()

logistic_model.fit(X_train, y_train)

logistic_prediction = logistic_model.predict(X_test)

logistic_accuracy = accuracy_score(

    y_test,
    logistic_prediction

)

print("\nLogistic Regression Accuracy:")

print(logistic_accuracy)

# =========================================================
# Step 7: Decision Tree Model
# =========================================================

decision_model = DecisionTreeClassifier()

decision_model.fit(X_train, y_train)

decision_prediction = decision_model.predict(X_test)

decision_accuracy = accuracy_score(

    y_test,
    decision_prediction

)

print("\nDecision Tree Accuracy:")

print(decision_accuracy)

# =========================================================
# Step 8: KNN Model
# =========================================================

knn_model = KNeighborsClassifier(n_neighbors=3)

knn_model.fit(X_train, y_train)

knn_prediction = knn_model.predict(X_test)

knn_accuracy = accuracy_score(

    y_test,
    knn_prediction

)

print("\nKNN Accuracy:")

print(knn_accuracy)

# =========================================================
# Step 9: Confusion Matrix
# =========================================================

print("\nConfusion Matrix for Logistic Regression:\n")

print(

    confusion_matrix(

        y_test,

        logistic_prediction

    )

)

# =========================================================
# Step 10: Customer Distribution Graph
# =========================================================

plt.bar(

    ['India', 'USA', 'Canada'],

    [

        list(data['Country']).count('India'),

        list(data['Country']).count('USA'),

        list(data['Country']).count('Canada')

    ]

)

plt.title(
    'Customer Distribution by Country'
)

plt.xlabel('Country')

plt.ylabel('Number of Customers')

plt.show()

# =========================================================
# Step 11: Model Accuracy Comparison Graph
# =========================================================

models = [

    'Logistic Regression',

    'Decision Tree',

    'KNN'

]

accuracies = [

    logistic_accuracy,

    decision_accuracy,

    knn_accuracy

]

# Different colors for each model
colors = [

    'blue',

    'green',

    'red'

]

bars = plt.bar(

    models,

    accuracies,

    color=colors

)

# Add accuracy values on top
for bar in bars:

    height = bar.get_height()

    plt.text(

        bar.get_x() + bar.get_width()/2,

        height,

        f'{height:.2f}',

        ha='center',

        va='bottom',

        fontsize=10

    )

plt.title(
    'Model Accuracy Comparison'
)

plt.xlabel(
    'Machine Learning Models'
)

plt.ylabel(
    'Accuracy'
)

plt.ylim(0, 1)

plt.show()

# =========================================================
# Step 12: Quantity vs UnitPrice Scatter Plot
# =========================================================

plt.scatter(

    df['Quantity'],

    df['UnitPrice']

)

plt.title(
    'Quantity vs UnitPrice'
)

plt.xlabel('Quantity')

plt.ylabel('UnitPrice')

plt.show()

# =========================================================
# Step 13: Predict New Customer Purchase
# =========================================================

new_customer = [[20, 210, 2]]

prediction = logistic_model.predict(
    new_customer
)

if prediction[0] == 1:

    print("\nCustomer Will Purchase")

else:

    print("\nCustomer Will Not Purchase")