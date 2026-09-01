import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
## TASK 1: LOAD CLEAN CHURN DATASET AND CHECK FOR MISSING VALUES AND DATAT TYPES
df = pd.read_csv("D:\\machine learning\\clean_churn.csv")
print(df.shape)
print(df.info())
print(df.head())
print(df["Churn"].value_counts())
## TASK : 2
## STATE THE PROBLEM IN YOUR OWN WORDS
##*ANS*.. THE PROBLM IS CUSTOMER CHURN AND TO PREDICT WETHER THE CUSTOMER WILL CHURN OR NOT USING PYTHON AND MACHINE LEARNING ALGORITHMS.
##  BEFORE WE VISULAIZE THE DATASET USING DEFFRENT PLOTS AND CLEANED IT USING DEFFERENT METHODS NOW WE WILL USE DECISION TREE ALGORITHM TO FURTHUR EXPLORE IT


###  TASK 3:
# DEFINE CHURN AS THE TARGET VARIABLE AND CONVERT IT INTO NUMERICAL FORMAT
# Define y
y = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

# Define X
X = df.drop(columns=["Churn", "customerID"])

# Find categorical columns
categorical_cols = X.select_dtypes(include=["object"]).columns

# One-hot encode them  because Because categories such as different
#  contract types don't have a natural numerical order. We don't want the model to incorrectly interpret:
X = pd.get_dummies(X, columns=categorical_cols, dtype=int)

# Check missing values
print(X.isnull().sum())

# Check data types
print(X.dtypes)

# Confirm all columns are numeric
print(X.select_dtypes(exclude="number").columns)
X_train, X_test , y_train , y_test = train_test_split(
X, 
y,
test_size=0.2,
random_state= 42,
stratify=y
)
##stratify=y is a sensible choice because the Churn 
# target is imbalanced. It ensures that the training and test sets maintain approximately the same proportion of Churn and non-Churn cases as the original dataset. This makes 
# the test set more representative of the data and gives a fairer evaluation of the model.
print("X train shape:", X_train.shape)
print("X test shape:", X_test.shape)
print("y train shape:", y_train.shape)
print("y test shape:", y_test.shape)
##TASK 8
model = DecisionTreeClassifier(random_state = 42)
model.fit(X_train , y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
#TASK 9
print("tree depth: ", model.get_depth())

print("train accuracy: ",model.score(X_train, y_train))
print("test accuracy: ",model.score(X_test , y_test))
## ans: The Decision Tree grew to a depth of 23. 
# This is not necessarily surprising because I did not specify a maximum depth,
#  so the Decision Tree was allowed to grow according to its default settings.
#  A depth of 23 indicates that the tree has many levels of decision rules 
# and may be fitting the training data very closely. This could lead to overfitting, 
# which we can investigate by comparing the training and test performance.
#task 10
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
#task 11
# Display confusion matrix
 

cm = confusion_matrix(y_test, y_test_pred)

print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Churn", "Churn"]
)

disp.plot()
plt.title("Confusion Matrix - Decision Tree")
plt.show()
#task 12
#answer:Because the Churn target is imbalanced, 
# accuracy alone should not be relied upon. Recall and 
# F1-score are more informative metrics because they focus on the
#  minority churn class and show how well the model identifies actual churners. 
# Recall is particularly important because it measures the proportion of 
# actual churners that the model correctly catches. 
# F1-score is also useful because it balances precision and recall.
from sklearn.metrics import accuracy_score

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
#task 13
print("Training accuracy:", train_accuracy)
print("Test accuracy:", test_accuracy)
print("Gap:", train_accuracy - test_accuracy)
## task 14
## answer: The baseline Decision Tree has a training accuracy 
# of .90 and a test accuracy of .70, giving an accuracy gap of 0.20. 
# Therefore, there is no large gap between training and test accuracy. 
# This suggests that the model is performing equally well on both datasets. 
# However, because the tree has a depth of 23 and achieves perfect accuracy, 
# it is still important to investigate 
# its performance at different depths and check for possible data leakage.
##task 15
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

depths = [2, 3, 4, 5, 6, 8, 10, None]

results = []

for depth in depths:
    tree = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    tree.fit(X_train, y_train)

    train_pred = tree.predict(X_train)
    test_pred = tree.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    results.append({
        "max_depth": "Unrestricted" if depth is None else depth,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc
    })

df_results = pd.DataFrame(results)

print(df_results)
#task 17
 
chosen_depth = 5

chosen_tree = DecisionTreeClassifier(
    max_depth=chosen_depth,
    random_state=42
)

chosen_tree.fit(X_train, y_train)

y_train_chosen = chosen_tree.predict(X_train)
y_test_chosen = chosen_tree.predict(X_test)

accuracy = accuracy_score(y_test, y_test_chosen)
precision = precision_score(y_test, y_test_chosen)
recall = recall_score(y_test, y_test_chosen)
f1 = f1_score(y_test, y_test_chosen)

print("Chosen depth:", chosen_depth)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
#task 18
entropy_tree = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=chosen_depth,
    random_state=42
)

entropy_tree.fit(X_train, y_train)

y_test_entropy = entropy_tree.predict(X_test)

entropy_accuracy = accuracy_score(y_test, y_test_entropy)
entropy_precision = precision_score(y_test, y_test_entropy)
entropy_recall = recall_score(y_test, y_test_entropy)
entropy_f1 = f1_score(y_test, y_test_entropy)

print("Entropy Tree")
print("Accuracy:", entropy_accuracy)
print("Precision:", entropy_precision)
print("Recall:", entropy_recall)
print("F1-score:", entropy_f1)

##task 19
comparison = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
    "Baseline (Gini)": [
        accuracy_score(y_test, y_test_pred),
        precision_score(y_test, y_test_pred),
        recall_score(y_test, y_test_pred),
        f1_score(y_test, y_test_pred)
    ],
    "Chosen Depth": [
        accuracy,
        precision,
        recall,
        f1
    ],
    "Entropy": [
        entropy_accuracy,
        entropy_precision,
        entropy_recall,
        entropy_f1
    ]
})

print(comparison)
#task 20
import pandas as pd
import matplotlib.pyplot as plt

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": chosen_tree.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print(feature_importance.head(5))

top_features = feature_importance.head(10)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Top Feature Importances")

plt.gca().invert_yaxis()

plt.show()

#task 21
#Conclusion: The final model selected was a Decision Tree with a maximum depth
 #chosen based on the training and test performance comparison. 
 #The model achieved very strong performance on the test set, 
 #with high accuracy, precision, recall, and F1-score. The unrestricted 
 #baseline tree reached a depth of 23, 
 # showing that a tree can become quite complex when no maximum depth is specified.
 #  A main limitation is that Decision Trees can be sensitive to the training data 
 # and can become overly complex, even when test performance appears strong.
 #The unusually perfect test performance should also be checked for possible data leakage.

 #task 22
 ## If I had more time, I would try other algorithms 
 # such as Random Forest or Gradient Boosting and compare 
 # their performance with the Decision Tree. I would also 
 # investigate feature engineering and check for data leakage. 
 # Because the Churn classes are imbalanced, I would consider techniques 
 # such as class weighting or resampling and focus particularly on recall 
 # and F1-score rather than accuracy alone. Finally, I would use cross-validation 
 # and hyperparameter
 #  tuning to obtain a more reliable estimate of model performance.