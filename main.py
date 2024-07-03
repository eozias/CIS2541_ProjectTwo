# Import Statements
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import neighbors, metrics
from sklearn import naive_bayes
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn import svm

def projectTwo():
    # step 1: load the Breast Cancer Wisconsin dataset
    breast_cancer = datasets.load_breast_cancer()
    X = breast_cancer.data
    y = breast_cancer.target

    # step 2: convert the data to a DataFrame
    dataframe = pd.DataFrame(X, columns=breast_cancer.feature_names)
    dataframe["target"] = y

    # step 3: do some EDA (exploratory data analysis)
    print("Displaying the first few rows in the dataset...")
    print(dataframe.head())
    print("Displaying the summary statistics...")
    print(dataframe.describe())
    print("Displaying the number of missing values...")
    print(dataframe.isnull().sum())

    # step 4: visualize the distribution of the target variable
    plt.title("Distribution of Malignant and Benign Tumors")
    sns.countplot(x=dataframe["target"], palette="coolwarm")
    plt.xlabel("Target")
    plt.ylabel("Count")
    plt.show()

    corr_matrix = dataframe.corr()
    plt.title("Correlation Matrix")
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm")
    plt.show()

    # step 5: create a standard scaler to standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # step 6: split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Training Features Shape:", X_train.shape)
    print("Testing Features Shape:", X_test.shape)

    # step 7: build and evaluate the SVM model
    svm_linear = svm.SVC(kernel="linear", C=1)
    svm_linear.fit(X_train, y_train)
    linear_pred = svm_linear.predict(X_test)
    print("Linear Kernel Accuracy:", metrics.accuracy_score(y_test, linear_pred))
    print("Linear Kernel F1 Score:", metrics.f1_score(y_test, linear_pred))

    svm_rbf = svm.SVC(kernel="rbf", C=1)
    svm_rbf.fit(X_train, y_train)
    rbf_pred = svm_rbf.predict(X_test)
    print("RBF Kernel Accuracy:", metrics.accuracy_score(y_test, rbf_pred))
    print("RBF Kernel F1 Score:", metrics.f1_score(y_test, rbf_pred))

    svm_poly = svm.SVC(kernel="poly", C=1)
    svm_poly.fit(X_train, y_train)
    poly_pred = svm_poly.predict(X_test)
    print("Polynomial Kernel Accuracy:", metrics.accuracy_score(y_test, poly_pred))
    print("Polynomial Kernel F1 Score:", metrics.f1_score(y_test, poly_pred))

    print("Classification Report for Linear Kernel:")
    print(metrics.classification_report(y_test, linear_pred))

    linear_cm = metrics.confusion_matrix(y_test, linear_pred)
    print("Confusion Matrix for Linear Kernel:")
    print(linear_cm)

    print("Classification Report for RBF Kernel:")
    print(metrics.classification_report(y_test, rbf_pred))

    rbf_cm = metrics.confusion_matrix(y_test, rbf_pred)
    print("Confusion Matrix for RBF Kernel:")
    print(rbf_cm)

    print("Classification Report for Polynomial Kernel:")
    print(metrics.classification_report(y_test, poly_pred))

    poly_cm = metrics.confusion_matrix(y_test, poly_pred)
    print("Confusion Matrix for Polynomial Kernel:")
    print(poly_cm)

    # step 8: create a confusion matrix heatmap
    sns.heatmap(linear_cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix for Linear Kernel")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    sns.heatmap(rbf_cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix for RBF Kernel")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    sns.heatmap(poly_cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix for Polynomial Kernel")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

if __name__ == "__main__":
    projectTwo()