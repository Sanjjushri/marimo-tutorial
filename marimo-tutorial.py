import marimo

__generated_with = "0.6.19"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell
def __(mo):
    mo.md("# Welcome to Marimo Tutorial")
    return


@app.cell
def __():
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    return (
        LogisticRegression,
        accuracy_score,
        classification_report,
        confusion_matrix,
        load_iris,
        pd,
        plt,
        sns,
        train_test_split,
    )


@app.cell
def __(load_iris, pd):
    # Load the Iris dataset
    iris = load_iris()
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df['species'] = iris.target
    return iris, iris_df


@app.cell
def __(iris_df):
    iris_df.head()
    return


@app.cell
def __(iris_df, plt, sns):
    # Visualize the dataset
    sns.pairplot(iris_df, hue='species', markers=["o", "s", "D"])
    plt.suptitle('Pairplot of Iris Features', y=1.02)
    plt.show()
    return


@app.cell
def __(iris_df, plt, sns):
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(iris_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap of Iris Features')
    plt.show()
    return


@app.cell
def __(iris, iris_df, train_test_split):
    # Prepare data for prediction
    X = iris_df[iris.feature_names]
    y = iris_df['species']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X, X_test, X_train, y, y_test, y_train


@app.cell
def __(LogisticRegression, X_train, y_train):
    # Train a logistic regression model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    return model,


@app.cell
def __(X_test, model):
    # Make predictions
    y_pred = model.predict(X_test)
    return y_pred,


@app.cell
def __(accuracy_score, classification_report, y_pred, y_test):
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')
    print('Classification Report:')
    print(classification_report(y_test, y_pred))
    return accuracy,


@app.cell
def __(confusion_matrix, iris, plt, sns, y_pred, y_test):
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    return conf_matrix,


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
