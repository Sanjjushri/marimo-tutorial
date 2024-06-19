import marimo

__generated_with = "0.6.19"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell
def __(mo):
    mo.md("# Welcome to AI with Sanjju :)")
    return


@app.cell
def __(mo):
    slider = mo.ui.slider(1, 22)
    return slider,


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
def __(iris_df, plt, sns):
    # Visualize the dataset
    sns.pairplot(iris_df, hue='species', markers=["o", "s", "D"])
    plt.suptitle('Pairplot of Iris Features', y=1.02)
    plt.show()
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
