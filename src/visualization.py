import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


def visualize_data(df, save_dir="plots", save=True):
    if save and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    df.hist(figsize=(12, 10), bins=20)
    plt.suptitle("Feature Distributions")
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(save_dir, "histograms.png"))
    plt.show()

    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df[[
        "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"
    ]])
    plt.title("Boxplot for key features")
    if save:
        plt.savefig(os.path.join(save_dir, "boxplots.png"))
    plt.show()

    plt.figure(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation matrix")
    if save:
        plt.savefig(os.path.join(save_dir, "correlation_matrix.png"))
    plt.show()

    pairplot_cols = ["Glucose", "BMI", "Age", "Insulin", "BloodPressure"]
    sns.pairplot(df, hue="Outcome", vars=pairplot_cols)
    plt.suptitle("Pairplot of key features", y=1.02)
    if save:
        plt.savefig(os.path.join(save_dir, "pairplot.png"))
    plt.show()

    plt.figure(figsize=(6, 4))
    sns.countplot(x="Outcome", data=df)
    plt.title("Class distribution")
    if save:
        plt.savefig(os.path.join(save_dir, "class_distribution.png"))
    plt.show()
