import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


titanic_df = pd.read_csv('train.csv')


print(titanic_df.head())


print(titanic_df.describe())


print(titanic_df.info())


print(titanic_df.isnull().sum())


sns.countplot(x='Survived', data=titanic_df)
plt.title('Survival Count')
plt.show()


sns.countplot(x='Survived', hue='Sex', data=titanic_df)
plt.title('Survival by Gender')
plt.show()


sns.countplot(x='Survived', hue='Pclass', data=titanic_df)
plt.title('Survival by Passenger Class')
plt.show()


plt.hist(titanic_df['Age'].dropna(), bins=20)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


plt.hist(titanic_df['Fare'], bins=20)
plt.title('Fare Distribution')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.show()


titanic_df['FamilySize'] = titanic_df['SibSp'] + titanic_df['Parch'] + 1


sns.countplot(x='Survived', hue='FamilySize', data=titanic_df)
plt.title('Survival by Family Size')
plt.show()


correlation_matrix = titanic_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
