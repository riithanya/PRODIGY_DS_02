import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


titanic_df = pd.read_csv('train.csv')


sns.countplot(x='Sex', data=titanic_df)
plt.title('Comparison of Men and Women')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.show()


sns.countplot(x='Survived', hue='Sex', data=titanic_df)
plt.title('Survival Comparison by Gender')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.legend(title='Sex', loc='upper right')
plt.show()
