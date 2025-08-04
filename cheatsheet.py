                                #PANDAS: Essential Patterns
    # Load & Inspect
df = pd.read_csv("file.csv")
df.head()
df.info()
df.describe()

    # Filter & Select
df[df['col'] > 10]
df.loc[df['col'] == 'value', ['col1', 'col2']]

    # Group & Aggregate
df.groupby('category')['value'].mean()

    # Merge / Join
pd.merge(df1, df2, on='id', how='left')

    # Reshape
df.pivot(index='date', columns='category', values='value')

    # Handle Nulls
df.isnull().sum()
df.fillna(0)

    # Time Series
df['date'] = pd.to_datetime(df['date'])
df.set_index('date').resample('M').mean()


                            #MATPLOTLIB & SEABORN: Visual Essentials
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

    # Basic Plots
sns.histplot(df['value'])
sns.boxplot(x='category', y='value', data=df)
sns.scatterplot(x='x', y='y', hue='label', data=df)

    # Correlation & Trends
sns.heatmap(df.corr(), annot=True)
sns.lineplot(x='date', y='sales', data=df)

    # Customize
plt.title("Title")
plt.tight_layout()
plt.show()


    #Practice Tips & Workflow
# Use .head(), .describe(), .groupby()

