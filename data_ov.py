import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("data/blood.csv")  # Adjust path if needed
df.head()
df.info()
df.describe()
df.isnull().sum()

# Check for duplicates
print("Duplicate rows:", df.duplicated().sum())

# Drop duplicates if any
df = df.drop_duplicates()

df.drop(columns=['Recency'], inplace=True, errors='ignore')  # Example

# Check target distribution
df['Frequency'].value_counts().plot(
    kind='bar', title='Target Distribution', figsize=(10, 6), rot=0)

# Impute numeric columns
df.fillna(df.median(numeric_only=True), inplace=True)

# Impute categorical with mode (most common value)
for col in df.select_dtypes(include='object').columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

df.to_csv("data/cleaned_blood_data.csv", index=False)
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()
