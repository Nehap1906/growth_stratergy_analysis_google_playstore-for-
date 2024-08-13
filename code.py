import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Load datasets/apps.csv into a DataFrame
apps_with_duplicates = pd.read_csv('datasets/apps.csv')

# Drop all duplicate rows from apps_with_duplicates
apps = apps_with_duplicates.drop_duplicates()

# Print the total number of apps
print('Total number of apps in the dataset =', len(apps))

# Display a random sample of 5 rows from apps
print(apps.sample(5))

# List of characters to remove
chars_to_remove = ['+', ',', '$']

# List of column names to clean
cols_to_clean = ['Installs', 'Price']

# Clean the specified columns
for col in cols_to_clean:
    for char in chars_to_remove:
        apps[col] = apps[col].apply(lambda x: x.replace(char, ''))

# Convert columns to float data type
apps['Installs'] = apps['Installs'].astype(float)
apps['Price'] = apps['Price'].astype(float)

# Print a summary of the apps dataframe
print(apps.info())

# Print the dtypes of the apps dataframe
print(apps.dtypes)

# Plot number of apps in each category
num_categories = len(apps['Category'].unique())
print('Number of categories =', num_categories)
num_apps_in_category = apps['Category'].value_counts()
sorted_num_apps_in_category = num_apps_in_category.sort_values(ascending=False)

plt.figure(figsize=(12, 6))
plt.bar(sorted_num_apps_in_category.index, sorted_num_apps_in_category.values)
plt.xticks(rotation=90)
plt.xlabel('Category')
plt.ylabel('Number of Apps')
plt.title('Number of Apps in Each Category')
plt.show()

# Plot distribution of app ratings
avg_app_rating = apps['Rating'].mean()
print('Average app rating =', avg_app_rating)

plt.figure(figsize=(12, 6))
plt.hist(apps['Rating'].dropna(), bins=30, edgecolor='k', alpha=0.7)
plt.axvline(avg_app_rating, color='r', linestyle='dashed', linewidth=1)
plt.xlabel('Rating')
plt.ylabel('Number of Apps')
plt.title('Distribution of App Ratings')
plt.show()

# Plot size vs. rating for categories with at least 250 apps
apps_with_size_and_rating_present = apps.dropna(subset=['Rating', 'Size'])
large_categories = apps_with_size_and_rating_present.groupby('Category').filter(lambda x: len(x) >= 250)

plt.figure(figsize=(12, 6))
sns.jointplot(x=large_categories['Size'], y=large_categories['Rating'], kind='scatter')
plt.suptitle('Size vs. Rating for Large Categories', y=1.02)
plt.show()

# Plot price vs. rating for paid apps
paid_apps = apps_with_size_and_rating_present[apps_with_size_and_rating_present['Type'] == 'Paid']

plt.figure(figsize=(12, 6))
sns.jointplot(x=paid_apps['Price'], y=paid_apps['Rating'], kind='scatter')
plt.suptitle('Price vs. Rating for Paid Apps', y=1.02)
plt.show()

# Plot price trend across selected categories
fig, ax = plt.subplots(figsize=(15, 8))
popular_app_cats = apps[apps['Category'].isin(['GAME', 'FAMILY', 'PHOTOGRAPHY', 'MEDICAL', 'TOOLS', 'FINANCE', 'LIFESTYLE', 'BUSINESS'])]
sns.stripplot(x=popular_app_cats['Price'], y=popular_app_cats['Category'], jitter=True, linewidth=1)
ax.set_title('App Pricing Trend Across Categories')
plt.xlabel('Price')
plt.ylabel('Category')
plt.show()

# Print apps with Price > $200
apps_above_200 = apps[apps['Price'] > 200]
print(apps_above_200[['Category', 'App', 'Price']])

# Plot price trend across selected categories after filtering for apps priced below $100
apps_under_100 = popular_app_cats[popular_app_cats['Price'] < 100]

fig, ax = plt.subplots(figsize=(15, 8))
sns.stripplot(x=apps_under_100['Price'], y=apps_under_100['Category'], jitter=True, linewidth=1)
ax.set_title('App Pricing Trend Across Categories After Filtering for Apps Priced Below $100')
plt.xlabel('Price')
plt.ylabel('Category')
plt.show()

# Plot number of installs for paid vs. free apps
plt.figure(figsize=(12, 6))
sns.boxplot(x=apps['Type'], y=np.log1p(apps['Installs']))
plt.xlabel('Type')
plt.ylabel('Log of Number of Installs')
plt.title('Number of Installs of Paid Apps vs. Free Apps')
plt.show()

# Load user_reviews.csv
reviews_df = pd.read_csv('datasets/user_reviews.csv')

# Join the two dataframes
merged_df = apps.merge(reviews_df, on='App')

# Drop NA values from Sentiment and Review columns
merged_df = merged_df.dropna(subset=['Sentiment', 'Review'])

# Plot sentiment polarity distribution for paid vs. free apps
plt.figure(figsize=(11, 8))
sns.boxplot(x=merged_df['Type'], y=merged_df['Sentiment_Polarity'])
plt.xlabel('Type')
plt.ylabel('Sentiment Polarity')
plt.title('Sentiment Polarity Distribution')
plt.show()
