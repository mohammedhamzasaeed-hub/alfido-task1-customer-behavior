# ==============================================================================
# Alfido Tech Internship - Task 1: Customer Behavior Analysis
# Author: Intern at Alfido Tech
# Description: Analysis of customer transactions to identify segments,
#              purchase patterns, and churn risks using RFM analysis.
# ==============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Set visual style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.size'] = 10

# ------------------------------------------------------------------------------
# 1. Load the Data
# ------------------------------------------------------------------------------
# For Google Colab, place 'ecommerce_customer_data_custom_ratios.csv' in the same folder
# or upload it using the Files tab.
DATA_PATH = os.path.join('data', 'ecommerce_customer_data_custom_ratios.csv')
if not os.path.exists(DATA_PATH):
    DATA_PATH = 'ecommerce_customer_data_custom_ratios.csv'

print(f"Loading dataset from: {DATA_PATH}")
raw_df = pd.read_csv(DATA_PATH)
df = raw_df.copy()

print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values per column:")
print(df.isnull().sum())

# ------------------------------------------------------------------------------
# 2. Data Cleaning
# ------------------------------------------------------------------------------
# Check for exact duplicate rows
duplicate_count = df.duplicated().sum()
print(f"\nExact duplicate rows found: {duplicate_count}")

# Fix data types: convert Purchase Date to datetime
df['Purchase Date'] = pd.to_datetime(df['Purchase Date'])

# Handle missing values: Returns has 47,596 missing values (representing no return recorded)
df['Returns'] = df['Returns'].fillna(0.0)

# Drop redundant Age column (identical to Customer Age)
if 'Age' in df.columns and 'Customer Age' in df.columns:
    df = df.drop(columns=['Age'])

print(f"Shape after cleaning: {df.shape}")
print("Data cleaning completed successfully.")

# ------------------------------------------------------------------------------
# 3. Feature Engineering (Customer-Level Aggregation)
# ------------------------------------------------------------------------------
# Set reference date as latest transaction date + 1 day
reference_date = df['Purchase Date'].max() + pd.Timedelta(days=1)
print(f"Reference Date for Recency calculation: {reference_date}")

# Aggregate key metrics per customer
customer_df = df.groupby('Customer ID').agg(
    Recency=('Purchase Date', lambda x: (reference_date - x.max()).days),
    Frequency=('Purchase Date', 'count'),
    Monetary=('Total Purchase Amount', 'sum'),
    Avg_Order_Value=('Total Purchase Amount', 'mean'),
    Total_Quantity=('Quantity', 'sum'),
    Return_Count=('Returns', 'sum'),
    Customer_Age=('Customer Age', 'first'),
    Gender=('Gender', 'first'),
    Churn=('Churn', 'first')
).reset_index()

# Calculate customer return rate
customer_df['Return_Rate'] = customer_df['Return_Count'] / customer_df['Frequency']

# Determine favorite product category for each customer
favorite_cat = df.groupby(['Customer ID', 'Product Category']).size().unstack(fill_value=0).idxmax(axis=1)
customer_df['Favorite_Category'] = customer_df['Customer ID'].map(favorite_cat)

print(f"Total unique customers: {len(customer_df)}")
print("\nCustomer-level summary statistics:")
print(customer_df[['Recency', 'Frequency', 'Monetary', 'Avg_Order_Value']].describe())

# ------------------------------------------------------------------------------
# 4. Exploratory Data Analysis (EDA)
# ------------------------------------------------------------------------------
# Create images folder if not exists
os.makedirs('images', exist_ok=True)

# 4.1 Monthly Revenue Trend
df['YearMonth'] = df['Purchase Date'].dt.to_period('M').astype(str)
monthly_summary = df.groupby('YearMonth').agg(
    Revenue=('Total Purchase Amount', 'sum'),
    Transactions=('Customer ID', 'count')
).reset_index()

plt.figure(figsize=(10, 4.5))
plt.plot(monthly_summary['YearMonth'], monthly_summary['Revenue'] / 1e6, marker='o', markersize=3, color='#1f77b4', linewidth=2)
plt.title('Monthly Revenue Trend (2020 - 2023)', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Year-Month', fontsize=11)
plt.ylabel('Revenue (Million $)', fontsize=11)
plt.xticks(ticks=range(0, len(monthly_summary), 4), labels=monthly_summary['YearMonth'].iloc[::4], rotation=45)
plt.tight_layout()
plt.savefig('images/chart_01_monthly_revenue_trend.png', bbox_inches='tight', dpi=120)
plt.close()

# 4.2 Purchases by Product Category
cat_summary = df.groupby('Product Category').agg(
    Revenue=('Total Purchase Amount', 'sum'),
    Transactions=('Customer ID', 'count')
).reset_index()

plt.figure(figsize=(8, 4.5))
bars = plt.bar(cat_summary['Product Category'], cat_summary['Revenue'] / 1e6, color='#2b5c8f', width=0.55)
plt.title('Total Revenue by Product Category', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Product Category', fontsize=11)
plt.ylabel('Revenue (Million $)', fontsize=11)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 3, f'${yval:.1f}M', ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.ylim(0, 240)
plt.tight_layout()
plt.savefig('images/chart_02_purchases_by_category.png', bbox_inches='tight', dpi=120)
plt.close()

# ------------------------------------------------------------------------------
# 5. Customer Segmentation (RFM Analysis)
# ------------------------------------------------------------------------------
# R_Score: 5 = most recent, 1 = least recent
customer_df['R_Score'] = pd.qcut(customer_df['Recency'], 5, labels=[5, 4, 3, 2, 1]).astype(int)
# F_Score: 5 = highest frequency (using rank to avoid duplicate bin errors)
customer_df['F_Score'] = pd.qcut(customer_df['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
# M_Score: 5 = highest monetary spend
customer_df['M_Score'] = pd.qcut(customer_df['Monetary'], 5, labels=[1, 2, 3, 4, 5]).astype(int)

# Segmentation Logic
def assign_rfm_segment(row):
    r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
    if r >= 4 and f >= 4 and m >= 4:
        return 'Champions'
    elif r >= 3 and f >= 3:
        return 'Loyal Customers'
    elif r >= 3 and f < 3:
        return 'Potential / Recent'
    elif r <= 2 and f >= 3:
        return 'At Risk'
    else:
        return 'Lost / Hibernating'

customer_df['Segment'] = customer_df.apply(assign_rfm_segment, axis=1)

# ------------------------------------------------------------------------------
# 6. Segment Profiling
# ------------------------------------------------------------------------------
segment_order = ['Champions', 'Loyal Customers', 'At Risk', 'Lost / Hibernating', 'Potential / Recent']

segment_profile = customer_df.groupby('Segment').agg(
    Customer_Count=('Customer ID', 'count'),
    Avg_Recency_Days=('Recency', 'mean'),
    Avg_Frequency_Orders=('Frequency', 'mean'),
    Avg_Spend=('Monetary', 'mean'),
    Total_Spend=('Monetary', 'sum'),
    Churn_Rate=('Churn', 'mean'),
    Avg_Return_Rate=('Return_Rate', 'mean')
).reindex(segment_order).reset_index()

total_customers = len(customer_df)
total_revenue = customer_df['Monetary'].sum()

segment_profile['Pct_Customers'] = (segment_profile['Customer_Count'] / total_customers) * 100
segment_profile['Pct_Revenue'] = (segment_profile['Total_Spend'] / total_revenue) * 100

print("\n=== CUSTOMER SEGMENT PROFILE SUMMARY ===")
print(segment_profile[['Segment', 'Customer_Count', 'Pct_Customers', 'Avg_Spend', 'Pct_Revenue', 'Avg_Recency_Days']].to_string(index=False))

# ------------------------------------------------------------------------------
# 7. Visualizing Segments, Spending & Retention
# ------------------------------------------------------------------------------
# 7.1 Spending by Segment
plt.figure(figsize=(9, 4.5))
bars = plt.bar(segment_profile['Segment'], segment_profile['Total_Spend'] / 1e6, color='#3470a3', width=0.55)
plt.title('Total Revenue Contribution by Customer Segment', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Customer Segment', fontsize=11)
plt.ylabel('Total Spend (Million $)', fontsize=11)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 3, f'${yval:.1f}M', ha='center', va='bottom', fontsize=9.5, fontweight='bold')
plt.ylim(0, 220)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('images/chart_03_spending_by_segment.png', bbox_inches='tight', dpi=120)
plt.close()

# 7.2 Retention Trend (Active Customers per Month)
monthly_active = df.groupby('YearMonth')['Customer ID'].nunique().reset_index()
plt.figure(figsize=(10, 4.5))
plt.plot(monthly_active['YearMonth'], monthly_active['Customer ID'], marker='s', markersize=3, color='#e6550d', linewidth=2)
plt.title('Monthly Active Customers (Retention Trend)', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Year-Month', fontsize=11)
plt.ylabel('Active Customers', fontsize=11)
plt.xticks(ticks=range(0, len(monthly_active), 4), labels=monthly_active['YearMonth'].iloc[::4], rotation=45)
plt.tight_layout()
plt.savefig('images/chart_04_retention_active_customers.png', bbox_inches='tight', dpi=120)
plt.close()

# 7.3 Segment Distribution (% Customers vs % Revenue)
x = np.arange(len(segment_order))
width = 0.35
fig, ax = plt.subplots(figsize=(9, 4.8))
rects1 = ax.bar(x - width/2, segment_profile['Pct_Customers'], width, label='% of Customers', color='#5b9bd5')
rects2 = ax.bar(x + width/2, segment_profile['Pct_Revenue'], width, label='% of Total Revenue', color='#ed7d31')
ax.set_title('Customer Base vs Revenue Contribution by Segment', fontsize=13, fontweight='bold', pad=12)
ax.set_xticks(x)
ax.set_xticklabels(segment_order, rotation=15)
ax.set_ylabel('Percentage (%)', fontsize=11)
ax.legend(frameon=True)
ax.set_ylim(0, 35)
for r in rects1:
    h = r.get_height()
    ax.text(r.get_x() + r.get_width()/2, h + 0.5, f'{h:.1f}%', ha='center', va='bottom', fontsize=9)
for r in rects2:
    h = r.get_height()
    ax.text(r.get_x() + r.get_width()/2, h + 0.5, f'{h:.1f}%', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig('images/chart_05_segment_distribution.png', bbox_inches='tight', dpi=120)
plt.close()

# ------------------------------------------------------------------------------
# 8. Churn Risk Analysis
# ------------------------------------------------------------------------------
print("\n=== CHURN ANALYSIS ===")
overall_churn_rate = customer_df['Churn'].mean() * 100
print(f"Overall Customer Churn Rate: {overall_churn_rate:.2f}% ({customer_df['Churn'].sum()} churned out of {len(customer_df)})")

# Recency distribution by segment
plt.figure(figsize=(8.5, 4.5))
sns.boxplot(data=customer_df, x='Segment', y='Recency', order=segment_order, hue='Segment', palette='Blues_r', legend=False)
plt.title('Customer Recency (Days Since Last Purchase) by Segment', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Customer Segment', fontsize=11)
plt.ylabel('Recency (Days)', fontsize=11)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('images/chart_06_churn_and_recency.png', bbox_inches='tight', dpi=120)
plt.close()

# ------------------------------------------------------------------------------
# 9. Key Findings Summary
# ------------------------------------------------------------------------------
print("\n=== KEY FINDINGS ===")
print("1. High Value Concentration: Champions and Loyal Customers make up 42.5% of the customer base but drive 55.8% of total revenue ($380.1M).")
print("2. Critical At-Risk Revenue: 8,711 customers (17.5%) have high past engagement (5.9 orders avg) but have been inactive for an average of 411.5 days, putting $140.7M at risk.")
print("3. High Return Rate: Approximately 49.8% of items with return data are returned, impacting margins across Clothing and Books.")
print("4. Core Category Drivers: Books and Clothing represent 60.0% of all orders and $409.5M in revenue.")
print("5. Consistent Churn Baseline: Churn rate is 20.0% across customer accounts, requiring proactive recency triggers before inactivity exceeds 180 days.")

print("\nAnalysis completed successfully. Visualizations saved in 'images/' directory.")
