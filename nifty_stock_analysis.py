#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Note: ensure you have all necessary libraries installed before proceeding, for pygwalker: pip install pygwalker

# Set default directory
import os
os.chdir(r'C:\Users\MSi\Desktop\Python Data\nifty_stock_analysis')

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pygwalker as pyg  # Assuming pygwalker is installed
from scipy.stats import linregress  # For linear regression to compute beta
from sklearn.linear_model import LinearRegression  # For ML beta prediction
from sklearn.preprocessing import StandardScaler
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# Set seaborn style for better visuals
sns.set(style="whitegrid")

# Step 1: Load and Clean Data
try:
    # Load NIFTY 50 data
    nifty_df = pd.read_csv('NIFTY_50_Historical_Perf_Data.csv')
    
    # Strip whitespace from column names
    nifty_df.columns = [col.strip() for col in nifty_df.columns]
    
    # Clean NIFTY data: Remove quotes, commas, and whitespace, convert to float
    for col in ['Open', 'Close']:
        if nifty_df[col].dtype == 'object':
            nifty_df[col] = nifty_df[col].str.strip().str.replace('"', '').str.replace(',', '').astype(float)
        else:
            nifty_df[col] = nifty_df[col].astype(float)
    
    nifty_df['Date'] = pd.to_datetime(nifty_df['Date'], format='%d-%b-%y')  # Convert date
    
    # Load Stocks data
    stocks_df = pd.read_csv('Stocks_Historical_Pricing_Data.csv')
    
    # Strip whitespace from column names
    stocks_df.columns = [col.strip() for col in stocks_df.columns]
    
    # Clean Stocks data: Remove quotes, commas, and whitespace, convert to float
    for col in ['Price (REDY)', 'Price (HALC)', 'Price (TAMO)', 'Price (VARB)', 'Price (KTKM)']:
        if stocks_df[col].dtype == 'object':
            stocks_df[col] = stocks_df[col].str.strip().str.replace('"', '').str.replace(',', '').astype(float)
        else:
            stocks_df[col] = stocks_df[col].astype(float)
    
    stocks_df['Date'] = pd.to_datetime(stocks_df['Date'], format='%d-%m-%Y')  # Convert date

    
#To ensure dataset is picked from correct directory set by you previously
except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure CSV files are in the directory: C:\\Users\\MSi\\Desktop\\Python Data")
    raise
except Exception as e:
    print(f"Error loading or cleaning data: {e}")
    raise

# Step 2: Merge Datasets
try:
    merged_df = pd.merge(nifty_df[['Date', 'Close']], stocks_df, on='Date', how='inner')
    merged_df = merged_df.rename(columns={'Close': 'NIFTY_Close'})  # Rename for clarity
    merged_df = merged_df.sort_values('Date')
    
    # Check for missing values
    if merged_df.isnull().any().any():
        print("Warning: Missing values detected. Filling with forward fill.")
        merged_df = merged_df.fillna(method='ffill')
    
    print("Merged Data Preview:")
    print(merged_df.head())
    
except Exception as e:
    print(f"Error merging datasets: {e}")
    raise

# Step 3: Compute Daily Returns
try:
    returns_df = merged_df.copy()
    returns_df['NIFTY_Return'] = returns_df['NIFTY_Close'].pct_change()
    returns_df['REDY_Return'] = returns_df['Price (REDY)'].pct_change()
    returns_df['HALC_Return'] = returns_df['Price (HALC)'].pct_change()
    returns_df['TAMO_Return'] = returns_df['Price (TAMO)'].pct_change()
    returns_df['VARB_Return'] = returns_df['Price (VARB)'].pct_change()
    returns_df['KTKM_Return'] = returns_df['Price (KTKM)'].pct_change()
    
    # Drop NaN rows
    returns_df = returns_df.dropna()
    
except Exception as e:
    print(f"Error computing returns: {e}")
    raise

# Step 4: Compute Correlations
try:
    corr_matrix = returns_df[['NIFTY_Return', 'REDY_Return', 'HALC_Return', 'TAMO_Return', 'VARB_Return', 'KTKM_Return']].corr()
    print("\nCorrelation Matrix:")
    print(corr_matrix.round(3))
    
except Exception as e:
    print(f"Error computing correlations: {e}")
    raise

# Step 5: Compute Beta
try:
    betas = {}
    for stock in ['REDY', 'HALC', 'TAMO', 'VARB', 'KTKM']:
        slope, intercept, r_value, p_value, std_err = linregress(returns_df['NIFTY_Return'], returns_df[f'{stock}_Return'])
        betas[stock] = {'Beta': slope, 'R-squared': r_value**2, 'p-value': p_value}
    
    betas_df = pd.DataFrame(betas).T
    print("\nBeta (Magnitude of Impact) for Each Stock:")
    print(betas_df.round(3))
    
except Exception as e:
    print(f"Error computing betas: {e}")
    raise

# Step 6: Inferences from Results
print("\nInferences from Correlation and Beta Analysis:")
print("-" * 50)
print("1. **Correlation Analysis**:")
print("   - The correlation matrix shows how closely each stock's daily returns move with the NIFTY 50 index.")
for stock in ['REDY', 'HALC', 'TAMO', 'VARB', 'KTKM']:
    corr = corr_matrix.loc['NIFTY_Return', f'{stock}_Return']
    if corr > 0.7:
        print(f"   - {stock}: Strong positive correlation ({corr:.3f}), indicating high sensitivity to NIFTY movements.")
    elif corr > 0.4:
        print(f"   - {stock}: Moderate positive correlation ({corr:.3f}), suggesting some alignment with NIFTY trends.")
    else:
        print(f"   - {stock}: Low correlation ({corr:.3f}), implying limited influence from NIFTY movements.")

print("\n2. **Beta Analysis**:")
print("   - Beta measures the magnitude of a stock's return sensitivity to NIFTY returns.")
for stock in ['REDY', 'HALC', 'TAMO', 'VARB', 'KTKM']:
    beta = betas_df.loc[stock, 'Beta']
    r_squared = betas_df.loc[stock, 'R-squared']
    if beta > 1:
        print(f"   - {stock}: Beta ({beta:.3f}) > 1, indicating higher volatility than NIFTY. A 1% change in NIFTY leads to a {beta:.3f}% change in {stock} returns (R² = {r_squared:.3f}).")
    elif beta > 0:
        print(f"   - {stock}: Beta ({beta:.3f}) < 1, indicating lower volatility than NIFTY. A 1% change in NIFTY leads to a {beta:.3f}% change in {stock} returns (R² = {r_squared:.3f}).")
    else:
        print(f"   - {stock}: Negative beta ({beta:.3f}), suggesting inverse movement to NIFTY (R² = {r_squared:.3f}).")

print("\n3. **Investment Implications**:")
print("   - Stocks with high beta (e.g., >1) are riskier but may offer higher returns during NIFTY uptrends.")
print("   - Stocks with low beta (e.g., <1) are more stable, suitable for risk-averse investors.")
print("   - Low R-squared values indicate that factors other than NIFTY influence stock returns, necessitating further analysis.")

# Step 7: Machine Learning for Beta Prediction
try:
    # Compute rolling betas over a 60-day window
    rolling_window = 60
    rolling_betas = pd.DataFrame(index=returns_df.index)
    
    for stock in ['REDY', 'HALC', 'TAMO', 'VARB', 'KTKM']:
        rolling_betas[stock] = returns_df.apply(
            lambda x: linregress(returns_df['NIFTY_Return'].iloc[max(0, x.name-rolling_window):x.name+1],
                                 returns_df[f'{stock}_Return'].iloc[max(0, x.name-rolling_window):x.name+1])[0]
            if x.name >= rolling_window else np.nan, axis=1)
    
    rolling_betas['Date'] = returns_df['Date']
    rolling_betas = rolling_betas.dropna()
    
    # Prepare features for ML
    features_df = rolling_betas.copy()
    features_df['NIFTY_Mean_Return'] = returns_df['NIFTY_Return'].rolling(window=rolling_window).mean()
    features_df['NIFTY_Volatility'] = returns_df['NIFTY_Return'].rolling(window=rolling_window).std()
    features_df['Days'] = (features_df['Date'] - features_df['Date'].min()).dt.days
    
    # Drop rows with NaN features
    features_df = features_df.dropna()
    
    # Train ML model for each stock
    future_betas = {}
    scaler = StandardScaler()
    
    for stock in ['REDY', 'HALC', 'TAMO', 'VARB', 'KTKM']:
        X = features_df[['Days', 'NIFTY_Mean_Return', 'NIFTY_Volatility', stock]].values
        y = features_df[stock].values
        
        # Scale features
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        # Predict future betas for next 30 trading days
        future_dates = [features_df['Date'].max() + timedelta(days=i) for i in range(1, 31)]
        future_days = [(d - features_df['Date'].min()).days for d in future_dates]
        last_nifty_mean = features_df['NIFTY_Mean_Return'].iloc[-1]
        last_nifty_vol = features_df['NIFTY_Volatility'].iloc[-1]
        last_beta = features_df[stock].iloc[-1]
        
        future_X = np.array([[day, last_nifty_mean, last_nifty_vol, last_beta] for day in future_days])
        future_X_scaled = scaler.transform(future_X)
        predicted_betas = model.predict(future_X_scaled)
        
        future_betas[stock] = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Beta': predicted_betas
        })
    
    # Combine future betas for visualization
    future_betas_df = pd.concat([future_betas[stock].assign(Stock=stock) for stock in future_betas], ignore_index=True)
    
except Exception as e:
    print(f"Error in ML beta prediction: {e}")
    raise

# Step 8: Data Visualizations using Seaborn
try:
    # 8.1: Line plot of normalized closing prices
    normalized_df = merged_df.copy()
    for col in ['NIFTY_Close', 'Price (REDY)', 'Price (HALC)', 'Price (TAMO)', 'Price (VARB)', 'Price (KTKM)']:
        normalized_df[col] = (normalized_df[col] / normalized_df[col].iloc[0]) * 100
    
    plt.figure(figsize=(14, 8))
    sns.lineplot(data=normalized_df.melt(id_vars='Date', value_vars=['NIFTY_Close', 'Price (REDY)', 'Price (HALC)', 'Price (TAMO)', 'Price (VARB)', 'Price (KTKM)']),
                 x='Date', y='value', hue='variable')
    plt.title('Normalized Closing Prices (Base=100) - Performance Comparison')
    plt.ylabel('Normalized Price')
    plt.xlabel('Date')
    plt.legend(title='Asset')
    plt.show()
    
    # 8.2: Heatmap of Correlation Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.3f')
    plt.title('Correlation Matrix: NIFTY vs Selected Stocks Returns')
    plt.show()
    
    # 8.3: Scatter plots for returns
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    axes = axes.flatten()
    stocks_list = ['REDY', 'HALC', 'TAMO', 'VARB', 'KTKM']
    for i, stock in enumerate(stocks_list):
        sns.scatterplot(ax=axes[i], data=returns_df, x='NIFTY_Return', y=f'{stock}_Return', alpha=0.5)
        sns.regplot(ax=axes[i], data=returns_df, x='NIFTY_Return', y=f'{stock}_Return', scatter=False, color='red')
        axes[i].set_title(f'NIFTY Return vs {stock} Return (Beta: {betas[stock]["Beta"]:.2f})')
        axes[i].set_xlabel('NIFTY Daily Return')
        axes[i].set_ylabel(f'{stock} Daily Return')
    
    fig.delaxes(axes[-1]) if len(stocks_list) % 2 != 0 else None
    plt.tight_layout()
    plt.show()
    
    # 8.4: Plot predicted betas
    plt.figure(figsize=(14, 8))
    sns.lineplot(data=future_betas_df, x='Date', y='Predicted_Beta', hue='Stock')
    plt.title('Predicted Betas for Next 30 Trading Days')
    plt.ylabel('Predicted Beta')
    plt.xlabel('Date')
    plt.legend(title='Stock')
    plt.show()
    
except Exception as e:
    print(f"Error in visualizations: {e}")
    raise

# Step 9: Interactive Visualization with Pygwalker
try:
    walker = pyg.walk(returns_df)
    walker
except Exception as e:
    print(f"Error in Pygwalker visualization: {e}")
    raise

# Step 10: Inferences for Predicted Betas
print("\nInferences for Predicted Betas:")
print("-" * 50)
print("1. **Beta Prediction Trends**:")
print("   - The machine learning model predicts future betas based on historical rolling betas, NIFTY mean returns, volatility, and time trends.")
for stock in ['REDY', 'HALC', 'TAMO', 'VARB', 'KTKM']:
    beta_trend = future_betas[stock]['Predicted_Beta'].iloc[-1] - future_betas[stock]['Predicted_Beta'].iloc[0]
    if beta_trend > 0:
        print(f"   - {stock}: Predicted beta is increasing (trend: {beta_trend:.3f}), suggesting growing sensitivity to NIFTY movements.")
    elif beta_trend < 0:
        print(f"   - {stock}: Predicted beta is decreasing (trend: {beta_trend:.3f}), indicating reduced sensitivity to NIFTY movements.")
    else:
        print(f"   - {stock}: Predicted beta is stable, implying consistent sensitivity to NIFTY movements.")
print("\n2. **Investment Strategy**:")
print("   - Stocks with increasing betas may offer higher returns in a bullish NIFTY market but carry higher risk.")
print("   - Stocks with decreasing betas may be safer during volatile NIFTY periods.")
print("   - These predictions assume historical trends continue; external factors (e.g., earnings, policy changes) may alter actual betas.")


# In[ ]:




