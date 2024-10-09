import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Title of the app
st.title('Copper Price Forecasting with Variance')

# Sample data (you can replace this with actual data)
data = {
    'Month': ['Jan 2023', 'Feb 2023', 'Mar 2023', 'Apr 2023', 'May 2023', 'Jun 2023', 'Jul 2023'],
    'Actual_Price': [774.4, 758.7, 784.95, 743.4, 703.55, 705.95, 745.15]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Convert Month to datetime for accurate forecasting
df['Month'] = pd.to_datetime(df['Month'])
df['Month_Number'] = np.arange(1, len(df) + 1)

# Linear Regression Model
X = df[['Month_Number']]
y = df['Actual_Price']
model = LinearRegression()
model.fit(X, y)

# Predict prices for historical months
df['Predicted_Price'] = model.predict(X)

# Input: Number of months to forecast
st.sidebar.header('Forecast Settings')
num_months = st.sidebar.slider('Select number of months to forecast:', 1, 3, 2)

# Generate future month dates
last_month = df['Month'].iloc[-1]
future_months = [last_month + pd.DateOffset(months=i) for i in range(1, num_months + 1)]
future_month_labels = [month.strftime('%b %Y') for month in future_months]

# Predict future prices
predicted_future_prices = model.predict(np.arange(len(df) + 1, len(df) + 1 + num_months).reshape(-1, 1))

# Create future dataframe
future_df = pd.DataFrame({
    'Month': future_month_labels,
    'Predicted_Price': predicted_future_prices
})

# Display future prices
st.subheader('Forecasted Prices for Upcoming Months')

# Input: Actual values for forecasted months (optional)
actual_future_prices = st.sidebar.text_input('Enter actual prices for forecasted months, separated by commas (e.g., 750, 760)', '')

# Check for correct number of actual prices
if actual_future_prices:
    actual_future_prices = [float(x) for x in actual_future_prices.split(',')]
    if len(actual_future_prices) == num_months:
        future_df['Actual_Price'] = actual_future_prices
        future_df['Variance (%)'] = ((future_df['Predicted_Price'] - future_df['Actual_Price']) / future_df['Actual_Price']) * 100
        st.write(future_df)
    else:
        st.error(f"Please provide exactly {num_months} actual price values.")
else:
    st.write(future_df)

# Plot actual vs predicted prices
st.subheader('Price Forecast Visualization')
fig, ax = plt.subplots(figsize=(10, 6))  # Increase figure size for better spacing
ax.plot(df['Month'], df['Actual_Price'], label='Actual Price', marker='o')
ax.plot(df['Month'], df['Predicted_Price'], label='Predicted Price', linestyle='--', marker='x')

# Plot future predicted prices
ax.plot(future_months, predicted_future_prices, label='Future Forecast', linestyle='-.', marker='s', color='red')

ax.set_xlabel('Year/Month')
ax.set_ylabel('Price (INR/KG)')
ax.set_title('Actual vs Predicted Prices')
ax.legend()

# Combine the datetime objects for ticks
all_months = np.concatenate([df['Month'], future_months])
ax.set_xticks(all_months)  # Set x-ticks for actual and future months
ax.set_xticklabels(np.concatenate([df['Month'].dt.strftime('%b %Y'), future_month_labels]), rotation=45, ha='right')

# Display the plot in Streamlit
st.pyplot(fig)

# Calculate variance (error percentage) for historical data
df['Variance (%)'] = ((df['Predicted_Price'] - df['Actual_Price']) / df['Actual_Price']) * 100

# Display variance table
st.subheader('Variance between Actual and Predicted Prices (For values used for Model Training)')
st.write(df[['Month', 'Actual_Price', 'Predicted_Price', 'Variance (%)']])
