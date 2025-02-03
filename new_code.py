import pandas as pd
import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt
import re

# Read the Excel file
bonds = pd.read_excel("Final_Transformed_APM466_Data.xlsx", sheet_name="Final_Transformed_APM466_Data")

# Copy of bonds data
bonds_data = bonds.copy()

# Calculate adjusted bond prices
for i, row in bonds.iterrows():
    for col in bonds.columns[2:]:  # Start from the second column, excluding "Date"
        # Convert the datetime column header to string format 'yyyy-mm-dd'
        day = float(col.strftime('%d'))  # Extract the day as a string
        
        # Extract the number from the first column value (e.g., 'CAN 1.25 Mar 25' -> '1.25')
        coupon = float(row[0].split()[1])  # The number is always the second item in the row[0] string

        # Make sure row[col] is a valid numeric value
        try:
            price = float(row[col])
        except ValueError:
            # Handle non-numeric values gracefully
            continue
        # Apply the bond pricing formula
        bonds_data.at[i, col] = price + ((4*30 + (day - 1)) / 360) * coupon

# Calculate yield data
yield_data = bonds_data.copy()
for i, row in bonds.iterrows():
    for col in bonds.columns[2:]:  # Start from the second column, excluding "Date"
        # Extract the number from the first column value (e.g., 'CAN 1.25 Mar 25' -> '1.25')
        coupon = float(row[0].split()[1])  # The number is always the second item in the row[0] string
        maturity = int(row[1])
        coupon_frequency = 2  # Semi-annual coupon payments (2 periods per year)

        # Make sure row[col] is a valid numeric value
        try:
            price = float(row[col])
        except ValueError:
            # Handle non-numeric values gracefully
            continue
        
        # Calculate the number of periods (in semi-annual periods)
        num_periods = int(maturity * coupon_frequency)  # Total number of semi-annual periods
        fractional_period = (maturity * coupon_frequency) - num_periods  # Fractional part of the maturity in periods

        # Generate the cash flows (coupon payments for all full periods)
        cash_flows = []

        if num_periods > 0:
            # Add the coupon payments for all full periods
            cash_flows = [coupon / coupon_frequency] * num_periods  # Semi-annual coupon payments
            # Add the face value to the last full period's cash flow
            cash_flows[-1] += 100
        else:
            # For fractional periods, we calculate the final payment as a fraction of the coupon and the face value
            final_coupon = (coupon / coupon_frequency) * fractional_period  # Pro-rated final coupon for the fractional period
            cash_flows = [final_coupon + 100]  # The final cash flow includes the coupon and face value at maturity

        # Adjust for the non-integer maturity (fractional period)
        # Insert the negative price at the start to represent the initial cash outflow (your investment)
        cash_flows.insert(0, -price)

        # Check if cash_flows is not empty and calculate IRR
        if cash_flows:
            irr = npf.irr(cash_flows)

            # Convert to annual percentage rate (APR) for easier interpretation
            irr_percentage = irr * 100
        else:
            print("Error: Cash flows list is empty, cannot compute IRR.")
        
        yield_data.at[i, col] = irr_percentage * 2

# Extract Maturity Date and process yield data
yield_data["Maturity Date"] = yield_data["Date"].apply(lambda x: " ".join(re.findall(r"([A-Za-z]+ \d{2})", x)))
yield_data["Maturity Date"] = pd.to_datetime(yield_data["Maturity Date"], format="%b %y")
yield_data = yield_data.drop(columns=["Maturity"])
yield_data = yield_data.sort_values(by="Maturity Date")

# Updated Yields Plotting Function
def plot_yields(yield_data):
    plt.figure(figsize=(12, 6))

    # Iterate through each collection date (columns from index 2 onwards)
    for col in yield_data.columns[2:-2]:  # Exclude 'Maturity Date' and 'Date' columns
        # Get the collection date for this iteration
        collection_date = col.strftime('%Y-%m-%d') if hasattr(col, 'strftime') else str(col)
        
        # Prepare x and y values
        x_values = yield_data["Maturity Date"]
        y_values = yield_data[col]
        
        # Plot line for this collection date
        plt.plot(x_values, y_values, label=collection_date, marker='o')

    # Labels & Formatting
    plt.xlabel("Maturity Date")
    plt.ylabel("Yield")
    plt.title("Bond Yields Over Maturity")
    plt.xticks(rotation=45)
    plt.grid(True)

    # Fix legend placement
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), title="Collection Dates")

    plt.tight_layout()  # Adjust layout
    plt.savefig("yields_plot.png", dpi=300, bbox_inches="tight")  # Save plot to PNG
    plt.close()  # Close the plot to free up memory

# Spot Rate Calculation
spot_rate_data = bonds_data.copy()
coupon = float(spot_rate_data.iloc[0, 0].split()[1].strip())  # Extract the coupon percentage
for col in spot_rate_data.columns[2:]:  # Skip the first 2 columns
    spot_rate_data.at[0, col] = -1/spot_rate_data.iloc[0, 1] * np.log(spot_rate_data.at[0, col] / (100 + coupon / 2))

def calculate_spot_rate(spot_rate_data):
    # Loop through rows starting from index 1 (assuming 0th row is the header)
    for row_index in range(1, len(spot_rate_data)):
        coupon = float(spot_rate_data.iloc[row_index, 0].split()[1].strip())
        
        # Loop through the columns starting from the 3rd column
        for col in spot_rate_data.columns[2:]:
            exp_term = 0
            # Loop through all previous rows to accumulate the exponent terms
            for prev_row in range(row_index):
                exp_term += (coupon / 2) * np.exp(-1 * spot_rate_data.at[prev_row, col] * spot_rate_data.iloc[prev_row, 1])
            
            spot_rate_data.at[row_index, col] = -1 / spot_rate_data.iloc[row_index, 1] * np.log(
                (spot_rate_data.at[row_index, col] - exp_term) / (100 + coupon / 2)
            )
    spot_rate_data["Maturity Date"] = spot_rate_data["Date"].apply(lambda x: " ".join(re.findall(r"([A-Za-z]+ \d{2})", x)))
    spot_rate_data["Maturity Date"] = pd.to_datetime(spot_rate_data["Maturity Date"], format="%b %y")
    spot_rate_data = spot_rate_data.drop(columns=["Maturity"])
    spot_rate_data = spot_rate_data.sort_values(by="Maturity Date")
    return spot_rate_data

spot_rate_data = calculate_spot_rate(spot_rate_data)

# Updated Spot Rates Plotting Function
def plot_spot_rates(spot_rate_data):
    plt.figure(figsize=(12, 6))

    # Iterate through each collection date (columns from index 2 onwards)
    for col in spot_rate_data.columns[2:-2]:  # Exclude 'Maturity Date' and 'Date' columns
        # Get the collection date for this iteration
        collection_date = col.strftime('%Y-%m-%d') if hasattr(col, 'strftime') else str(col)
        
        # Prepare x and y values
        x_values = spot_rate_data["Maturity Date"]
        y_values = spot_rate_data[col]
        
        # Plot line for this collection date
        plt.plot(x_values, y_values, label=collection_date, marker='o')

    # Labels & Formatting
    plt.xlabel("Maturity Date")
    plt.ylabel("Spot Rate")
    plt.title("Spot Rates Over Maturity")
    plt.xticks(rotation=45)
    plt.grid(True)

    # Fix legend placement
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), title="Collection Dates")

    plt.tight_layout()  # Adjust layout
    plt.savefig("spot_rates_plot.png", dpi=300, bbox_inches="tight")  # Save plot to PNG
    plt.close()  # Close the plot to free up memory

# Forward Rates Calculation
forward_rates_data = spot_rate_data.copy()
forward_rates_data = forward_rates_data[forward_rates_data.index % 2 == 0]

def calculate_forward_rates(forward_rates_data):
    for i, row in forward_rates_data.iterrows():
        for col in forward_rates_data.columns[1:-1]:
            if i != 8:
                forward_rates_data.at[i + 2, col] = ((i / 2 + 2) * forward_rates_data.at[i + 2, col] - forward_rates_data.at[0, col]) / (i / 2 + 1)
    # Alternatively, to reset the index after dropping the row
    forward_rates_data = forward_rates_data.drop(forward_rates_data.index[0]).reset_index(drop=True)
    return forward_rates_data

forward_rates_data = calculate_forward_rates(forward_rates_data)

# Updated Forward Rates Plotting Function
def plot_forward_rates(forward_rates_data):
    plt.figure(figsize=(12, 6))

    # Iterate through each collection date (columns from index 2 onwards)
    for col in forward_rates_data.columns[2:-2]:  # Exclude 'Maturity Date' and 'Date' columns
        # Get the collection date for this iteration
        collection_date = col.strftime('%Y-%m-%d') if hasattr(col, 'strftime') else str(col)
        
        # Prepare x and y values
        x_values = forward_rates_data["Maturity Date"]
        y_values = forward_rates_data[col]
        
        # Plot line for this collection date
        plt.plot(x_values, y_values, label=collection_date, marker='o')

    # Labels & Formatting
    plt.xlabel("Maturity Date")
    plt.ylabel("Forward Rate")
    plt.title("Forward Rates Over Maturity")
    plt.xticks(rotation=45)
    plt.grid(True)

    # Fix legend placement
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), title="Collection Dates")

    plt.tight_layout()  # Adjust layout
    plt.savefig("forward_rates_plot.png", dpi=300, bbox_inches="tight")  # Save plot to PNG
    plt.close()  # Close the plot to free up memory

# Covariance Matrix Calculation
def get_covariance_matrix(yield_df, forward_rate_df):
    # Prepare data for covariance calculation
    yield_data = yield_df.drop(columns=["Date", "Maturity Date"]).values
    forward_data = forward_rate_df.drop(columns=["Date", "Maturity Date"]).values

    # Compute log-returns for yield rates
    log_returns_yield = np.log(yield_data[:, 1:] / yield_data[:, :-1])

    # Compute covariance matrix for yield rates
    cov_matrix_yield = np.cov(log_returns_yield)

    # Compute log-returns for forward rates
    log_returns_forward = np.log(forward_data[:, 1:] / forward_data[:, :-1])

    # Compute covariance matrix for forward rates
    cov_matrix_forward = np.cov(log_returns_forward)

    # Convert results to DataFrames for better visualization
    cov_df_yield = pd.DataFrame(cov_matrix_yield, 
                                index=yield_df["Maturity Date"], 
                                columns=yield_df["Maturity Date"])

    cov_df_forward = pd.DataFrame(cov_matrix_forward, 
                                index=forward_rate_df["Maturity Date"], 
                                columns=forward_rate_df["Maturity Date"])

    # Output the covariance matrices
    print("Covariance Matrix for Yield Rates:")
    cov_df_yield.to_excel("Cov_Yield.xlsx")
    print(cov_df_yield)

    print("\nCovariance Matrix for Forward Rates:")
    cov_df_forward.to_excel("Cov_Forward.xlsx")
    print(cov_df_forward)
    return cov_df_yield, cov_df_forward

# Select data for covariance calculation
cov_yield_data = yield_data[yield_data.index % 2 == 0]

# Calculate and print covariance matrices
cov_y, cov_f = get_covariance_matrix(cov_yield_data, forward_rates_data)

# Compute eigenvalues and eigenvectors
eigenvalues_y, eigenvectors_y = np.linalg.eig(cov_y.values)
eigenvalues_f, eigenvectors_f = np.linalg.eig(cov_f.values)

# Print results
print("Eigenvalues of Yield Covariance Matrix:")
print(eigenvalues_y)

print("\nEigenvectors of Yield Covariance Matrix:")
print(eigenvectors_y)

print("\nEigenvalues of Forward Rate Covariance Matrix:")
print(eigenvalues_f)

print("\nEigenvectors of Forward Rate Covariance Matrix:")
print(eigenvectors_f)

# Generate plots
plot_yields(yield_data)
plot_spot_rates(spot_rate_data)
plot_forward_rates(forward_rates_data)