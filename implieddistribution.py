import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import pandas as pd
import matplotlib.pyplot as plt

def black_scholes_call_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def implied_volatility(S, K, T, r, market_price):
    # Function to find the implied volatility
    objective = lambda sigma: market_price - black_scholes_call_price(S, K, T, r, sigma)
    return brentq(objective, 1e-6, 2)

def breeden_litzenberger(C, K, r, T):
    # Numerically estimate second derivative of call price wrt strike
    d2C_dK2 = np.gradient(np.gradient(C, K), K)
    f_RN = np.exp(r * T) * d2C_dK2
    return f_RN

# Example data
S = 100  # Underlying price
r = 0.02  # Risk-free rate
T = 1  # Time to maturity in years
strikes = np.array([80, 90, 100, 110, 120])
market_prices = np.array([25, 17, 10, 5, 2])  # Example market prices for call options

# Calculate implied volatilities
implied_vols = np.array([implied_volatility(S, K, T, r, mp) for K, mp in zip(strikes, market_prices)])

# Calculate risk-neutral density
f_RN = breeden_litzenberger(market_prices, strikes, r, T)

# Adjust to real-world probabilities (assuming a market price of risk)
sigma_avg = np.mean(implied_vols)
lambda_mpr = 0.3  # Example value for market price of risk
mu_real = r + lambda_mpr * sigma_avg

# Expected return (simplified)
expected_return = mu_real

print(f"f_RN : {f_RN}")
print(f"Market Price of Risk (lambda): {lambda_mpr}")
print(f"Expected Return (mu): {expected_return:.2f}")

df = pd.DataFrame({
    'Strike': strikes,
    'Risk-Neutral Density': f_RN
})



print(df)

plt.figure(figsize=(10, 6))
plt.plot(strikes, f_RN, marker='o', linestyle='-', color='b')
plt.title('Risk-Neutral Density Distribution')
plt.xlabel('Strike Price')
plt.ylabel('Risk-Neutral Density')
plt.grid(True)
plt.show()