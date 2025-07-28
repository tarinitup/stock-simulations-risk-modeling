# Quantitative Analysis: Simulating Stock Behavior & Modeling Portfolio Risk

---

As a casual retail investor, I've always been curious as to how anyone can be so certain that a portfolio or single stock will head in a particular direction. For many years, I had always treated the market as little more than laying bets in Vegas. Driven by curiosity and my own research, this led me to learning about quantitative analysis, the holy grail of what modern analysts wield to tame uncertainty in the markets, turning investing and trading from blind luck into a mathmatical-based calculated strategies.

This project showcases how we can leverage mathmatical models and historical data using Python to realistically model price movement behaviors of stocks, analyze holistic portfolio performance, and quantify financial risk levels to guide strategic planning and data-driven decision making. We will cover fundamental quantitative finance techniques like Geometric Brownian Motion (GBM), Monte Carlo Simulations, and Cholesky Decomposition to calculate expected return values, Value-at-Risk (VaR), and downside probabilities.

---

## Geometric Brownian Motion

Geometric Brownian Motion (GBM) is a stochastic process (a model combining both a predictable trend and random fluctuations) and fundamental technique widely used to simulate and analyze asset prices in quantitative finance. GBM captures two core effects:

1. **Drift** ($\mu$) – The steady, average percentage growth over time.  
2. **Volatility** ($\sigma$) – The size of random fluctuating swings around that trend.

GBM’s simplicity and realistic price-dynamics make it the foundation for models like Black–Scholes and conducting Monte Carlo simulations.

---

## Using GBM with Financial Data

GBM relies on a few assumptions, yet are suitable for working with financial data since they reproduce many observed features of real markets:

- **Constant drift** ($\mu$) and **constant volatility** ($\sigma$) over the simulation period.  
- **Log-normal distribution** of prices, ensuring ($S_t>0$) and symmetric behavior of percentage returns in log-space.  
- **Continuous compounding** of returns, so tiny gains and losses accumulate smoothly.

In other words, GBM assumes that stock prices grow continuously over time, but with random fluctuations. These properties make GBM a great tool for risk analysis, option pricing, and scenario generation.

---

## Continuous-Time GBM Formula

In a theoretical continuous time, GBM is defined by this stochastic differential equation:

$$
dS_t = \mu\,S_t\,dt \;+\; \sigma\,S_t\,dW_t
$$

- **$dS_t$**: Infinitesimal change in price over a tiny interval \(dt\)  
- **$\mu$**: Drift rate (the expected return per unit of time)
- **$S_t$**: Asset price at time \(t\)
- **$dt$**: Infinitesimal time increment
- **$\sigma$**: Volatility, the scale of random shocks per $\sqrt{t}$ 
- **$dW_t$**: Infinitesimal random shock, $\sim \mathcal{N}(0,dt)$

### Explicit Solution

Integrating gives the log-normal form:

$$
S_t = S_0 \exp\!\Bigl(\bigl(\mu - \tfrac12\sigma^2\bigr)\,t + \sigma\,W_t\Bigr)
$$

- The term $-\tfrac12\sigma^2$ adjusts for the fact that random volatility inflates the mean in log-space.

---

## Discrete-Time GBM for Simulation

 Our computer can simulate GBM by dividing time into $N$ finite steps of size $\Delta t$. The recursive update is:

$$
S_{t+1} = S_t \cdot \exp\left( \left( \mu - \frac{1}{2} \sigma^2 \right) \Delta t + \sigma \sqrt{\Delta t} \cdot Z_t \right)
$$

- **$S_{t+1}$**: Simulated price at the next step ($t+1$); "Tomorrow's Price = ..." 
- **$S_t$**: Price at step $t$  
- **$\Delta t$**: Time increment (e.g. $1/252$ for daily increments out of total trading days per year) 
- **$\mu$**: Drift rate (annualized average return) 
- **$\sigma$**: Volatility (annualized standard deviation of returns)  
- **$\exp(\cdot)$**: Exponential function, ensuring prices remain positive
- **$Z_t$**: Random numerical draw from a standard normal distribution $\mathcal{N}(0,1)$, representing the discrete shock

- **$\bigl(\mu - \tfrac12\sigma^2\bigr)\Delta t$**: Deterministic “drift adjustment” in log-space  

- **$\sigma\sqrt{\Delta t}\,Z_t$**: Stochastic “volatility shock” scaled by step size and randomness


Essentially, this is what our formula is doing in plain english:

"To Simulate Future Change in Stock Price at Given Time = Take Current Price, Adjust it Using Expected Average Return and Expected Volatility, Plus Inject Some Randomness"

---

## Simulating GBM Using Python

Check out how I simulated an asset's price trajectory [HERE](https://github.com/tarinitup/stock-simulations-risk-modeling/blob/main/gbm_sim.ipynb) using Python.

For this example, I used ticker **VOO** (Vanguard S&P 500). This will take VOO's last close price, then project a potential 1-year future price path using the GBM concepts discussed above.

```python

# annualize the mean and volatility
trading_days = 252
mu = log_returns.mean() * trading_days  # expected annual return
sigma = log_returns.std() * np.sqrt(trading_days)   # annual volatility

# starting price for the simulation (the last close price)
S0 = prices.iloc[-1]

# formula parameters
T  = 1             # horizon: 1 year
N  = trading_days  # number of "steps" (≈ trading days)
dt = T / N         # time increment in years

# generate random shocks and volatility
Z = np.random.normal(size=N)    # drawing random numbers from a normal distribution
S = np.zeros(N)     # pre-allocate the array for the simulated prices
S[0] = S0   # set the starting price
for t in range(1, N):
    S[t] = S[t-1] * np.exp(
        (mu - 0.5 * sigma**2) * dt
        + sigma * np.sqrt(dt) * Z[t]
    )

```

**Here is an example output chart:**

![GBM Output Example](https://github.com/tarinitup/stock-simulations-risk-modeling/blob/main/figures/gbm_sim_output.png)

---

## Monte Carlo Simulation

Now - simulating one possibly price path using GBM is great, but we still don't know the full range of all potential futures, given the fact that there could be many scenarios for random volatility to affect the market to different degrees and in different ways. 

If you ran the GBM simulation code eariler multiple times, you may have noticed that the price paths of the output charts are different each time. This is due to the random volatility we inject at each time step ($\sigma\sqrt{\Delta t}\,Z_t$). So - How can we capture and visualize all the different potential price paths?

This is where Monte Carlo simulation comes in - a method for estimating the distribution of potential outcomes by running many randomized trials. Applying Monte Carlo simulations on top of our GBM model will help with:

- **Capturing Uncertainty**  
  Markets are random. Monte Carlo shows the full spread of possibilities, not just one “most likely” path.

- **Quantifying Risk**  
  By looking at the worst 5% or 1% of outcomes, you can compute VaR, stress-test your portfolio, or set risk limits.

- **Adjusting Scenarios**  
  You can easily change inputs—drift, volatility, time horizon—and see how the distribution shifts.

- **Guiding Decisions**  
  Helps answer practical questions like:  
  - “What’s the chance my retirement fund drops below X?”  
  - “How often would a new trading strategy lose money over a year?”


### Adjusted GBM for Monte Carlo

To run a Monte Carlo simulation on top of our GBM framework, we will add a new index to track each of the **$m$** number of paths we want to simulate.
Here is the adjusted equation:

$$
S_{m,t+1} = S_{m,t} \cdot \exp\left( \left( \mu - \frac{1}{2} \sigma^2 \right) \Delta t + \sigma \sqrt{\Delta t} \cdot Z_{m,t} \right)
$$

- **$m = 1,2,\dots,M$**  
  Index for each Monte Carlo path.

- **$S_{m,t}$**  
  Price on path $m$ at time step $t$.

- **$Z_{m,t}\sim\mathcal{N}(0,1)$**  
  Independent standard-normal shock for path $m$ at step $t$.

Other symbols remain:

- **$\mu$**: annualized drift  
- **$\sigma$**: annualized volatility  
- **$\Delta t$**: time increment per step  

By looping this update for $t=0,\ldots,N-1$, we generate $M$ full GBM trajectories $\{S_{m,0\ldots N}\}$ and can then analyze the distribution of all possible outcomes.  


---

## Monte Carlo Simulation Using Python

You can see how I applied a Monte Carlo simulation to our GBM model [HERE](https://github.com/tarinitup/stock-simulations-risk-modeling/blob/main/monte_carlo.ipynb).

Earlier, we simulated a single possible 1-year future price path of VOO using GBM. Now when applying our Monte Carlo simulation on top of our GBM model, we can now run as many simulations for stress testing and measuring risk under exteme market conditions.

For this excercise, I am running 10,000 simulations as a starting baseline. This should help give a better full-picture view on the full range of potential outcomes.

```python

# annualize the mean and volatility
trading_days = 252
mu = log_returns.mean() * trading_days
sigma = log_returns.std() * np.sqrt(trading_days)

# starting price for the simulation (the last close price)
S0 = prices.iloc[-1]

# monte carlo parameters
T  = 1
N  = trading_days
dt = T / N
M  = 10000

# pre-allocate paths and draw shocks
price_paths = np.zeros((M, N+1))    # pre-allocate the array for the simulated prices
price_paths[:, 0] = S0              # set the starting price
Z = np.random.normal(size=(M, N))   # drawing random numbers from a normal distribution

# simulate gbm across all paths - price_paths[:, t] creates the vector of prices at time t for each path
for t in range(1, N+1):
    price_paths[:, t] = price_paths[:, t-1] * np.exp(
        (mu - 0.5 * sigma**2) * dt
        + sigma * np.sqrt(dt) * Z[:, t-1]
    )

# extract the final prices from the last column of the price_paths array
final_prices = price_paths[:, -1] 

```

Here is the output of the full range of potential paths VOO can take over 1 year using our Monte Carlo and GBM model.

![Monte Carlo Simulation Output](https://github.com/tarinitup/stock-simulations-risk-modeling/blob/main/figures/monte_carlo_output.png)

Our plots here let us see our 10,000 simulated paths (left) and a distribution histogram of our simulated final prices (right).

Across our 10,000 simulated paths, we can now see the average mean final price outcome, and measure downside risk metrics like VaR and Probability of Loss.

- **Value at Risk (VaR)**: Over a specified time horizon, what's the largest loss I can expect with **$x$%** confidence?

  (For example: **5% VaR: $100 (Loss: $25)** means a 5% chance the price will drop to $100 or less; We are 95% confident that we will not lose more than $25)

- **Probability of Loss**: The probability your return over the specificed time horizon falls below the asset's starting value

---



## Simulating Multi-Stock Portfolio with Correlated Assets (GBM + Monte Carlo)

So far, we've simulated the potential price path of an individual stock using GBM. Then, we gathered the full range of potential price path outcomes that individual stock could have over a specified time period. Now, how do we apply this to an entire portfolio that is comprised of many assets?

In this section, we'll build on top of our GBM and Monte Carlo simulation models, but applied on a portfolio-level scale.

However, we need to keep in mind that stocks don’t just move independently in isolation — they often move together in related patterns. For example, when a tech stocks rises, others in the same sector often rise together with them. To simulate this behavior, we will integrate correlations between the stocks in our portfolio using matrix multiplication!

Doing so will allow us to model how an entire multi-stock portfolio may realistically behave, and analyze the financial risk of our portfolio as a whole. 

Let's break down the methodology of how we'll build the correlated behaviors between our portfolio's assets in 4 steps:

---

### Step 1: Build the Correlation Matrix

First, we create a **correlation matrix** using our portfolio's stocks. Think of this matrix as a table that shows how each stock’s daily returns relate to the others, based on historical price data over our chosen time window (e.x. 1 year, 3 years, max available, etc.).

We use the **Pearson correlation coefficient** (also known as the **r-value**) to fill in the matrix. These values range from:  
- $+1$: perfectly positively correlated  
- $0$: no correlation  
- $-1$: perfectly negatively correlated  

**FOR EXAMPLE:**

Let’s say we calculate daily return correlations using the tickers in the portfolio I am using for this excersise:  
`["NVDA", "PLTR", "ANET", "VRT", "VOO"]`

In Python, this step is done using:

```python
corr_matrix = np.corrcoef(log_returns_matrix.T)
```

This line calculates the Pearson correlation coefficients between each pair of stocks, based on their daily log returns.

Behind the scenes, the correlation matrix might look something like this:

|         | NVDA | PLTR | ANET | VRT  | VOO  |
|--------:|:----:|:----:|:----:|:----:|:----:|
| **NVDA**| 1.00 | 0.65 | 0.55 | 0.60 | 0.75 |
| **PLTR**| 0.65 | 1.00 | 0.40 | 0.45 | 0.55 |
| **ANET**| 0.55 | 0.40 | 1.00 | 0.50 | 0.60 |
| **VRT** | 0.60 | 0.45 | 0.50 | 1.00 | 0.65 |
| **VOO** | 0.75 | 0.55 | 0.60 | 0.65 | 1.00 |

This table shows, for example, that:
- NVDA and VOO have a strong positive correlation (0.75)
- PLTR and ANET are less correlated (0.40)
- Each stock is perfectly correlated with itself (1.00 on the diagonal)

This matrix is the foundation for creating realistic interdependencies between asset movements.

---

### Step 2: Decompose the Correlation Matrix with Cholesky

Once we have our correlation matrix, we need to construct the **covariance matrix**, which includes both:
- The relationships between the stocks (correlation), and  
- The magnitude of their individual volatilities (standard deviations)

This is done with the formula:
```python
cov_matrix = np.outer(sigma, sigma) * corr_matrix
```
This combines each stock's volatility with the correlation matrix to give us the full covariance matrix.

Next, we use **Cholesky decomposition** to factor the covariance matrix into:

$$
A = L \cdot L^T
$$

Where:
- $A$ is the covariance matrix  
- $L$ is the lower-triangular matrix  
- $L^T$ is its transpose  

In Python, this is done with:
```python
L = np.linalg.cholesky(cov_matrix)
```

We use **only the $L$ factor**, which allows us to transform independent random noise into realistic, correlated movements — mimicing the relationships between the stocks.

---

### Step 3: Generate Random Shock Vectors ($Z$)

Next, we generate a large matrix of random values from a **standard normal distribution**. These are your $Z$ values — the random "shocks" that simulate market noise or surprise events.

These shocks are:
- Centered around zero  
- Uncorrelated with each other  

Each asset gets its own set of random values. But at this stage, they're still totally independent — not yet tied together by correlation.

---

### Step 4: Create Correlated Shocks via Matrix Multiplication

Now we multiply the Cholesky $L$ matrix by our random shocks $Z$:

$$
dW = L \cdot Z
$$

This final step gives us a matrix of **correlated shocks** — random movements that now realistically reflect how our stocks behave together.

> Now, stocks like NVDA and VOO now move in sync (if they’re highly correlated), while stocks like PLTR and ANET may move more independently.

We then use these correlated shocks in our Monte Carlo simulations to generate realistic price paths for the entire portfolio across thousands of future scenarios.

---

## Portfolio Simulation Using Python

You can see how I applied a portfolio-level simulation with multi-asset correlation behavior on top of our GBM and Monte Carlo framework [HERE](https://github.com/tarinitup/stock-simulations-risk-modeling/blob/main/portfolio_analysis.ipynb).

```python

# dynamically compute correlation and covariance matrices
corr_matrix = np.corrcoef(log_returns_matrix.T)
cov_matrix = np.outer(sigma, sigma) * corr_matrix
L = np.linalg.cholesky(cov_matrix)

# simulate GBM paths
price_paths = np.zeros((M, N+1, n_assets))
price_paths[:, 0, :] = S0

for m in range(M):
    Z = np.random.normal(size=(N, n_assets))
    shocks = Z @ L.T
    for t in range(1, N+1):
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = np.sqrt(dt) * shocks[t-1]
        price_paths[m, t, :] = price_paths[m, t-1, :] * np.exp(drift + diffusion)
```

Here is the output of our portfolio-level Monte Carlo simulation plot (left) and our distribution histogram (right):

![Portfolio Simulation Output](https://github.com/tarinitup/stock-simulations-risk-modeling/blob/main/figures/portfolio_sim_output.png)

Now we can realistically measure our various metrics like Expect Return, VaR, and Probability Loss over the projected time period for our portfolio!
