# IMC Prosperity 4 - Trading Algorithms

IMC Prosperity 4 trading algorithms

## Overview
This repository contains the source code for the algorithmic trading challenge in IMC Prosperity 4. The main algorithm is implemented as a `Trader` class in Python, designed to trade on the Prosperity exchange against bots to maximize profitability. 

## Project Structure
- `submission.py`: Contains the `Trader` class.
- `datamodel.py`: Provided by IMC. Contains data structures like `TradingState`, `Order`, and `OrderDepth`.



## Trading strategy for each item

### Emeralds:

At each timestep, immediately take any favorable trades available - Buying below 10,000 or selling above it. Afterward, place passive quotes slightly better than any existing liquidity (existing orders in orderbook): overbidding on bids and undercutting on asks while maintaining positive edge. If inventory becomes too skewed, we flatten it at exactly 10,000 to free up risk capacity for the next opportunities.

### Tomatoes:

At each time step, immediately take any favorable trades available relative to the current mid, then place slightly improved passive orders (overbidding and undercutting) around the fair price. If inventory becomes too large, we neutralize it by trading at zero edge relative to the current price estimate.


Slight mean reversion (Regression towards mean): True price of tomatoes are floating point numbers but orders can be placed only as integers.




