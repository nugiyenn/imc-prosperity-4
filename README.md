# IMC Prosperity 4 - Algorithmic Trading

This repository contains our Python submissions for the **IMC Prosperity 4** algorithmic trading challenge. It was our first time participating, and we placed **#1,299 globally** (out of 18,800+ teams) and **#2 in Finland** with a final PnL of **172,790 XIRECs**.

The code here consists of standalone trading algorithms designed to interact with IMC's simulated order book matching engine—handling quoting, taking liquidity, and managing risk across different products. The official API structures (like `TradingState` and `Order`) are defined in `datamodel.py`.

---

## Strategy Breakdown

### Round 1: Trend & Mean Reversion
* **Assets**: `INTARIAN_PEPPER_ROOT` (IPR) and `ASH_COATED_OSMIUM` (ASH).
* **Strategy**: 
  * **IPR**: Exploited a deterministic linear upward drift by building a maximum long position early and holding, followed by aggressive unwinding in the final ticks to lock in PnL.
  * **ASH**: Modeled as an Ornstein-Uhlenbeck (OU) mean-reversion process around a fair value of 10,000, executed using both passive quoting (market-making) and active crossing (market-taking) layers.

### Round 2: Sealed-Bid Auction Game
* **Assets**: Same as Round 1.
* **Strategy**: 
  * Implemented an auction-bidding mechanism to secure additional bot trading volume.
  * Extended the Round 1 model by implementing conservative inventory limits and skew management to hedge against potential distribution shifts (mean drifting) in the underlying assets.

### Round 3: Options & Spot
* **Assets**: `HYDROGEL_PACK` (Spot), `VELVETFRUIT_EXTRACT` (Spot), and 10 call option strikes (`VEV_4000` to `VEV_6500`).
* **Strategy**: 
  * Calculated theoretical option prices, Greeks (Delta), and implied volatility using a Black-Scholes bisection solver in pure Python.
  * Used deviations from short-window Exponential Moving Averages (EMAs) to execute mean-reversion trades across both spot assets and options.

### Round 4: Bot Profiling & Follow-Trading
* **Assets**: Same as Round 3.
* **Strategy**: 
  * Analyzed historical trade logs to profile and rank individual trading bots by profitability.
  * Tracked informed traders (following the orders of `Mark 67` and `Mark 14`, and fading the noise-driven orders of `Mark 38`) and integrated a follow-trading overlay to shadow their execution.

### Round 5: Multi-Asset Arbitrage & Pairs
* **Assets**: 50 assets grouped across 10 sectors (including `PEBBLES`, `SNACKPACK`, `MICROCHIP`, `ROBOT`, etc.).
* **Strategy**: 
  * Portfolio-scale statistical arbitrage, pairs trading, and selective market-making. 
  * Exploited highly correlated structures (e.g., trading the sum of negatively correlated pairs like chocolate/vanilla snackpacks, or the spread of positively correlated pairs like vacuum/laundry robots).
  * Implemented cross-pricing models for related assets (e.g., pricing panels and oxygen shake variants using co-movement deviations) and utilized cross-group hedging (hedging robot positions using sleep pods) to control risk.



## Tools & Validation

To test and refine our strategies, we utilized these external tools:
* [imc-prosperity-4-backtester](https://github.com/nabayansaha/imc-prosperity-4-backtester) for local backtesting and simulation.
* [Equirag Prosperity Log Visualizer](https://prosperity.equirag.com/) for inspecting sandbox logs and analyzing trade execution.

---

## File Structure

* `round1.py` to `round5.py` - Standalone trading algorithm submissions containing the `Trader` class for each corresponding round.
* `datamodel.py` - Official IMC Prosperity library containing state definitions and order object classes.