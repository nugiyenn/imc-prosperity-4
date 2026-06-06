import json
import math
from typing import Dict, List, Any, Optional, Tuple
from datamodel import OrderDepth, TradingState, Order, Trade


def norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def bs_call_price(S: float, K: float, T: float, r: float, vol: float) -> float:
    """Black-Scholes theoretical pricing for a European Call option."""
    if T <= 0 or vol <= 0:
        return max(0.0, S - K)
    d1 = (math.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * math.sqrt(T))
    d2 = d1 - vol * math.sqrt(T)
    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)


def bs_call_delta(S: float, K: float, T: float, r: float, vol: float) -> float:
    """Black-Scholes theoretical Delta for a European Call option."""
    if T <= 0 or vol <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * math.sqrt(T))
    return norm_cdf(d1)


class Trader:
    MARKET_ACCESS_BID = 20

    HYDRO = "HYDROGEL_PACK"
    HYDRO_LIMIT = 200

    # Hydrogel parameters
    EMA_ALPHA = 0.015
    DEV_DEADBAND = 0.5
    DEV_TO_TARGET = 10.0
    SOFT_POS = 190
    INV_PENALTY = 0.10

    TAKER_DEV = 9.0
    TAKER_CLIP = 28

    BASE_SIZE = 34
    MAX_ADD_SIZE = 52

    HYDRO_FLOW_CLIP = 25.0
    HYDRO_FLOW_TO_TARGET = 0.8
    HYDRO_FLOW_TO_FAIR = 0.35

    def __init__(self):
        # Position limits
        self.limits = {"HYDROGEL_PACK": 200, "VELVETFRUIT_EXTRACT": 200}
        # Vouchers have 300 limit each
        self.vouchers = [
            "VEV_4000",
            "VEV_4500",
            "VEV_5000",
            "VEV_5100",
            "VEV_5200",
            "VEV_5300",
            "VEV_5400",
            "VEV_5500",
            "VEV_6000",
            "VEV_6500",
        ]
        for v in self.vouchers:
            self.limits[v] = 300

        # Volatility guess for VELVETFRUIT_EXTRACT
        self.vol = 0.02
        self.risk_free_rate = 0.0

    def bid(self) -> int:
        return self.MARKET_ACCESS_BID

    @staticmethod
    def _best_bid_ask(depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
        best_bid = max(depth.buy_orders) if depth.buy_orders else None
        best_ask = min(depth.sell_orders) if depth.sell_orders else None
        return best_bid, best_ask

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    @staticmethod
    def _remaining_capacity(position: int, limit: int) -> Tuple[int, int]:
        max_buy = max(0, limit - position)
        max_sell = max(0, limit + position)
        return max_buy, max_sell

    def _trade_hydro(self, state: TradingState, depth: OrderDepth, persistent: Dict) -> List[Order]:
        best_bid, best_ask = self._best_bid_ask(depth)
        if best_bid is None or best_ask is None:
            return []

        mid = 0.5 * (best_bid + best_ask)
        spread = best_ask - best_bid

        hydro_state = persistent.get("hydro", {})
        if not isinstance(hydro_state, dict):
            hydro_state = {}

        ema_old = hydro_state.get("ema")
        if not isinstance(ema_old, (int, float)):
            ema_old = mid
        ema_new = (1.0 - self.EMA_ALPHA) * float(ema_old) + self.EMA_ALPHA * mid
        hydro_state["ema"] = ema_new
        persistent["hydro"] = hydro_state

        position = int(state.position.get(self.HYDRO, 0))
        rem_buy, rem_sell = self._remaining_capacity(position, self.HYDRO_LIMIT)

        flow_signal = float(persistent.get("signal_hydro", 0.0))
        flow_signal = self._clamp(flow_signal, -self.HYDRO_FLOW_CLIP, self.HYDRO_FLOW_CLIP)

        dev = mid - ema_new
        dev_eff = 0.0 if abs(dev) < self.DEV_DEADBAND else dev

        raw_target = (
            -self.DEV_TO_TARGET * dev_eff
            - self.INV_PENALTY * position
            + self.HYDRO_FLOW_TO_TARGET * flow_signal
        )
        target = int(round(self._clamp(raw_target, -self.SOFT_POS, self.SOFT_POS)))

        buy_need = max(0, target - position)
        sell_need = max(0, position - target)

        orders: List[Order] = []
        projected = position

        if dev <= -self.TAKER_DEV and rem_buy > 0:
            avail = max(0, -int(depth.sell_orders.get(best_ask, 0)))
            boost = int(max(0.0, -dev - self.TAKER_DEV))
            qty = min(rem_buy, avail, self.TAKER_CLIP + boost)
            if qty > 0:
                orders.append(Order(self.HYDRO, int(best_ask), int(qty)))
                rem_buy -= qty
                rem_sell += qty
                projected += qty
        elif dev >= self.TAKER_DEV and rem_sell > 0:
            avail = max(0, int(depth.buy_orders.get(best_bid, 0)))
            boost = int(max(0.0, dev - self.TAKER_DEV))
            qty = min(rem_sell, avail, self.TAKER_CLIP + boost)
            if qty > 0:
                orders.append(Order(self.HYDRO, int(best_bid), -int(qty)))
                rem_sell -= qty
                rem_buy += qty
                projected -= qty

        fair = (
            ema_new
            - self.INV_PENALTY * projected
            + self.HYDRO_FLOW_TO_FAIR * flow_signal
        )
        inside = 1 if spread >= 2 else 0
        skew = int(self._clamp(round((fair - mid) / 2.0), -3, 3))

        bid_quote = int(best_bid + inside + skew)
        ask_quote = int(best_ask - inside + skew)
        if bid_quote >= ask_quote:
            bid_quote = int(best_bid)
            ask_quote = int(best_ask)
        if bid_quote >= ask_quote:
            ask_quote = bid_quote + 1

        buy_size = min(rem_buy, self.BASE_SIZE + min(self.MAX_ADD_SIZE, buy_need // 2))
        sell_size = min(rem_sell, self.BASE_SIZE + min(self.MAX_ADD_SIZE, sell_need // 2))

        if projected >= self.SOFT_POS:
            buy_size = 0
        if projected <= -self.SOFT_POS:
            sell_size = 0

        if buy_size > 0:
            orders.append(Order(self.HYDRO, bid_quote, int(buy_size)))
        if sell_size > 0:
            orders.append(Order(self.HYDRO, ask_quote, -int(sell_size)))

        return orders

    def run(self, state: TradingState) -> tuple[dict[str, list[Order]], int, str]:
        orders = {}

        # 1. Parse and update state (Signal tracking)
        trader_data = {
            "signal_velvet": 0.0,
            "signal_hydro": 0.0,
            "last_time": 0,
            "day": 1,
        }
        if state.traderData:
            try:
                trader_data = json.loads(state.traderData)
            except:
                pass

        # Update day tracking
        if state.timestamp < trader_data.get("last_time", 0):
            trader_data["day"] = trader_data.get("day", 1) + 1
        trader_data["last_time"] = state.timestamp

        # Decay signals exponentially
        trader_data["signal_velvet"] *= 0.90
        trader_data["signal_hydro"] *= 0.90

        # Analyze current market trades to build signals
        for product, trades in state.market_trades.items():
            for trade in trades:
                if product == "VELVETFRUIT_EXTRACT":
                    # Mark 67 is highly informed (bullish when buying)
                    if trade.buyer == "Mark 67":
                        trader_data["signal_velvet"] += 1.5 * trade.quantity

                elif product == "HYDROGEL_PACK":
                    # Mark 14 is smart (follow)
                    if trade.buyer == "Mark 14":
                        trader_data["signal_hydro"] += 0.5 * trade.quantity
                    elif trade.seller == "Mark 14":
                        trader_data["signal_hydro"] -= 0.5 * trade.quantity

                    # Mark 38 is noise (fade)
                    elif trade.buyer == "Mark 38":
                        trader_data["signal_hydro"] -= 0.3 * trade.quantity
                    elif trade.seller == "Mark 38":
                        trader_data["signal_hydro"] += 0.3 * trade.quantity

        # 2. Extract mid prices
        mids = {}
        for product, depth in state.order_depths.items():
            if depth.sell_orders and depth.buy_orders:
                best_ask = min(depth.sell_orders.keys())
                best_bid = max(depth.buy_orders.keys())
                mids[product] = (best_ask + best_bid) / 2.0
            else:
                mids[product] = None

        # 3. Process Options (VELVETFRUIT_EXTRACT_VOUCHERS) and calculate Total Delta
        total_delta = 0.0
        velvet_mid = mids.get("VELVETFRUIT_EXTRACT", 5245.0)

        # Estimate Time To Expiry
        day = trader_data.get("day", 1)
        tte_days = max(0.001, 4.0 - (day - 1) - (state.timestamp / 1_000_000.0))
        T = tte_days / 252.0

        for voucher in self.vouchers:
            if voucher not in state.order_depths:
                continue
            depth = state.order_depths[voucher]
            pos = state.position.get(voucher, 0)
            limit = self.limits[voucher]
            strike = float(voucher.split("_")[1])

            # Calculate BS price and delta
            bs_price = bs_call_price(
                velvet_mid, strike, T, self.risk_free_rate, self.vol
            )
            delta = bs_call_delta(velvet_mid, strike, T, self.risk_free_rate, self.vol)

            # Track overall delta for hedging
            total_delta += pos * delta

            product_orders = []

            # Simple market taking if options are mispriced
            if depth.sell_orders:
                best_ask = min(depth.sell_orders.keys())
                ask_vol = depth.sell_orders[best_ask]
                if best_ask < bs_price - 1.5:  # Underpriced, buy
                    buy_amount = min(-ask_vol, limit - pos)
                    if buy_amount > 0:
                        product_orders.append(Order(voucher, best_ask, buy_amount))
                        pos += buy_amount
                        total_delta += buy_amount * delta  # Update delta

            if depth.buy_orders:
                best_bid = max(depth.buy_orders.keys())
                bid_vol = depth.buy_orders[best_bid]
                if best_bid > bs_price + 1.5:  # Overpriced, sell
                    sell_amount = min(bid_vol, pos + limit)
                    if sell_amount > 0:
                        product_orders.append(Order(voucher, best_bid, -sell_amount))
                        pos -= sell_amount
                        total_delta -= sell_amount * delta  # Update delta

            if product_orders:
                orders[voucher] = product_orders

        # 4. Market Making on HYDROGEL_PACK (from hydro.py)
        product = self.HYDRO
        if product in state.order_depths:
            hydro_orders = self._trade_hydro(state, state.order_depths[product], trader_data)
            if hydro_orders:
                orders[product] = hydro_orders

        # 5. Market Making & Hedging on VELVETFRUIT_EXTRACT
        product = "VELVETFRUIT_EXTRACT"
        if product in state.order_depths and mids[product]:
            depth = state.order_depths[product]
            pos = state.position.get(product, 0)
            limit = self.limits[product]
            mid = mids[product]

            # Incorporate signal
            fair = mid + (trader_data["signal_velvet"] * 0.05)

            # Adjusted inventory (Removed option delta hedge to free up limit for market making)
            hedge_target = 0
            effective_pos = pos - hedge_target

            skew = (effective_pos / limit) * 2.0

            # Spread is ~5, half spread 2.5
            ideal_bid = int(round(fair - 1.5 - skew))
            ideal_ask = int(round(fair + 1.5 - skew))

            best_ask = min(depth.sell_orders.keys())
            best_bid = max(depth.buy_orders.keys())

            # Jump the queue if profitable
            my_bid = min(ideal_bid, best_bid + 1)
            my_ask = max(ideal_ask, best_ask - 1)

            # Don't cross the book
            my_bid = min(my_bid, best_ask - 1)
            my_ask = max(my_ask, best_bid + 1)
            if my_bid >= my_ask:
                my_bid = my_ask - 1

            velvet_orders = []
            # Check if we need to cross the spread purely for hedging / strong signal
            if effective_pos > limit * 0.5 and depth.buy_orders:
                sell_amount = min(depth.buy_orders[best_bid], pos + limit)
                if sell_amount > 0:
                    velvet_orders.append(Order(product, best_bid, -sell_amount))
                    pos -= sell_amount
            elif effective_pos < -limit * 0.5 and depth.sell_orders:
                buy_amount = min(-depth.sell_orders[best_ask], limit - pos)
                if buy_amount > 0:
                    velvet_orders.append(Order(product, best_ask, buy_amount))
                    pos += buy_amount

            # Regular quoting
            if pos < limit:
                velvet_orders.append(Order(product, my_bid, limit - pos))
            if pos > -limit:
                velvet_orders.append(Order(product, my_ask, -(limit + pos)))

            if velvet_orders:
                orders[product] = velvet_orders

        # Serialize state
        next_state = json.dumps(trader_data)

        return orders, 0, next_state