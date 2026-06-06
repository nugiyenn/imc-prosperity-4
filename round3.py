import json
import math
from typing import Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, Product, TradingState


# ── Black-Scholes helpers (pure Python, no numpy) ──────────────────────
def _norm_cdf(x: float) -> float:
    """Abramowitz-Stegun approximation to the normal CDF."""
    a1, a2, a3, a4, a5 = (
        0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429,
    )
    sign = 1.0 if x >= 0.0 else -1.0
    z = abs(x)
    t = 1.0 / (1.0 + 0.2316419 * z)
    poly = (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t
    pdf = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    cdf_pos = 1.0 - pdf * poly
    return cdf_pos if sign > 0.0 else (1.0 - cdf_pos)


def _bs_call_price(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """Black-Scholes call price for a single option."""
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    sqrt_t = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def _implied_vol_bisect(
    option_price: float, spot: float, strike: float,
    tte_years: float, r: float = 0.0,
    lo: float = 1e-6, hi: float = 5.0, steps: int = 40,
) -> Optional[float]:
    """Bisection implied-vol solver for a single option."""
    if spot <= 0 or strike <= 0 or tte_years <= 0:
        return None
    intrinsic = max(spot - strike * math.exp(-r * tte_years), 0.0)
    if option_price < intrinsic - 1e-8:
        return None
    low_price = _bs_call_price(spot, strike, tte_years, lo, r)
    high_price = _bs_call_price(spot, strike, tte_years, hi, r)
    if option_price < low_price - 1e-8 or option_price > high_price + 1e-8:
        return None
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        mid_price = _bs_call_price(spot, strike, tte_years, mid, r)
        if mid_price > option_price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _fit_quadratic(xs: List[float], ys: List[float]) -> Optional[Tuple[float, float, float]]:
    """Least-squares fit y = a*x^2 + b*x + c.  Returns (a, b, c) or None."""
    n = len(xs)
    if n < 3:
        return None
    # Build normal equations for degree-2 polynomial
    s0 = float(n)
    s1 = sum(xs)
    s2 = sum(x * x for x in xs)
    s3 = sum(x * x * x for x in xs)
    s4 = sum(x * x * x * x for x in xs)
    t0 = sum(ys)
    t1 = sum(x * y for x, y in zip(xs, ys))
    t2 = sum(x * x * y for x, y in zip(xs, ys))
    # Solve 3×3 system via Cramer's rule
    # | s4 s3 s2 |   | a |   | t2 |
    # | s3 s2 s1 | × | b | = | t1 |
    # | s2 s1 s0 |   | c |   | t0 |
    det = (s4 * (s2 * s0 - s1 * s1)
           - s3 * (s3 * s0 - s1 * s2)
           + s2 * (s3 * s1 - s2 * s2))
    if abs(det) < 1e-30:
        return None
    a = (t2 * (s2 * s0 - s1 * s1)
         - s3 * (t1 * s0 - s1 * t0)
         + s2 * (t1 * s1 - s2 * t0)) / det
    b = (s4 * (t1 * s0 - s1 * t0)
         - t2 * (s3 * s0 - s1 * s2)
         + s2 * (s3 * t0 - t1 * s2)) / det
    c = (s4 * (s2 * t0 - t1 * s1)
         - s3 * (s3 * t0 - t1 * s2)
         + t2 * (s3 * s1 - s2 * s2)) / det
    return (a, b, c)


class Trader:
    MARKET_ACCESS_BID = 20

    HYDRO = "HYDROGEL_PACK"
    HYDRO_LIMIT = 200

    VELVET = "VELVETFRUIT_EXTRACT"
    VELVET_LIMIT = 200

    # --- Hydrogel parameters ---
    EMA_ALPHA = 0.015
    DEV_DEADBAND = 0.5
    DEV_TO_TARGET = 10.0
    SOFT_POS = 190
    INV_PENALTY = 0.10

    TAKER_DEV = 9.0
    TAKER_CLIP = 28

    BASE_SIZE = 34
    MAX_ADD_SIZE = 52

    # --- Velvetfruit Extract parameters ---
    VE_EMA_ALPHA = 0.008
    VE_ANCHOR_ALPHA = 0.003
    VE_DEV_DEADBAND = 1.5
    VE_DEV_TO_TARGET = 8.0
    VE_SOFT_POS = 150
    VE_INV_PENALTY = 0.12
    VE_TAKER_DEV = 12.0
    VE_TAKER_CLIP = 20
    VE_BASE_SIZE = 15
    VE_MAX_ADD_SIZE = 30
    VE_DIRECTIONAL_DEV = 1.0

    # --- Voucher (VEV) option mean-reversion parameters ---
    VEV_LIMIT = 300
    VEV_TTE_START_DAYS = 5.0        # Round 3 starts at TTE=5 days
    VEV_TIMESTAMPS_PER_DAY = 999900  # last timestamp in a day
    VEV_R = 0.0                     # risk-free rate

    # Strikes we actively trade
    VEV_ATM_STRIKES = [5000, 5100, 5200, 5300, 5400, 5500]
    VEV_ITM_STRIKES = [4000, 4500]
    VEV_ALL_STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500]

    # EMA alpha for per-strike Implied Volatility
    VEV_IV_EMA_ALPHA = 0.02
    VEV_DEV_DEADBAND = 1.0
    VEV_DEV_TO_TARGET = 10.0
    VEV_INV_PENALTY = 0.10
    VEV_SOFT_POS = 250

    # Taker thresholds
    VEV_TAKER_DEV = 5.0
    VEV_TAKER_CLIP = 25

    # Passive quoting sizes
    VEV_ATM_BASE_SIZE = 25
    VEV_ATM_MAX_ADD = 40
    VEV_ITM_BASE_SIZE = 8
    VEV_ITM_MAX_ADD = 12

    def bid(self) -> int:
        return self.MARKET_ACCESS_BID

    @staticmethod
    def _load_state(trader_data: str) -> Dict:
        if not trader_data:
            return {}
        try:
            loaded = json.loads(trader_data)
            return loaded if isinstance(loaded, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _dump_state(data: Dict) -> str:
        try:
            return json.dumps(data, separators=(",", ":"), sort_keys=True)
        except Exception:
            return ""

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

        dev = mid - ema_new
        dev_eff = 0.0 if abs(dev) < self.DEV_DEADBAND else dev

        # Mean-reversion target: long when below mean, short when above mean.
        raw_target = -self.DEV_TO_TARGET * dev_eff - self.INV_PENALTY * position
        target = int(round(self._clamp(raw_target, -self.SOFT_POS, self.SOFT_POS)))

        buy_need = max(0, target - position)
        sell_need = max(0, position - target)

        orders: List[Order] = []
        projected = position

        # Aggressive capture on large dislocations from mean.
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

        fair = ema_new - self.INV_PENALTY * projected
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

    def _trade_velvet(self, state: TradingState, depth: OrderDepth, persistent: Dict) -> List[Order]:
        best_bid, best_ask = self._best_bid_ask(depth)
        if best_bid is None or best_ask is None:
            return []

        mid = 0.5 * (best_bid + best_ask)
        spread = best_ask - best_bid

        velvet_state = persistent.get("velvet", {})
        if not isinstance(velvet_state, dict):
            velvet_state = {}

        # --- Dual EMA tracking ---
        # Fast signal EMA: used for position targets and taker decisions.
        ema_old = velvet_state.get("ema")
        if not isinstance(ema_old, (int, float)):
            ema_old = mid
        ema_new = (1.0 - self.VE_EMA_ALPHA) * float(ema_old) + self.VE_EMA_ALPHA * mid

        # Slow anchor EMA: captures session-level fair value.
        # Barely moves during swings, so a recovery from a deep trough
        # stays in "buy zone" until price actually exceeds session average.
        anchor_old = velvet_state.get("anchor")
        if not isinstance(anchor_old, (int, float)):
            anchor_old = mid
        anchor_new = (1.0 - self.VE_ANCHOR_ALPHA) * float(anchor_old) + self.VE_ANCHOR_ALPHA * mid

        velvet_state["ema"] = ema_new
        velvet_state["anchor"] = anchor_new
        persistent["velvet"] = velvet_state

        position = int(state.position.get(self.VELVET, 0))
        rem_buy, rem_sell = self._remaining_capacity(position, self.VELVET_LIMIT)

        # Signal deviation (fast EMA) — drives target position and taker.
        dev = mid - ema_new
        dev_eff = 0.0 if abs(dev) < self.VE_DEV_DEADBAND else dev

        # Anchor deviation (slow EMA) — drives directional filter.
        # This tells us whether price is above or below the SESSION mean.
        anchor_dev = mid - anchor_new

        # Mean-reversion target: buy below signal EMA, sell above.
        raw_target = -self.VE_DEV_TO_TARGET * dev_eff - self.VE_INV_PENALTY * position
        target = int(round(self._clamp(raw_target, -self.VE_SOFT_POS, self.VE_SOFT_POS)))

        buy_need = max(0, target - position)
        sell_need = max(0, position - target)

        orders: List[Order] = []
        projected = position

        # Aggressive taker ONLY on very large dislocations from signal EMA.
        if dev <= -self.VE_TAKER_DEV and rem_buy > 0:
            avail = max(0, -int(depth.sell_orders.get(best_ask, 0)))
            qty = min(rem_buy, avail, self.VE_TAKER_CLIP)
            if qty > 0:
                orders.append(Order(self.VELVET, int(best_ask), int(qty)))
                rem_buy -= qty
                rem_sell += qty
                projected += qty
        elif dev >= self.VE_TAKER_DEV and rem_sell > 0:
            avail = max(0, int(depth.buy_orders.get(best_bid, 0)))
            qty = min(rem_sell, avail, self.VE_TAKER_CLIP)
            if qty > 0:
                orders.append(Order(self.VELVET, int(best_bid), -int(qty)))
                rem_sell -= qty
                rem_buy += qty
                projected -= qty

        # Passive quoting — directional based on ANCHOR deviation.
        fair = ema_new - self.VE_INV_PENALTY * projected
        inside = 1 if spread >= 2 else 0
        skew = int(self._clamp(round((fair - mid) / 2.0), -3, 3))

        bid_quote = int(best_bid + inside + skew)
        ask_quote = int(best_ask - inside + skew)
        if bid_quote >= ask_quote:
            bid_quote = int(best_bid)
            ask_quote = int(best_ask)
        if bid_quote >= ask_quote:
            ask_quote = bid_quote + 1

        buy_size = min(rem_buy, self.VE_BASE_SIZE + min(self.VE_MAX_ADD_SIZE, buy_need // 2))
        sell_size = min(rem_sell, self.VE_BASE_SIZE + min(self.VE_MAX_ADD_SIZE, sell_need // 2))

        # DIRECTIONAL FILTER using slow anchor EMA.
        # Price above session mean → only sell (mean revert down)
        # Price below session mean → only buy (mean revert up)
        # This prevents selling during recoveries from troughs and
        # buying during pullbacks from peaks.
        if anchor_dev > self.VE_DIRECTIONAL_DEV:
            buy_size = 0
        elif anchor_dev < -self.VE_DIRECTIONAL_DEV:
            sell_size = 0
        # Near session mean: allow tiny two-sided quoting.
        else:
            buy_size = min(buy_size, 5)
            sell_size = min(sell_size, 5)

        if projected >= self.VE_SOFT_POS:
            buy_size = 0
        if projected <= -self.VE_SOFT_POS:
            sell_size = 0

        if buy_size > 0:
            orders.append(Order(self.VELVET, bid_quote, int(buy_size)))
        if sell_size > 0:
            orders.append(Order(self.VELVET, ask_quote, -int(sell_size)))

        return orders

    # ── Voucher (VEV) option mean-reversion logic ──────────────────────
    # Uses an EMA of Implied Volatility to establish fair value.
    # By tracking IV rather than option price directly, we account for
    # changes in the underlying spot price (delta).

    def _get_spot_mid(self, state: TradingState) -> Optional[float]:
        depth = state.order_depths.get(self.VELVET)
        if depth is None:
            return None
        bb, ba = self._best_bid_ask(depth)
        if bb is None or ba is None:
            return None
        return 0.5 * (bb + ba)

    def _get_tte_years(self, timestamp: int) -> float:
        frac = timestamp / self.VEV_TIMESTAMPS_PER_DAY
        tte_days = max(self.VEV_TTE_START_DAYS - frac, 0.0001)
        return tte_days / 365.0

    @staticmethod
    def _vev_symbol(strike: int) -> str:
        return f"VEV_{strike}"

    def _trade_vouchers(
        self, state: TradingState, persistent: Dict,
    ) -> Dict[str, List[Order]]:
        """Per-strike IV-based mean-reversion for VEV options."""
        orders_by_symbol: Dict[str, List[Order]] = {}

        spot = self._get_spot_mid(state)
        if spot is None or spot <= 0:
            return orders_by_symbol

        tte = self._get_tte_years(state.timestamp)

        vev_state = persistent.get("vev", {})
        if not isinstance(vev_state, dict):
            vev_state = {}

        for strike in self.VEV_ALL_STRIKES:
            sym = self._vev_symbol(strike)
            depth = state.order_depths.get(sym)
            if depth is None:
                continue
            bb, ba = self._best_bid_ask(depth)
            if bb is None or ba is None:
                continue

            mid = 0.5 * (bb + ba)
            spread = ba - bb

            # Compute current implied volatility
            current_iv = _implied_vol_bisect(mid, spot, float(strike), tte)
            
            # --- Per-strike EMA of IV ---
            ema_key = f"ema_iv_{strike}"
            ema_old = vev_state.get(ema_key)
            
            # Initialization
            if not isinstance(ema_old, (int, float)):
                if current_iv is not None and 0.01 <= current_iv <= 2.0:
                    ema_new = current_iv
                else:
                    ema_new = 0.20  # Fallback guess
            else:
                if current_iv is not None and 0.01 <= current_iv <= 2.0:
                    ema_new = (1.0 - self.VEV_IV_EMA_ALPHA) * float(ema_old) + self.VEV_IV_EMA_ALPHA * current_iv
                else:
                    ema_new = float(ema_old)
            
            vev_state[ema_key] = ema_new

            # Compute theoretical fair price using EMA IV and current spot
            fair_theo = _bs_call_price(spot, float(strike), tte, ema_new)

            position = int(state.position.get(sym, 0))
            rem_buy, rem_sell = self._remaining_capacity(position, self.VEV_LIMIT)

            # Deviation from fair theo price (positive = option market overpriced)
            dev = mid - fair_theo
            dev_eff = 0.0 if abs(dev) < self.VEV_DEV_DEADBAND else dev

            is_atm = strike in self.VEV_ATM_STRIKES
            base_size = self.VEV_ATM_BASE_SIZE if is_atm else self.VEV_ITM_BASE_SIZE
            max_add = self.VEV_ATM_MAX_ADD if is_atm else self.VEV_ITM_MAX_ADD

            # Mean-reversion target: buy when underpriced, sell when overpriced
            raw_target = -self.VEV_DEV_TO_TARGET * dev_eff - self.VEV_INV_PENALTY * position
            target = int(round(self._clamp(raw_target, -self.VEV_SOFT_POS, self.VEV_SOFT_POS)))

            buy_need = max(0, target - position)
            sell_need = max(0, position - target)

            orders: List[Order] = []
            projected = position

            # --- Taker: cross spread on large deviations ---
            if dev <= -self.VEV_TAKER_DEV and rem_buy > 0:
                avail = max(0, -int(depth.sell_orders.get(ba, 0)))
                qty = min(rem_buy, avail, self.VEV_TAKER_CLIP)
                if qty > 0:
                    orders.append(Order(sym, int(ba), int(qty)))
                    rem_buy -= qty
                    rem_sell += qty
                    projected += qty
            elif dev >= self.VEV_TAKER_DEV and rem_sell > 0:
                avail = max(0, int(depth.buy_orders.get(bb, 0)))
                qty = min(rem_sell, avail, self.VEV_TAKER_CLIP)
                if qty > 0:
                    orders.append(Order(sym, int(bb), -int(qty)))
                    rem_sell -= qty
                    rem_buy += qty
                    projected -= qty

            # --- Directional passive quoting ---
            fair_adjusted = fair_theo - self.VEV_INV_PENALTY * projected
            inside = 1 if spread >= 2 else 0
            skew = int(self._clamp(round((fair_adjusted - mid) / 2.0), -3, 3))

            bid_q = int(bb + inside + skew)
            ask_q = int(ba - inside + skew)
            if bid_q >= ask_q:
                bid_q = int(bb)
                ask_q = int(ba)
            if bid_q >= ask_q:
                ask_q = bid_q + 1

            buy_size = min(rem_buy, base_size + min(max_add, buy_need // 2))
            sell_size = min(rem_sell, base_size + min(max_add, sell_need // 2))

            # DIRECTIONAL FILTER
            if dev > self.VEV_DEV_DEADBAND:
                buy_size = 0
            elif dev < -self.VEV_DEV_DEADBAND:
                sell_size = 0
            else:
                buy_size = min(buy_size, 3)
                sell_size = min(sell_size, 3)

            if projected >= self.VEV_SOFT_POS:
                buy_size = 0
            if projected <= -self.VEV_SOFT_POS:
                sell_size = 0

            if buy_size > 0:
                orders.append(Order(sym, bid_q, int(buy_size)))
            if sell_size > 0:
                orders.append(Order(sym, ask_q, -int(sell_size)))

            if orders:
                orders_by_symbol[sym] = orders

        persistent["vev"] = vev_state
        return orders_by_symbol

    # ── Main entry point ──────────────────────────────────────────────

    def run(self, state: TradingState):
        persistent = self._load_state(state.traderData)
        if not isinstance(persistent, dict):
            persistent = {}

        result: Dict[Product, List[Order]] = {}
        for product in state.order_depths:
            if product == self.HYDRO:
                result[product] = self._trade_hydro(state, state.order_depths[product], persistent)
            elif product == self.VELVET:
                result[product] = self._trade_velvet(state, state.order_depths[product], persistent)
            # VEV products are handled in bulk below

        # Trade all vouchers in one pass (needs cross-strike smile fit)
        voucher_orders = self._trade_vouchers(state, persistent)
        result.update(voucher_orders)

        # Ensure every product has an entry in result
        for product in state.order_depths:
            if product not in result:
                result[product] = []

        trader_data = self._dump_state(persistent)
        return result, 0, trader_data