import json
import math
from typing import Dict, List, Optional

from datamodel import OrderDepth, TradingState, Order

LIMIT = 10  # All products have position limit of 10

# --- Product groups (Round 5) ---
GALAXY_SOUNDS = [
    "GALAXY_SOUNDS_DARK_MATTER",
    "GALAXY_SOUNDS_BLACK_HOLES",
    "GALAXY_SOUNDS_PLANETARY_RINGS",
    "GALAXY_SOUNDS_SOLAR_WINDS",
    "GALAXY_SOUNDS_SOLAR_FLAMES",
]
SLEEP_POD = [
    "SLEEP_POD_SUEDE",
    "SLEEP_POD_LAMB_WOOL",
    "SLEEP_POD_POLYESTER",
    "SLEEP_POD_NYLON",
    "SLEEP_POD_COTTON",
]
MICROCHIP = [
    "MICROCHIP_CIRCLE",
    "MICROCHIP_OVAL",
    "MICROCHIP_SQUARE",
    "MICROCHIP_RECTANGLE",
    "MICROCHIP_TRIANGLE",
]
PEBBLES = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"]
ROBOT = [
    "ROBOT_VACUUMING",
    "ROBOT_MOPPING",
    "ROBOT_DISHES",
    "ROBOT_LAUNDRY",
    "ROBOT_IRONING",
]
UV_VISOR = ["UV_VISOR_YELLOW", "UV_VISOR_AMBER", "UV_VISOR_ORANGE", "UV_VISOR_RED", "UV_VISOR_MAGENTA"]
TRANSLATOR = [
    "TRANSLATOR_SPACE_GRAY",
    "TRANSLATOR_ASTRO_BLACK",
    "TRANSLATOR_ECLIPSE_CHARCOAL",
    "TRANSLATOR_GRAPHITE_MIST",
    "TRANSLATOR_VOID_BLUE",
]
PANEL = ["PANEL_1X2", "PANEL_2X2", "PANEL_1X4", "PANEL_2X4", "PANEL_4X4"]
OXYGEN_SHAKE = [
    "OXYGEN_SHAKE_MORNING_BREATH",
    "OXYGEN_SHAKE_EVENING_BREATH",
    "OXYGEN_SHAKE_MINT",
    "OXYGEN_SHAKE_CHOCOLATE",
    "OXYGEN_SHAKE_GARLIC",
]
SNACKPACK = [
    "SNACKPACK_CHOCOLATE",
    "SNACKPACK_VANILLA",
    "SNACKPACK_PISTACHIO",
    "SNACKPACK_STRAWBERRY",
    "SNACKPACK_RASPBERRY",
]

ALL_PRODUCTS = GALAXY_SOUNDS + SLEEP_POD + MICROCHIP + PEBBLES + ROBOT + UV_VISOR + TRANSLATOR + PANEL + OXYGEN_SHAKE + SNACKPACK

# --- Strategy pairs from CONTEXT_ROUND_5.md ---
# Negatively correlated pairs: trade the SUM (long both / short both)
NEG_CORR_PAIRS = [
    ("SNACKPACK_CHOCOLATE", "SNACKPACK_VANILLA"),
    ("MICROCHIP_SQUARE", "MICROCHIP_RECTANGLE"),
    ("TRANSLATOR_ASTRO_BLACK", "TRANSLATOR_VOID_BLUE"),
]

# Positively correlated pairs: trade the SPREAD (A - B)
POS_CORR_PAIRS = [
    ("ROBOT_VACUUMING", "ROBOT_LAUNDRY"),
]


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def best_bid_ask(depth: OrderDepth):
    bb = max(depth.buy_orders) if depth.buy_orders else None
    ba = min(depth.sell_orders) if depth.sell_orders else None
    return bb, ba


def mid_price(depth: OrderDepth):
    bb, ba = best_bid_ask(depth)
    if bb is not None and ba is not None:
        return (bb + ba) / 2.0
    return None


def spread_ticks(depth: OrderDepth) -> Optional[int]:
    bb, ba = best_bid_ask(depth)
    if bb is None or ba is None:
        return None
    return int(ba - bb)


def remaining_capacity(pos: int, limit: int = LIMIT) -> tuple[int, int]:
    # remaining buy, remaining sell
    return max(0, limit - pos), max(0, limit + pos)


def price_for(side: str, bb: int, ba: int, *, aggressive: bool) -> int:
    """Return a limit price for buy/sell.

    aggressive=True crosses the spread (buy@ask, sell@bid).
    aggressive=False improves inside the spread when possible.
    """
    if side == "buy":
        if aggressive:
            return int(ba)
        return int(bb + 1) if bb + 1 < ba else int(bb)
    else:
        if aggressive:
            return int(bb)
        return int(ba - 1) if ba - 1 > bb else int(ba)


class Trader:
    # EMA parameters
    EMA_ALPHA_SLOW = 0.008
    EMA_ALPHA_FAST = 0.03
    EMA_ALPHA_MM = 0.02

    # Pair trading thresholds
    PAIR_ENTRY_Z = 1.6
    PAIR_EXIT_Z = 0.4
    PAIR_EXTREME_Z = 2.4  # allow aggressive crossing when signal is large

    # Execution gates
    ENTRY_SPREAD_GATE = 1.3  # require deviation > gate * (half-spread costs)

    # Sizes
    PAIR_CLIP = 3
    TREND_CLIP = 2
    MM_CLIP = 4

    # Market making
    MM_INV_PENALTY = 0.15
    MM_MAX_SPREAD = 6  # do not quote in very wide markets

    def run(self, state: TradingState) -> tuple[dict[str, list[Order]], int, str]:
        orders: Dict[str, List[Order]] = {}

        # --- Restore persistent state ---
        td: dict = {}
        if state.traderData:
            try:
                td = json.loads(state.traderData)
            except Exception:
                td = {}

        # --- Compute mids for available products ---
        mids: dict[str, float] = {}
        for product, depth in state.order_depths.items():
            m = mid_price(depth)
            if m is not None:
                mids[product] = m

        # --- Cross-group hedge signal (Robots vs Sleep Pods) ---
        robot_net_pos = sum(state.position.get(p, 0) for p in ROBOT)
        # If long robots, bias to short sleep pods (lower their fair); vice versa.
        cross_hedge_ticks = clamp(-0.3 * robot_net_pos, -2.0, 2.0)

        # Track which products are handled by non-MM modules.
        reserved: set[str] = set()

        # --- 1) Negatively correlated pairs: trade SUM ---
        for a, b in NEG_CORR_PAIRS:
            if a not in mids or b not in mids:
                continue
            if a not in state.order_depths or b not in state.order_depths:
                continue

            depth_a = state.order_depths[a]
            depth_b = state.order_depths[b]
            bb_a, ba_a = best_bid_ask(depth_a)
            bb_b, ba_b = best_bid_ask(depth_b)
            if bb_a is None or ba_a is None or bb_b is None or ba_b is None:
                continue

            sp_a = ba_a - bb_a
            sp_b = ba_b - bb_b

            key = f"nc_{a}_{b}"
            st = td.get(key, {})

            current = mids[a] + mids[b]
            is_stationary = (a, b) == ("SNACKPACK_CHOCOLATE", "SNACKPACK_VANILLA")
            alpha = self.EMA_ALPHA_SLOW if is_stationary else self.EMA_ALPHA_FAST

            ema = st.get("ema", current)
            ema = (1 - alpha) * ema + alpha * current
            var = st.get("var", 50.0**2)
            var = (1 - alpha) * var + alpha * (current - ema) ** 2
            std = max(math.sqrt(var), 10.0)
            st["ema"] = ema
            st["var"] = var
            td[key] = st

            dev = current - ema
            z = dev / std

            # Spread-aware entry gate
            half_cost = 0.5 * (sp_a + sp_b)
            if abs(z) >= self.PAIR_ENTRY_Z and abs(dev) < self.ENTRY_SPREAD_GATE * half_cost:
                reserved.update([a, b])
                continue

            pos_a = state.position.get(a, 0)
            pos_b = state.position.get(b, 0)
            buy_cap_a, sell_cap_a = remaining_capacity(pos_a)
            buy_cap_b, sell_cap_b = remaining_capacity(pos_b)

            aggressive = abs(z) >= self.PAIR_EXTREME_Z or (sp_a <= 2 and sp_b <= 2)

            if z < -self.PAIR_ENTRY_Z:
                qty_a = min(buy_cap_a, self.PAIR_CLIP)
                qty_b = min(buy_cap_b, self.PAIR_CLIP)
                if qty_a > 0:
                    orders.setdefault(a, []).append(Order(a, price_for("buy", bb_a, ba_a, aggressive=aggressive), qty_a))
                if qty_b > 0:
                    orders.setdefault(b, []).append(Order(b, price_for("buy", bb_b, ba_b, aggressive=aggressive), qty_b))
            elif z > self.PAIR_ENTRY_Z:
                qty_a = min(sell_cap_a, self.PAIR_CLIP)
                qty_b = min(sell_cap_b, self.PAIR_CLIP)
                if qty_a > 0:
                    orders.setdefault(a, []).append(Order(a, price_for("sell", bb_a, ba_a, aggressive=aggressive), -qty_a))
                if qty_b > 0:
                    orders.setdefault(b, []).append(Order(b, price_for("sell", bb_b, ba_b, aggressive=aggressive), -qty_b))
            elif abs(z) < self.PAIR_EXIT_Z:
                # Prefer not to pay huge spreads on exits.
                exit_aggr = (sp_a <= 2 and sp_b <= 2)
                if pos_a > 0:
                    orders.setdefault(a, []).append(Order(a, price_for("sell", bb_a, ba_a, aggressive=exit_aggr), -pos_a))
                elif pos_a < 0:
                    orders.setdefault(a, []).append(Order(a, price_for("buy", bb_a, ba_a, aggressive=exit_aggr), -pos_a))
                if pos_b > 0:
                    orders.setdefault(b, []).append(Order(b, price_for("sell", bb_b, ba_b, aggressive=exit_aggr), -pos_b))
                elif pos_b < 0:
                    orders.setdefault(b, []).append(Order(b, price_for("buy", bb_b, ba_b, aggressive=exit_aggr), -pos_b))

            reserved.update([a, b])

        # --- 2) Positively correlated pairs: trade SPREAD (A - B) ---
        for a, b in POS_CORR_PAIRS:
            if a not in mids or b not in mids:
                continue
            if a not in state.order_depths or b not in state.order_depths:
                continue

            depth_a = state.order_depths[a]
            depth_b = state.order_depths[b]
            bb_a, ba_a = best_bid_ask(depth_a)
            bb_b, ba_b = best_bid_ask(depth_b)
            if bb_a is None or ba_a is None or bb_b is None or ba_b is None:
                continue

            sp_a = ba_a - bb_a
            sp_b = ba_b - bb_b

            key = f"pc_{a}_{b}"
            st = td.get(key, {})

            current = mids[a] - mids[b]
            alpha = self.EMA_ALPHA_SLOW

            ema = st.get("ema", current)
            ema = (1 - alpha) * ema + alpha * current
            var = st.get("var", 200.0**2)
            var = (1 - alpha) * var + alpha * (current - ema) ** 2
            std = max(math.sqrt(var), 12.0)
            st["ema"] = ema
            st["var"] = var
            td[key] = st

            dev = current - ema
            z = dev / std

            half_cost = 0.5 * (sp_a + sp_b)
            if abs(z) >= self.PAIR_ENTRY_Z and abs(dev) < self.ENTRY_SPREAD_GATE * half_cost:
                reserved.update([a, b])
                continue

            pos_a = state.position.get(a, 0)
            pos_b = state.position.get(b, 0)
            buy_cap_a, sell_cap_a = remaining_capacity(pos_a)
            buy_cap_b, sell_cap_b = remaining_capacity(pos_b)

            aggressive = abs(z) >= self.PAIR_EXTREME_Z or (sp_a <= 2 and sp_b <= 2)

            if z > self.PAIR_ENTRY_Z:
                # Spread too wide -> sell A, buy B
                qty_a = min(sell_cap_a, self.PAIR_CLIP)
                qty_b = min(buy_cap_b, self.PAIR_CLIP)
                if qty_a > 0:
                    orders.setdefault(a, []).append(Order(a, price_for("sell", bb_a, ba_a, aggressive=aggressive), -qty_a))
                if qty_b > 0:
                    orders.setdefault(b, []).append(Order(b, price_for("buy", bb_b, ba_b, aggressive=aggressive), qty_b))
            elif z < -self.PAIR_ENTRY_Z:
                # Spread too narrow -> buy A, sell B
                qty_a = min(buy_cap_a, self.PAIR_CLIP)
                qty_b = min(sell_cap_b, self.PAIR_CLIP)
                if qty_a > 0:
                    orders.setdefault(a, []).append(Order(a, price_for("buy", bb_a, ba_a, aggressive=aggressive), qty_a))
                if qty_b > 0:
                    orders.setdefault(b, []).append(Order(b, price_for("sell", bb_b, ba_b, aggressive=aggressive), -qty_b))
            elif abs(z) < self.PAIR_EXIT_Z:
                exit_aggr = (sp_a <= 2 and sp_b <= 2)
                if pos_a > 0:
                    orders.setdefault(a, []).append(Order(a, price_for("sell", bb_a, ba_a, aggressive=exit_aggr), -pos_a))
                elif pos_a < 0:
                    orders.setdefault(a, []).append(Order(a, price_for("buy", bb_a, ba_a, aggressive=exit_aggr), -pos_a))
                if pos_b > 0:
                    orders.setdefault(b, []).append(Order(b, price_for("sell", bb_b, ba_b, aggressive=exit_aggr), -pos_b))
                elif pos_b < 0:
                    orders.setdefault(b, []).append(Order(b, price_for("buy", bb_b, ba_b, aggressive=exit_aggr), -pos_b))

            reserved.update([a, b])

        # --- 3) Pebbles relative value: XS vs M (bounded) ---
        # Implements: if XS moves above M/L, consider short XS / long M/L.
        if "PEBBLES_XS" in mids and "PEBBLES_L" in mids and "PEBBLES_XS" in state.order_depths and "PEBBLES_L" in state.order_depths:
            # Backtest showed PEBBLES_M is consistently loss-making; use L as the reference leg instead.
            a, b = "PEBBLES_XS", "PEBBLES_L"
            depth_a = state.order_depths[a]
            depth_b = state.order_depths[b]
            bb_a, ba_a = best_bid_ask(depth_a)
            bb_b, ba_b = best_bid_ask(depth_b)
            if bb_a is not None and ba_a is not None and bb_b is not None and ba_b is not None:
                sp_a = ba_a - bb_a
                sp_b = ba_b - bb_b
                key = "rv_pebbles_xs_l"
                st = td.get(key, {})

                current = mids[a] - mids[b]
                alpha = self.EMA_ALPHA_FAST

                ema = st.get("ema", current)
                ema = (1 - alpha) * ema + alpha * current
                var = st.get("var", 50.0**2)
                var = (1 - alpha) * var + alpha * (current - ema) ** 2
                std = max(math.sqrt(var), 8.0)
                st["ema"] = ema
                st["var"] = var
                td[key] = st

                dev = current - ema
                z = dev / std

                pos_a = state.position.get(a, 0)
                pos_b = state.position.get(b, 0)
                buy_cap_a, sell_cap_a = remaining_capacity(pos_a)
                buy_cap_b, sell_cap_b = remaining_capacity(pos_b)

                half_cost = 0.5 * (sp_a + sp_b)
                if abs(z) >= 1.5 and abs(dev) >= 1.0 * half_cost:
                    aggressive = abs(z) >= 2.4 or (sp_a <= 2 and sp_b <= 2)
                    clip = 2
                    if z > 1.5:
                        # XS too rich vs M
                        q_a = min(sell_cap_a, clip)
                        q_b = min(buy_cap_b, clip)
                        if q_a > 0:
                            orders.setdefault(a, []).append(Order(a, price_for("sell", bb_a, ba_a, aggressive=aggressive), -q_a))
                        if q_b > 0:
                            orders.setdefault(b, []).append(Order(b, price_for("buy", bb_b, ba_b, aggressive=aggressive), q_b))
                    elif z < -1.5:
                        # XS too cheap vs M
                        q_a = min(buy_cap_a, clip)
                        q_b = min(sell_cap_b, clip)
                        if q_a > 0:
                            orders.setdefault(a, []).append(Order(a, price_for("buy", bb_a, ba_a, aggressive=aggressive), q_a))
                        if q_b > 0:
                            orders.setdefault(b, []).append(Order(b, price_for("sell", bb_b, ba_b, aggressive=aggressive), -q_b))
                    reserved.update([a, b])

        # --- 4) Trend overlays (bounded) ---
        # Pebbles: XS down, XL up
        # Snackpacks: Strawberry up, Pistachio down
        # Galaxy Sounds: Black Holes trending up
        trend_list = [
            ("PEBBLES_XS", -1),
            ("PEBBLES_XL", 1),
            ("SNACKPACK_STRAWBERRY", 1),
            ("SNACKPACK_PISTACHIO", -1),
            ("GALAXY_SOUNDS_BLACK_HOLES", 1),
            ("ROBOT_MOPPING", 1),
            ("ROBOT_IRONING", -1),
        ]
        for product, direction in trend_list:
            if product not in mids or product not in state.order_depths:
                continue
            depth = state.order_depths[product]
            bb, ba = best_bid_ask(depth)
            if bb is None or ba is None:
                continue

            key = f"trend_{product}"
            st = td.get(key, {})

            m = mids[product]
            alpha = self.EMA_ALPHA_FAST
            ema = st.get("ema", m)
            ema = (1 - alpha) * ema + alpha * m
            var = st.get("var", 100.0**2)
            var = (1 - alpha) * var + alpha * (m - ema) ** 2
            std = max(math.sqrt(var), 10.0)
            st["ema"] = ema
            st["var"] = var
            td[key] = st

            pos = state.position.get(product, 0)
            buy_cap, sell_cap = remaining_capacity(pos)
            sp = ba - bb
            aggressive = sp <= 2

            if direction == 1:
                # Uptrend: buy dips, take profit when very extended.
                if m < ema - 0.5 * std and buy_cap > 0:
                    q = min(buy_cap, self.TREND_CLIP)
                    orders.setdefault(product, []).append(Order(product, price_for("buy", bb, ba, aggressive=aggressive), q))
                elif m > ema + 2.0 * std and pos > 0:
                    orders.setdefault(product, []).append(Order(product, price_for("sell", bb, ba, aggressive=aggressive), -pos))
            else:
                # Downtrend: sell spikes, cover when very extended.
                if m > ema + 0.5 * std and sell_cap > 0:
                    q = min(sell_cap, self.TREND_CLIP)
                    orders.setdefault(product, []).append(Order(product, price_for("sell", bb, ba, aggressive=aggressive), -q))
                elif m < ema - 2.0 * std and pos < 0:
                    orders.setdefault(product, []).append(Order(product, price_for("buy", bb, ba, aggressive=aggressive), -pos))

            reserved.add(product)

        # --- 5) Market making (selective universe + cross-pricing + hedging) ---
        # Universe per context: Galaxy Sounds (MM), Panels (MM with cross-pricing), Oxygen Shakes (MM w/ cross-pricing),
        # Translators except the traded pair (MM), plus non-pair Sleep Pods to implement cross-group hedging.
        sleep_pod_hedge_names = ["SLEEP_POD_SUEDE", "SLEEP_POD_LAMB_WOOL", "SLEEP_POD_NYLON"]

        # Prune consistently loss-making MM names based on backtest.md aggregation.
        panel_mm = [p for p in PANEL if p not in {"PANEL_2X2", "PANEL_4X4"}]
        oxygen_mm = [p for p in OXYGEN_SHAKE if p not in {"OXYGEN_SHAKE_MORNING_BREATH"}]
        visor_mm = [p for p in UV_VISOR if p not in {"UV_VISOR_AMBER"}]

        mm_candidates = set(GALAXY_SOUNDS + panel_mm + oxygen_mm + TRANSLATOR + sleep_pod_hedge_names + visor_mm)
        # Exclude anything already handled by pairs/trends/RV.
        mm_products = [p for p in mm_candidates if p in mids and p in state.order_depths and p not in reserved]

        # Panel cross-pricing adjustment for PANEL_1X4 based on deviations of PANEL_2X2 (positive) and PANEL_2X4 (negative).
        panel_adj_1x4 = 0
        # PANEL_2X2 was consistently negative in the provided backtest; drop it from the signal.
        if "PANEL_1X4" in mids and "PANEL_2X4" in mids:
            for p in ("PANEL_1X4", "PANEL_2X4"):
                k = f"mm_{p}"
                st = td.get(k, {})
                ema = st.get("ema", mids[p])
                ema = (1 - self.EMA_ALPHA_MM) * ema + self.EMA_ALPHA_MM * mids[p]
                st["ema"] = ema
                td[k] = st

            dev_2x4 = mids["PANEL_2X4"] - td["mm_PANEL_2X4"]["ema"]
            panel_adj_1x4 = int(round(clamp(-0.25 * dev_2x4, -2.0, 2.0)))

        # Oxygen shake cross-pricing: chocolate and garlic move together.
        oxy_adj: dict[str, int] = {}
        if "OXYGEN_SHAKE_CHOCOLATE" in mids and "OXYGEN_SHAKE_GARLIC" in mids:
            for p in ("OXYGEN_SHAKE_CHOCOLATE", "OXYGEN_SHAKE_GARLIC"):
                k = f"mm_{p}"
                st = td.get(k, {})
                ema = st.get("ema", mids[p])
                ema = (1 - self.EMA_ALPHA_MM) * ema + self.EMA_ALPHA_MM * mids[p]
                st["ema"] = ema
                td[k] = st
            dev_choc = mids["OXYGEN_SHAKE_CHOCOLATE"] - td["mm_OXYGEN_SHAKE_CHOCOLATE"]["ema"]
            dev_gar = mids["OXYGEN_SHAKE_GARLIC"] - td["mm_OXYGEN_SHAKE_GARLIC"]["ema"]
            oxy_adj["OXYGEN_SHAKE_CHOCOLATE"] = int(round(clamp(0.15 * dev_gar, -2.0, 2.0)))
            oxy_adj["OXYGEN_SHAKE_GARLIC"] = int(round(clamp(0.15 * dev_choc, -2.0, 2.0)))

        for product in mm_products:
            depth = state.order_depths[product]
            bb, ba = best_bid_ask(depth)
            if bb is None or ba is None:
                continue

            sp = ba - bb
            if sp <= 0 or sp > self.MM_MAX_SPREAD:
                continue

            m = mids[product]
            key = f"mm_{product}"
            st = td.get(key, {})
            ema = st.get("ema", m)
            ema = (1 - self.EMA_ALPHA_MM) * ema + self.EMA_ALPHA_MM * m
            st["ema"] = ema
            td[key] = st

            pos = state.position.get(product, 0)
            buy_cap, sell_cap = remaining_capacity(pos)

            # Inventory-aware fair
            fair = ema - self.MM_INV_PENALTY * pos

            # Apply panel cross-pricing
            if product == "PANEL_1X4":
                fair += panel_adj_1x4

            # Apply oxygen cross-pricing
            if product in oxy_adj:
                fair += oxy_adj[product]

            # Apply Robots↔SleepPods hedge only to the non-pair SleepPod names.
            if product in sleep_pod_hedge_names:
                fair += cross_hedge_ticks

            # Quote inside when spread is wide enough.
            inside = 1 if sp >= 4 else 0
            bid_q = int(bb + inside)
            ask_q = int(ba - inside)

            fair_shift = int(round((fair - m) / 2.0))
            fair_shift = int(clamp(fair_shift, -2.0, 2.0))
            bid_q += fair_shift
            ask_q += fair_shift

            if bid_q >= ask_q:
                bid_q = int(bb)
                ask_q = int(ba)
            if bid_q >= ask_q:
                ask_q = bid_q + 1

            # Reduce size / one-side quoting near limits.
            size = min(self.MM_CLIP, 2 if sp >= 5 else self.MM_CLIP)
            if pos >= 8:
                buy_cap = 0
            if pos <= -8:
                sell_cap = 0

            if buy_cap > 0:
                orders.setdefault(product, []).append(Order(product, bid_q, min(buy_cap, size)))
            if sell_cap > 0:
                orders.setdefault(product, []).append(Order(product, ask_q, -min(sell_cap, size)))

        # --- Persist state (truncate to safety) ---
        next_state = json.dumps(td, separators=(",", ":"), sort_keys=True)
        if len(next_state) > 49000:
            essential = {k: v for k, v in td.items() if k.startswith("nc_") or k.startswith("pc_") or k.startswith("rv_") or k.startswith("trend_") or k.startswith("mm_")}
            next_state = json.dumps(essential, separators=(",", ":"), sort_keys=True)

        return orders, 0, next_state