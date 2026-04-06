
import json
from typing import Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, Product, TradingState


class Trader:
    """Simple inventory-aware market maker for the tutorial round.

    Calibrated from `TUTORIAL_ROUND_1` sample data:
    - EMERALDS: very stable around 10_000, spread typically 16.
    - TOMATOES: noisier mid, spread typically ~13-14.
    """

    POSITION_LIMITS: Dict[Product, int] = {
        "EMERALDS": 80,
        "TOMATOES": 80,
    }

    # Fair-value process: EWMA on observed mid.
    FAIR_EWMA_ALPHA: Dict[Product, float] = {
        "EMERALDS": 0.0,  # fixed fair below
        "TOMATOES": 0.90,
    }

    # Half-spread (edge) we require around the reservation price.
    EDGE: Dict[Product, int] = {
        "EMERALDS": 3,
        "TOMATOES": 4,
    }

    # Inventory skew in price units per position unit.
    INV_SKEW_PER_UNIT: Dict[Product, float] = {
        "EMERALDS": 0.10,
        "TOMATOES": 0.20,
    }

    BASE_MAKER_SIZE: Dict[Product, int] = {
        "EMERALDS": 12,
        "TOMATOES": 10,
    }

    def bid(self) -> int:
        # Ignored outside Round 2; safe to include.
        return 0

    @staticmethod
    def _best_bid_ask(depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
        best_bid = max(depth.buy_orders) if depth.buy_orders else None
        best_ask = min(depth.sell_orders) if depth.sell_orders else None
        return best_bid, best_ask

    @staticmethod
    def _mid_from_depth(depth: OrderDepth) -> Optional[float]:
        best_bid, best_ask = Trader._best_bid_ask(depth)
        if best_bid is None or best_ask is None:
            return None
        return (best_bid + best_ask) / 2.0

    @staticmethod
    def _load_state(traderData: str) -> Dict:
        if not traderData:
            return {}
        try:
            data = json.loads(traderData)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _dump_state(data: Dict) -> str:
        try:
            return json.dumps(data, separators=(",", ":"), sort_keys=True)
        except Exception:
            return ""

    def _position(self, state: TradingState, product: Product) -> int:
        return int(state.position.get(product, 0))

    def _remaining_capacity(self, pos: int, limit: int) -> Tuple[int, int]:
        # Max additional buy / sell quantities we can submit this iteration.
        max_buy = max(0, limit - pos)
        max_sell = max(0, limit + pos)
        return max_buy, max_sell

    def _take_opportunities(
        self,
        product: Product,
        depth: OrderDepth,
        fair: float,
        pos: int,
        limit: int,
    ) -> Tuple[List[Order], int, int]:
        """Aggressively take mispriced liquidity, bounded by position limits.

        Returns: (orders, remaining_buy, remaining_sell)
        """
        orders: List[Order] = []
        remaining_buy, remaining_sell = self._remaining_capacity(pos, limit)
        edge = self.EDGE[product]
        take_threshold = edge + 1

        # Buy cheap asks.
        for ask_px in sorted(depth.sell_orders):
            if remaining_buy <= 0:
                break
            if ask_px > fair - take_threshold:
                break
            ask_vol = -int(depth.sell_orders[ask_px])  # sell volumes are negative
            qty = min(ask_vol, remaining_buy)
            if qty > 0:
                orders.append(Order(product, int(ask_px), int(qty)))
                remaining_buy -= qty
                remaining_sell += qty

        # Sell rich bids.
        for bid_px in sorted(depth.buy_orders, reverse=True):
            if remaining_sell <= 0:
                break
            if bid_px < fair + take_threshold:
                break
            bid_vol = int(depth.buy_orders[bid_px])
            qty = min(bid_vol, remaining_sell)
            if qty > 0:
                orders.append(Order(product, int(bid_px), -int(qty)))
                remaining_sell -= qty
                remaining_buy += qty

        return orders, remaining_buy, remaining_sell

    def _maker_quotes(
        self,
        product: Product,
        depth: OrderDepth,
        reservation: float,
        pos: int,
        limit: int,
        remaining_buy: int,
        remaining_sell: int,
    ) -> List[Order]:
        best_bid, best_ask = self._best_bid_ask(depth)
        if best_bid is None or best_ask is None:
            return []

        edge = self.EDGE[product]
        bid_target = int(round(reservation - edge))
        ask_target = int(round(reservation + edge))

        # Never cross the book for maker quotes.
        bid_px = min(bid_target, best_ask - 1)
        ask_px = max(ask_target, best_bid + 1)

        # If there is room, optionally penny inside the spread without worsening our edge.
        if best_bid + 1 <= best_ask - 1:
            max_bid_px = int(round(reservation - edge))
            if best_bid + 1 <= max_bid_px:
                bid_px = max(bid_px, best_bid + 1)

            min_ask_px = int(round(reservation + edge))
            if best_ask - 1 >= min_ask_px:
                ask_px = min(ask_px, best_ask - 1)

        if bid_px >= ask_px:
            return []

        base = self.BASE_MAKER_SIZE[product]
        # Scale maker size down as we approach limits.
        buy_size = int(round(base * (remaining_buy / limit))) if limit > 0 else 0
        sell_size = int(round(base * (remaining_sell / limit))) if limit > 0 else 0

        # Keep at least 1-lot quotes when we have capacity.
        if remaining_buy > 0:
            buy_size = max(1, min(buy_size, remaining_buy))
        else:
            buy_size = 0
        if remaining_sell > 0:
            sell_size = max(1, min(sell_size, remaining_sell))
        else:
            sell_size = 0

        orders: List[Order] = []
        if buy_size > 0:
            orders.append(Order(product, int(bid_px), int(buy_size)))
        if sell_size > 0:
            orders.append(Order(product, int(ask_px), -int(sell_size)))
        return orders

    def run(self, state: TradingState):
        result: Dict[Product, List[Order]] = {}
        persistent = self._load_state(state.traderData)
        fair_state: Dict[str, float] = persistent.get("fair", {}) if isinstance(persistent.get("fair", {}), dict) else {}

        for product, depth in state.order_depths.items():
            if product not in self.POSITION_LIMITS:
                continue

            limit = self.POSITION_LIMITS[product]
            pos = self._position(state, product)

            mid = self._mid_from_depth(depth)
            if mid is None:
                result[product] = []
                continue

            # Fair value: EMERALDS are essentially fixed; TOMATOES use EWMA of mid.
            if product == "EMERALDS":
                fair = 10000.0
            else:
                prev = float(fair_state.get(product, mid))
                a = float(self.FAIR_EWMA_ALPHA[product])
                fair = a * prev + (1.0 - a) * float(mid)
                fair_state[product] = fair

            # Reservation price with inventory skew (pushes us to mean-revert inventory).
            reservation = fair - (float(pos) * float(self.INV_SKEW_PER_UNIT[product]))

            orders: List[Order] = []

            taker, remaining_buy, remaining_sell = self._take_opportunities(
                product=product,
                depth=depth,
                fair=fair,
                pos=pos,
                limit=limit,
            )
            orders.extend(taker)

            orders.extend(
                self._maker_quotes(
                    product=product,
                    depth=depth,
                    reservation=reservation,
                    pos=pos,
                    limit=limit,
                    remaining_buy=remaining_buy,
                    remaining_sell=remaining_sell,
                )
            )

            result[product] = orders

        persistent["fair"] = fair_state
        traderData = self._dump_state(persistent)
        conversions = 0
        return result, conversions, traderData
