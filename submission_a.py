import json
from typing import Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, Product, TradingState


class Trader:
    POSITION_LIMITS: Dict[Product, int] = {
        "EMERALDS": 80,
        "TOMATOES": 80,
    }

    EMERALDS_FAIR = 10000

    TOMATOES_EWMA_ALPHA = 0.90
    TOMATOES_EDGE = 3
    TOMATOES_INV_SKEW = 0.20
    TOMATOES_BASE_MAKER_SIZE = 10
    TOMATOES_SKEW_TRIGGER = 50
    TOMATOES_NEUTRALIZE_CLIP = 20

    EMERALDS_BASE_MAKER_SIZE = 20
    EMERALDS_SKEW_TRIGGER = 50
    EMERALDS_FLATTEN_CLIP = 50

    def bid(self) -> int:
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
    def _load_state(trader_data: str) -> Dict:
        if not trader_data:
            return {}
        try:
            data = json.loads(trader_data)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _dump_state(data: Dict) -> str:
        try:
            return json.dumps(data, separators=(",", ":"), sort_keys=True)
        except Exception:
            return ""

    @staticmethod
    def _remaining_capacity(position: int, limit: int) -> Tuple[int, int]:
        max_buy = max(0, limit - position)
        max_sell = max(0, limit + position)
        return max_buy, max_sell

    def _trade_emeralds(self, state: TradingState, depth: OrderDepth) -> List[Order]:
        product = "EMERALDS"
        limit = self.POSITION_LIMITS[product]
        fair = self.EMERALDS_FAIR
        position = int(state.position.get(product, 0))

        orders: List[Order] = []
        remaining_buy, remaining_sell = self._remaining_capacity(position, limit)

        # 1) Take favorable liquidity immediately.
        for ask_px in sorted(depth.sell_orders):
            if remaining_buy <= 0:
                break
            if ask_px >= fair:
                break

            ask_qty = -int(depth.sell_orders[ask_px])
            qty = min(ask_qty, remaining_buy)
            if qty > 0:
                orders.append(Order(product, int(ask_px), int(qty)))
                remaining_buy -= qty
                remaining_sell += qty
                position += qty

        for bid_px in sorted(depth.buy_orders, reverse=True):
            if remaining_sell <= 0:
                break
            if bid_px <= fair:
                break

            bid_qty = int(depth.buy_orders[bid_px])
            qty = min(bid_qty, remaining_sell)
            if qty > 0:
                orders.append(Order(product, int(bid_px), -int(qty)))
                remaining_sell -= qty
                remaining_buy += qty
                position -= qty

        # 2) If inventory is too skewed, place a flattening order at fair.
        if abs(position) >= self.EMERALDS_SKEW_TRIGGER:
            flatten_qty = min(abs(position), self.EMERALDS_FLATTEN_CLIP)
            if position > 0 and remaining_sell > 0:
                qty = min(flatten_qty, remaining_sell)
                if qty > 0:
                    orders.append(Order(product, fair, -int(qty)))
                    remaining_sell -= qty
                    remaining_buy += qty
                    position -= qty
            elif position < 0 and remaining_buy > 0:
                qty = min(flatten_qty, remaining_buy)
                if qty > 0:
                    orders.append(Order(product, fair, int(qty)))
                    remaining_buy -= qty
                    remaining_sell += qty
                    position += qty

        # 3) Passive quotes: overbid/undercut while keeping positive edge.
        best_bid, best_ask = self._best_bid_ask(depth)
        if best_bid is None or best_ask is None:
            return orders

        bid_quote = min(best_bid + 1, fair - 1)
        ask_quote = max(best_ask - 1, fair + 1)

        inv_ratio = position / float(limit) if limit else 0.0
        buy_scale = max(0.0, 1.0 - max(inv_ratio, 0.0))
        sell_scale = max(0.0, 1.0 - max(-inv_ratio, 0.0))
        buy_size = min(remaining_buy, int(round(self.EMERALDS_BASE_MAKER_SIZE * buy_scale)))
        sell_size = min(remaining_sell, int(round(self.EMERALDS_BASE_MAKER_SIZE * sell_scale)))

        if position >= limit - 5:
            buy_size = 0
        if position <= -limit + 5:
            sell_size = 0

        if bid_quote < ask_quote and buy_size > 0:
            orders.append(Order(product, int(bid_quote), int(buy_size)))
        if bid_quote < ask_quote and sell_size > 0:
            orders.append(Order(product, int(ask_quote), -int(sell_size)))

        return orders

    def _trade_tomatoes(
        self,
        state: TradingState,
        depth: OrderDepth,
        fair_state: Dict[str, float],
    ) -> List[Order]:
        product = "TOMATOES"
        limit = self.POSITION_LIMITS[product]
        position = int(state.position.get(product, 0))

        mid = self._mid_from_depth(depth)
        if mid is None:
            return []

        prev = float(fair_state.get(product, mid))
        fair = self.TOMATOES_EWMA_ALPHA * prev + (1.0 - self.TOMATOES_EWMA_ALPHA) * float(mid)
        fair_state[product] = fair

        edge = self.TOMATOES_EDGE
        orders: List[Order] = []
        remaining_buy, remaining_sell = self._remaining_capacity(position, limit)
        projected_position = position

        # 1) Take immediately favorable liquidity versus current mid.
        for ask_px in sorted(depth.sell_orders):
            if remaining_buy <= 0:
                break
            if ask_px >= mid:
                break

            ask_qty = -int(depth.sell_orders[ask_px])
            qty = min(ask_qty, remaining_buy)
            if qty > 0:
                orders.append(Order(product, int(ask_px), int(qty)))
                remaining_buy -= qty
                remaining_sell += qty
                projected_position += qty

        for bid_px in sorted(depth.buy_orders, reverse=True):
            if remaining_sell <= 0:
                break
            if bid_px <= mid:
                break

            bid_qty = int(depth.buy_orders[bid_px])
            qty = min(bid_qty, remaining_sell)
            if qty > 0:
                orders.append(Order(product, int(bid_px), -int(qty)))
                remaining_sell -= qty
                remaining_buy += qty
                projected_position -= qty

        # 2) If inventory is too skewed, neutralize at zero edge to estimate.
        if abs(projected_position) >= self.TOMATOES_SKEW_TRIGGER:
            neutral_px = int(round(fair))
            neutral_qty = min(abs(projected_position), self.TOMATOES_NEUTRALIZE_CLIP)
            if projected_position > 0 and remaining_sell > 0:
                qty = min(neutral_qty, remaining_sell)
                if qty > 0:
                    orders.append(Order(product, neutral_px, -int(qty)))
                    remaining_sell -= qty
                    remaining_buy += qty
                    projected_position -= qty
            elif projected_position < 0 and remaining_buy > 0:
                qty = min(neutral_qty, remaining_buy)
                if qty > 0:
                    orders.append(Order(product, neutral_px, int(qty)))
                    remaining_buy -= qty
                    remaining_sell += qty
                    projected_position += qty

        # 3) Maker leg: overbid/undercut while quoting around fair.
        best_bid, best_ask = self._best_bid_ask(depth)
        if best_bid is None or best_ask is None:
            return orders

        fair_bid = int(round(fair - edge))
        fair_ask = int(round(fair + edge))
        bid_quote = min(best_bid + 1, fair_bid)
        ask_quote = max(best_ask - 1, fair_ask)

        if bid_quote >= ask_quote:
            return orders

        inv_ratio = projected_position / float(limit) if limit else 0.0
        buy_scale = max(0.0, 1.0 - max(inv_ratio, 0.0))
        sell_scale = max(0.0, 1.0 - max(-inv_ratio, 0.0))
        buy_size = min(remaining_buy, int(round(self.TOMATOES_BASE_MAKER_SIZE * buy_scale)))
        sell_size = min(remaining_sell, int(round(self.TOMATOES_BASE_MAKER_SIZE * sell_scale)))

        if remaining_buy > 0:
            buy_size = max(1, buy_size)
        if remaining_sell > 0:
            sell_size = max(1, sell_size)

        if buy_size > 0:
            orders.append(Order(product, int(bid_quote), int(buy_size)))
        if sell_size > 0:
            orders.append(Order(product, int(ask_quote), -int(sell_size)))

        return orders

    def run(self, state: TradingState):
        result: Dict[Product, List[Order]] = {}
        persistent = self._load_state(state.traderData)
        fair_state = persistent.get("fair", {}) if isinstance(persistent.get("fair", {}), dict) else {}

        for product, depth in state.order_depths.items():
            if product == "EMERALDS":
                result[product] = self._trade_emeralds(state, depth)
            elif product == "TOMATOES":
                result[product] = self._trade_tomatoes(state, depth, fair_state)
            else:
                result[product] = []

        persistent["fair"] = fair_state
        trader_data = self._dump_state(persistent)
        return result, 0, trader_data
