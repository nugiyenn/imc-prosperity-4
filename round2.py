import json
from typing import Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, Product, TradingState


class Trader:
    MARKET_ACCESS_BID = 20

    POSITION_LIMITS: Dict[Product, int] = {
        "ASH_COATED_OSMIUM": 80,
        "INTARIAN_PEPPER_ROOT": 80,
    }

    ASH_FAIR_VALUE = 10000
    ASH_BASE_MAKER_SIZE = 20
    ASH_SKEW_TRIGGER = 70
    ASH_FLATTEN_CLIP = 40

    IPR_BASE_MAKER_SIZE = 8
    IPR_TREND_EDGE = 2
    IPR_ENTRY_BUFFER = 3
    IPR_FINAL_LIQUIDATION_START = 995000

    def bid(self) -> int:
        return self.MARKET_ACCESS_BID

    @staticmethod
    def _best_bid_ask(depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
        best_bid = max(depth.buy_orders) if depth.buy_orders else None
        best_ask = min(depth.sell_orders) if depth.sell_orders else None
        return best_bid, best_ask

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

    @staticmethod
    def _safe_int_price(value: object, fallback: int) -> int:
        try:
            fval = float(value)
            if fval != fval:
                return fallback
            return int(round(fval))
        except Exception:
            return fallback

    def _trade_ash(self, state: TradingState, depth: OrderDepth) -> List[Order]:
        product = "ASH_COATED_OSMIUM"
        limit = self.POSITION_LIMITS[product]
        fair = self.ASH_FAIR_VALUE
        position = int(state.position.get(product, 0))

        orders: List[Order] = []
        remaining_buy, remaining_sell = self._remaining_capacity(position, limit)
        projected_position = position

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
                projected_position += qty

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
                projected_position -= qty

        if abs(projected_position) >= self.ASH_SKEW_TRIGGER:
            flatten_qty = min(abs(projected_position), self.ASH_FLATTEN_CLIP)
            if projected_position > 0 and remaining_sell > 0:
                qty = min(flatten_qty, remaining_sell)
                if qty > 0:
                    orders.append(Order(product, fair, -int(qty)))
                    remaining_sell -= qty
                    remaining_buy += qty
                    projected_position -= qty
            elif projected_position < 0 and remaining_buy > 0:
                qty = min(flatten_qty, remaining_buy)
                if qty > 0:
                    orders.append(Order(product, fair, int(qty)))
                    remaining_buy -= qty
                    remaining_sell += qty
                    projected_position += qty

        best_bid, best_ask = self._best_bid_ask(depth)
        if best_bid is None or best_ask is None:
            return orders

        bid_quote = min(best_bid + 1, fair - 1)
        ask_quote = max(best_ask - 1, fair + 1)
        if bid_quote >= ask_quote:
            return orders

        inv_ratio = projected_position / float(limit) if limit else 0.0
        buy_scale = max(0.0, 1.0 - max(inv_ratio, 0.0))
        sell_scale = max(0.0, 1.0 - max(-inv_ratio, 0.0))
        buy_size = min(remaining_buy, int(round(self.ASH_BASE_MAKER_SIZE * buy_scale)))
        sell_size = min(remaining_sell, int(round(self.ASH_BASE_MAKER_SIZE * sell_scale)))

        if projected_position >= limit - 5:
            buy_size = 0
        if projected_position <= -limit + 5:
            sell_size = 0

        if buy_size > 0:
            orders.append(Order(product, int(bid_quote), int(buy_size)))
        if sell_size > 0:
            orders.append(Order(product, int(ask_quote), -int(sell_size)))

        return orders

    def _trade_ipr(
        self,
        state: TradingState,
        depth: OrderDepth,
        persistent: Dict,
    ) -> List[Order]:
        product = "INTARIAN_PEPPER_ROOT"
        limit = self.POSITION_LIMITS[product]
        position = int(state.position.get(product, 0))
        timestamp = int(state.timestamp)

        best_bid, best_ask = self._best_bid_ask(depth)
        if best_bid is None and best_ask is None:
            return []

        if best_bid is None:
            best_bid = best_ask
        if best_ask is None:
            best_ask = best_bid

        ipr_state = persistent.get("ipr", {})
        if not isinstance(ipr_state, dict):
            ipr_state = {}

        anchor_ask = self._safe_int_price(ipr_state.get("anchor_ask"), int(best_ask))
        if "anchor_ask" not in ipr_state:
            anchor_ask = int(best_ask)

        progress = max(0.0, min(1.0, timestamp / 999900.0))
        expected_mid = anchor_ask + int(round(1000.0 * progress))

        orders: List[Order] = []
        remaining_buy, remaining_sell = self._remaining_capacity(position, limit)
        projected_position = position

        if timestamp >= self.IPR_FINAL_LIQUIDATION_START:
            if projected_position > 0 and remaining_sell > 0:
                qty = min(projected_position, remaining_sell)
                if qty > 0:
                    orders.append(Order(product, int(best_bid), -int(qty)))
            elif projected_position < 0 and remaining_buy > 0:
                qty = min(-projected_position, remaining_buy)
                if qty > 0:
                    orders.append(Order(product, int(best_ask), int(qty)))

            ipr_state["anchor_ask"] = int(anchor_ask)
            persistent["ipr"] = ipr_state
            return orders

        if projected_position < limit and remaining_buy > 0:
            if int(best_ask) <= expected_mid + self.IPR_ENTRY_BUFFER:
                qty = min(remaining_buy, -int(depth.sell_orders.get(best_ask, 0)))
                if qty > 0:
                    orders.append(Order(product, int(best_ask), int(qty)))
                    remaining_buy -= qty
                    remaining_sell += qty
                    projected_position += qty

            if projected_position < limit and remaining_buy > 0:
                join_buy_px = min(int(best_bid) + 1, int(best_ask))
                maker_buy = min(remaining_buy, self.IPR_BASE_MAKER_SIZE)
                if maker_buy > 0:
                    orders.append(Order(product, int(join_buy_px), int(maker_buy)))
                    remaining_buy -= maker_buy
                    remaining_sell += maker_buy
                    projected_position += maker_buy

        if projected_position >= limit - 2 and remaining_sell > 0:
            target_ask = max(int(best_ask) - 1, expected_mid + self.IPR_TREND_EDGE)
            maker_sell = min(remaining_sell, 2)
            if maker_sell > 0 and target_ask > int(best_bid):
                orders.append(Order(product, int(target_ask), -int(maker_sell)))

        ipr_state["anchor_ask"] = int(anchor_ask)
        persistent["ipr"] = ipr_state
        return orders

    def run(self, state: TradingState):
        result: Dict[Product, List[Order]] = {}
        persistent = self._load_state(state.traderData)

        for product, depth in state.order_depths.items():
            if product == "ASH_COATED_OSMIUM":
                result[product] = self._trade_ash(state, depth)
            elif product == "INTARIAN_PEPPER_ROOT":
                result[product] = self._trade_ipr(state, depth, persistent)
            else:
                result[product] = []

        trader_data = self._dump_state(persistent)
        return result, 0, trader_data