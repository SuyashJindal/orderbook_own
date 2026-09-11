from dataclasses import dataclass, field
from collections import deque
import heapq
from enum import Enum
from typing import Optional


class Side(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"


@dataclass
class Order:
    order_id: int
    side: Side
    price: Optional[float]
    quantity: int
    timestamp: int
    order_type: OrderType = OrderType.LIMIT

    # Remaining quantity
    remaining: int = field(init=False)

    def __post_init__(self):
        self.remaining = self.quantity


@dataclass
class Trade:
    trade_id: int
    price: float
    quantity: int
    buy_order_id: int
    sell_order_id: int


class OrderBook:

    def __init__(self):
        # price -> FIFO queue of orders
        self.bids = {}   # BUY
        self.asks = {}   # SELL

        # heaps containing prices
        # Python has min heap, so bids use negative prices
        self.bid_heap = []
        self.ask_heap = []

        self.orders = {}

    # --------------------------------------------------
    # Add order to book
    # --------------------------------------------------

    def add_order(self, order: Order):

        self.orders[order.order_id] = order

        if order.side == Side.BUY:

            if order.price not in self.bids:
                self.bids[order.price] = deque()
                heapq.heappush(self.bid_heap, -order.price)

            self.bids[order.price].append(order)

        else:

            if order.price not in self.asks:
                self.asks[order.price] = deque()
                heapq.heappush(self.ask_heap, order.price)

            self.asks[order.price].append(order)

    # --------------------------------------------------
    # Remove empty price levels
    # --------------------------------------------------

    def clean_bids(self):

        while self.bid_heap:

            price = -self.bid_heap[0]

            if price in self.bids and self.bids[price]:
                break

            heapq.heappop(self.bid_heap)

    def clean_asks(self):

        while self.ask_heap:

            price = self.ask_heap[0]

            if price in self.asks and self.asks[price]:
                break

            heapq.heappop(self.ask_heap)

    # --------------------------------------------------
    # Best bid / ask
    # --------------------------------------------------

    def best_bid(self):

        self.clean_bids()

        if not self.bid_heap:
            return None

        return -self.bid_heap[0]

    def best_ask(self):

        self.clean_asks()

        if not self.ask_heap:
            return None

        return self.ask_heap[0]

    # --------------------------------------------------
    # Remove completed order
    # --------------------------------------------------

    def remove_front_order(self, side, price):

        book = self.bids if side == Side.BUY else self.asks

        queue = book[price]

        order = queue.popleft()

        if not queue:
            del book[price]

        return order

    # --------------------------------------------------
    # Display order book
    # --------------------------------------------------

    def display(self):

        print("\n----------- ORDER BOOK -----------")

        print("ASKS:")

        for price in sorted(self.asks):
            qty = sum(o.remaining for o in self.asks[price])
            print(f"{price:>10.2f} | {qty}")

        print("---------------------------------")

        print("BIDS:")

        for price in sorted(self.bids, reverse=True):
            qty = sum(o.remaining for o in self.bids[price])
            print(f"{price:>10.2f} | {qty}")

        print("---------------------------------\n")


class MatchingEngine:

    def __init__(self):

        self.book = OrderBook()

        self.order_id = 0
        self.timestamp = 0
        self.trade_id = 0

        self.trades = []

    # --------------------------------------------------
    # Generate IDs
    # --------------------------------------------------

    def next_order_id(self):

        self.order_id += 1
        return self.order_id

    def next_timestamp(self):

        self.timestamp += 1
        return self.timestamp

    def next_trade_id(self):

        self.trade_id += 1
        return self.trade_id

    # --------------------------------------------------
    # Submit order
    # --------------------------------------------------

    def submit_order(
        self,
        side,
        quantity,
        price=None,
        order_type=OrderType.LIMIT
    ):

        order = Order(
            order_id=self.next_order_id(),
            side=side,
            price=price,
            quantity=quantity,
            timestamp=self.next_timestamp(),
            order_type=order_type
        )

        print(
            f"NEW ORDER -> "
            f"id={order.order_id}, "
            f"{side.value}, "
            f"qty={quantity}, "
            f"price={price}"
        )

        self.match(order)

        # If quantity remains, limit order rests in book
        if order.remaining > 0 and order.order_type == OrderType.LIMIT:

            self.book.add_order(order)

            print(
                f"RESTED -> "
                f"id={order.order_id}, "
                f"remaining={order.remaining}"
            )

        elif order.remaining > 0:

            print(
                f"MARKET ORDER CANCELLED -> "
                f"remaining={order.remaining}"
            )

        return order

    # --------------------------------------------------
    # Matching algorithm
    # --------------------------------------------------

    def match(self, incoming):

        while incoming.remaining > 0:

            # ------------------------------------------
            # Incoming BUY
            # ------------------------------------------

            if incoming.side == Side.BUY:

                best_ask = self.book.best_ask()

                # No asks
                if best_ask is None:
                    break

                # Limit buy cannot cross ask
                if (
                    incoming.order_type == OrderType.LIMIT
                    and incoming.price < best_ask
                ):
                    break

                queue = self.book.asks[best_ask]

                resting = queue[0]

            # ------------------------------------------
            # Incoming SELL
            # ------------------------------------------

            else:

                best_bid = self.book.best_bid()

                # No bids
                if best_bid is None:
                    break

                # Limit sell cannot cross bid
                if (
                    incoming.order_type == OrderType.LIMIT
                    and incoming.price > best_bid
                ):
                    break

                queue = self.book.bids[best_bid]

                resting = queue[0]

            # ------------------------------------------
            # Execute trade
            # ------------------------------------------

            trade_quantity = min(
                incoming.remaining,
                resting.remaining
            )

            # Trade happens at resting order's price
            trade_price = resting.price

            incoming.remaining -= trade_quantity
            resting.remaining -= trade_quantity

            if incoming.side == Side.BUY:

                buy_id = incoming.order_id
                sell_id = resting.order_id

            else:

                buy_id = resting.order_id
                sell_id = incoming.order_id

            trade = Trade(
                trade_id=self.next_trade_id(),
                price=trade_price,
                quantity=trade_quantity,
                buy_order_id=buy_id,
                sell_order_id=sell_id
            )

            self.trades.append(trade)

            print(
                f"TRADE -> "
                f"price={trade_price}, "
                f"qty={trade_quantity}, "
                f"buy={buy_id}, "
                f"sell={sell_id}"
            )

            # ------------------------------------------
            # Remove fully filled resting order
            # ------------------------------------------

            if resting.remaining == 0:

                self.book.remove_front_order(
                    resting.side,
                    resting.price
                )

                print(
                    f"FILLED -> order {resting.order_id}"
                )

        return incoming


# ======================================================
# Example
# ======================================================

engine = MatchingEngine()

# SELL 100 @ 101
engine.submit_order(
    Side.SELL,
    quantity=100,
    price=101
)

# SELL 50 @ 102
engine.submit_order(
    Side.SELL,
    quantity=50,
    price=102
)

# BUY 80 @ 100
engine.submit_order(
    Side.BUY,
    quantity=80,
    price=100
)

# BUY 120 @ 101
engine.submit_order(
    Side.BUY,
    quantity=120,
    price=101
)

engine.book.display()

print("TRADES:")

for trade in engine.trades:
    print(trade)
