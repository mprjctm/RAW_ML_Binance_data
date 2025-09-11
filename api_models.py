from pydantic import BaseModel, Field
from typing import List, Tuple
from datetime import datetime

# --- Base Models ---

class Trade(BaseModel):
    """Represents a single trade."""
    price: float = Field(..., description="Price of the trade")
    quantity: float = Field(..., alias="q", description="Quantity of the trade")
    timestamp: datetime = Field(..., alias="T", description="Timestamp of the trade")
    is_buyer_maker: bool = Field(..., alias="m", description="Was the buyer the maker?")

class OrderBookEntry(BaseModel):
    """Represents a single entry (price and quantity) in the order book."""
    price: float
    quantity: float

class OrderBook(BaseModel):
    """Represents a full order book snapshot."""
    timestamp: datetime
    symbol: str
    bids: List[OrderBookEntry]
    asks: List[OrderBookEntry]

class LiquidationOrder(BaseModel):
    """Represents a single liquidation order."""
    symbol: str = Field(..., alias="s", description="The symbol")
    side: str = Field(..., alias="S", description="Side of the order (BUY or SELL)")
    order_type: str = Field(..., alias="o", description="Order type (e.g., LIMIT, MARKET)")
    time_in_force: str = Field(..., alias="f", description="Time in force (e.g., IOC, GTC)")
    original_quantity: float = Field(..., alias="q", description="Original quantity of the order")
    price: float = Field(..., alias="p", description="Price of the order")
    average_price: float = Field(..., alias="ap", description="Average price")
    order_status: str = Field(..., alias="X", description="Order status (e.g., FILLED)")
    last_filled_quantity: float = Field(..., alias="l", description="Last filled quantity")
    cumulative_filled_quantity: float = Field(..., alias="z", description="Cumulative filled quantity")
    trade_time: datetime = Field(..., alias="T", description="Timestamp of the trade")


class LiquidationEvent(BaseModel):
    """Represents a full liquidation event message from Binance."""
    event_time: datetime = Field(..., alias="E", description="The event time")
    liquidation_order: LiquidationOrder = Field(..., alias="o", description="The liquidation order details")


# --- API Request/Response Models ---

class OrderRequest(BaseModel):
    """Represents the request body for placing a new order."""
    symbol: str
    side: str # "BUY" or "SELL"
    order_type: str # "MARKET" or "LIMIT"
    quantity: float
    price: float | None = None # Required for LIMIT orders

class OrderResponse(BaseModel):
    """Represents the response after placing an order."""
    status: str
    order_id: str | None = None
    message: str
