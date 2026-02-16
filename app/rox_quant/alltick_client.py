import json
import threading
import time
import websocket
import ssl
from typing import Dict, List, Callable, Optional
from enum import Enum
import logging

# Configure logging
logger = logging.getLogger(__name__)

class ConnectionStatus(Enum):
    """连接状态"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"


class AllTickClient:
    """
    Client for AllTick WebSocket API (Real-time Market Data).
    Supports A-share, HK stocks, US stocks, Forex, Crypto, etc.
    """
    
    WS_URL = "wss://quote.alltick.co/quote-stock-b-ws-api"
    
    def __init__(self, token: str, max_reconnects: int = 5, reconnect_delay: int = 3):
        self.token = token
        self.ws_url = f"{self.WS_URL}?token={self.token}"
        self.ws: Optional[websocket.WebSocketApp] = None
        self.wst: Optional[threading.Thread] = None
        self.status = ConnectionStatus.DISCONNECTED
        self.callbacks: List[Callable] = []
        self.status_callbacks: List[Callable] = []
        self.subscribed_symbols: Dict[str, int] = {} # code -> depth_level
        self.latest_ticks: Dict[str, dict] = {} # code -> tick_data
        self.tick_history: Dict[str, List[dict]] = {} # code -> list of ticks
        self.history_max_size = 50  # Maximum number of ticks to keep per symbol
        self._trace_id = 0
        self.max_reconnects = max_reconnects
        self.reconnect_delay = reconnect_delay
        self.reconnect_count = 0
        self.heartbeat_interval = 15  # Heartbeat interval in seconds
        self.last_heartbeat = 0
        self.last_message_time = 0
        self.message_timeout = 30  # Timeout if no message received in seconds
        self._lock = threading.RLock()  # Thread-safe operations
        self._reconnecting = False
        self._health_check_thread = None
        self._connection_quality = 1.0  # 0.0-1.0, 1.0 is perfect
        self._last_connection_time = 0
        self._total_reconnects = 0
        self._message_count = 0
        self._error_count = 0
        self._rate_limit_window = []  # Recent message timestamps for rate limiting
        self._max_messages_per_second = 100
        
    def _get_trace(self):
        self._trace_id += 1
        return f"rox_quant_{self._trace_id}"

    def connect(self):
        """Start the WebSocket connection in a separate thread."""
        with self._lock:
            if self.status in [ConnectionStatus.CONNECTED, ConnectionStatus.CONNECTING, ConnectionStatus.RECONNECTING]:
                return

            logger.info(f"Connecting to AllTick WS: {self.WS_URL}")
            self.status = ConnectionStatus.CONNECTING
            
        try:
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            
            self.wst = threading.Thread(target=self.ws.run_forever, kwargs={"sslopt": {"cert_reqs": ssl.CERT_NONE}})
            self.wst.daemon = True
            self.wst.start()
        except Exception as e:
            logger.error(f"Failed to create WebSocket connection: {e}")
            with self._lock:
                self.status = ConnectionStatus.DISCONNECTED
                self._reconnecting = False
            self._notify_status_change(ConnectionStatus.DISCONNECTED)

    def disconnect(self):
        """Close the WebSocket connection."""
        with self._lock:
            if self.status == ConnectionStatus.DISCONNECTED:
                return
                
            logger.info("Disconnecting from AllTick WS")
            
            if self.ws:
                try:
                    self.ws.close()
                except Exception as e:
                    logger.error(f"Error closing WebSocket: {e}")
            
            self.status = ConnectionStatus.DISCONNECTED
            self._reconnecting = False
            self.reconnect_count = 0
        
        self._notify_status_change(ConnectionStatus.DISCONNECTED)

    def subscribe(self, symbols: List[str], depth_level: int = 5):
        """
        Subscribe to market depth/tick data for a list of symbols.
        
        Args:
            symbols: List of symbol codes (e.g., ["700.HK", "AAPL.US", "600519.SH"])
            depth_level: Depth level (e.g., 5 for 5-level order book)
        """
        if not symbols:
            return

        # Update local subscription list
        with self._lock:
            new_symbols = []
            for s in symbols:
                if self.subscribed_symbols.get(s) != depth_level:
                    self.subscribed_symbols[s] = depth_level
                    new_symbols.append(s)
            
            if self.status == ConnectionStatus.CONNECTED and new_symbols:
                self._send_subscription(new_symbols, depth_level)
            elif self.status != ConnectionStatus.CONNECTED and new_symbols:
                logger.info(f"Will subscribe to {new_symbols} when connected")

    def _send_subscription(self, symbols: List[str], depth_level: int):
        """Send subscription request with error handling and retry."""
        symbol_list = [{"code": s, "depth_level": depth_level} for s in symbols]
        
        # Use modulo to ensure seq_id doesn't overflow 32-bit int
        seq_id = int(time.time() * 1000) % 2147483647
        
        payload = {
            "cmd_id": 22002, # Subscribe
            "seq_id": seq_id,
            "trace": self._get_trace(),
            "data": {
                "symbol_list": symbol_list
            }
        }
        
        for attempt in range(3):  # 3 retry attempts
            try:
                if self.status == ConnectionStatus.CONNECTED and self.ws:
                    # Rate limiting
                    self._rate_limit()
                    
                    self.ws.send(json.dumps(payload))
                    logger.info(f"Subscribed to {symbols} (depth: {depth_level})")
                    return
                else:
                    logger.warning("WebSocket not connected, skipping subscription")
                    return
            except Exception as e:
                logger.error(f"Failed to send subscription (attempt {attempt + 1}/3): {e}")
                if attempt < 2:
                    time.sleep(1)

    def add_callback(self, callback: Callable):
        """Add a callback function to handle incoming data."""
        with self._lock:
            if callback not in self.callbacks:
                self.callbacks.append(callback)

    def remove_callback(self, callback: Callable):
        """Remove a callback function."""
        with self._lock:
            if callback in self.callbacks:
                self.callbacks.remove(callback)
    
    def add_status_callback(self, callback: Callable):
        """Add a callback function to handle connection status changes."""
        with self._lock:
            if callback not in self.status_callbacks:
                self.status_callbacks.append(callback)
    
    def remove_status_callback(self, callback: Callable):
        """Remove a status callback function."""
        with self._lock:
            if callback in self.status_callbacks:
                self.status_callbacks.remove(callback)
    
    def _notify_status_change(self, status: ConnectionStatus):
        """Notify all status callbacks of a status change."""
        with self._lock:
            for cb in self.status_callbacks:
                try:
                    cb(status)
                except Exception as e:
                    logger.error(f"Status callback error: {e}")
    
    def _rate_limit(self):
        """Rate limit to avoid sending too many messages."""
        current_time = time.time()
        
        # Remove timestamps older than 1 second
        self._rate_limit_window = [t for t in self._rate_limit_window if current_time - t < 1]
        
        # Wait if we've reached the limit
        if len(self._rate_limit_window) >= self._max_messages_per_second:
            sleep_time = 1 - (current_time - self._rate_limit_window[0])
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _on_open(self, ws):
        logger.info("AllTick WS Connected")
        current_time = time.time()
        
        with self._lock:
            self.status = ConnectionStatus.CONNECTED
            self._reconnecting = False
            self.reconnect_count = 0
            self.last_message_time = current_time
            self._last_connection_time = current_time
            self._connection_quality = min(1.0, self._connection_quality + 0.1)
        
        self._notify_status_change(ConnectionStatus.CONNECTED)
        
        # Resubscribe if we have pending subscriptions
        with self._lock:
            if self.subscribed_symbols:
                # Group by depth level to send efficiently
                depth_groups = {}
                for code, depth in self.subscribed_symbols.items():
                    if depth not in depth_groups:
                        depth_groups[depth] = []
                    depth_groups[depth].append(code)
                
                for depth, codes in depth_groups.items():
                    self._send_subscription(codes, depth)
                    
        # Start heartbeat thread
        if not self._health_check_thread or not self._health_check_thread.is_alive():
            self._health_check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
            self._health_check_thread.start()

    def _health_check_loop(self):
        """Health check loop for heartbeat and reconnection."""
        while True:
            try:
                current_time = time.time()
                
                # Send heartbeat
                if self.status == ConnectionStatus.CONNECTED and current_time - self.last_heartbeat > self.heartbeat_interval:
                    self._send_heartbeat()
                    self.last_heartbeat = current_time
                
                # Check for message timeout
                if self.status == ConnectionStatus.CONNECTED and current_time - self.last_message_time > self.message_timeout:
                    logger.warning(f"No message received for {self.message_timeout} seconds, reconnecting...")
                    self._trigger_reconnect()
                
                # Check connection quality
                if self.status == ConnectionStatus.CONNECTED:
                    time_since_last_message = current_time - self.last_message_time
                    if time_since_last_message > 5:
                        # Degrade connection quality based on time since last message
                        degradation = min(0.5, time_since_last_message / self.message_timeout)
                        with self._lock:
                            self._connection_quality = max(0.1, self._connection_quality - degradation * 0.1)
                
                time.sleep(1)
            except Exception as e:
                logger.error(f"Health check error: {e}")
                time.sleep(1)

    def _send_heartbeat(self):
        """Send heartbeat message."""
        try:
            seq_id = int(time.time() * 1000) % 2147483647
            payload = {
                "cmd_id": 22000, # Heartbeat
                "seq_id": seq_id,
                "trace": self._get_trace(),
                "data": {}
            }
            if self.status == ConnectionStatus.CONNECTED and self.ws:
                # Rate limiting
                self._rate_limit()
                
                self.ws.send(json.dumps(payload))
        except Exception as e:
            logger.error(f"Failed to send heartbeat: {e}")
            self._trigger_reconnect()

    def _on_message(self, ws, message):
        """Handle incoming WebSocket messages with enhanced error handling."""
        try:
            current_time = time.time()
            self.last_message_time = current_time
            self._message_count += 1
            
            # Update connection quality
            with self._lock:
                self._connection_quality = min(1.0, self._connection_quality + 0.05)
            
            # Parse message
            data = json.loads(message)
            
            # Validate message structure
            if 'data' in data and isinstance(data['data'], dict):
                tick_data = data['data']
                
                # Store latest tick if it's market data
                if 'code' in tick_data:
                    code = tick_data['code']
                    
                    # Validate tick data integrity
                    if self._validate_tick_data(tick_data):
                        with self._lock:
                            self.latest_ticks[code] = tick_data
                            
                            # Store tick history
                            if code not in self.tick_history:
                                self.tick_history[code] = []
                            self.tick_history[code].append(tick_data)
                            
                            # Limit history size
                            if len(self.tick_history[code]) > self.history_max_size:
                                self.tick_history[code] = self.tick_history[code][-self.history_max_size:]
                        # logger.debug(f"Updated tick for {code}: {tick_data.get('last_price', 'N/A')}")
                    else:
                        logger.warning(f"Invalid tick data for {code}: {tick_data}")
            
            # Dispatch to callbacks
            with self._lock:
                for cb in self.callbacks:
                    try:
                        cb(data)
                    except Exception as cb_error:
                        logger.error(f"Callback error: {cb_error}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            self._error_count += 1
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self._error_count += 1

    def _validate_tick_data(self, tick_data: dict) -> bool:
        """Validate tick data integrity."""
        # Basic validation
        if not tick_data.get('code'):
            return False
        
        # Validate bids/asks data
        bids = tick_data.get('bids', [])
        asks = tick_data.get('asks', [])
        
        if bids:
            for bid in bids:
                if not isinstance(bid, dict) or 'price' not in bid or 'volume' not in bid:
                    return False
        
        if asks:
            for ask in asks:
                if not isinstance(ask, dict) or 'price' not in ask or 'volume' not in ask:
                    return False
        
        return True

    def _on_error(self, ws, error):
        logger.error(f"AllTick WS Error: {error}")
        self._error_count += 1
        with self._lock:
            self._connection_quality = max(0.1, self._connection_quality - 0.1)
        # Trigger reconnect on error
        self._trigger_reconnect()

    def _on_close(self, ws, close_status_code, close_msg):
        logger.info(f"AllTick WS Closed: {close_status_code} - {close_msg}")
        with self._lock:
            self.status = ConnectionStatus.DISCONNECTED
        # Trigger reconnect on close
        self._trigger_reconnect()

    def _trigger_reconnect(self):
        """Trigger reconnection with backoff."""
        with self._lock:
            if self._reconnecting or self.reconnect_count >= self.max_reconnects:
                return
                
            self.status = ConnectionStatus.RECONNECTING
            self._reconnecting = True
            self.reconnect_count += 1
            self._total_reconnects += 1
            self._connection_quality = max(0.1, self._connection_quality - 0.2)
            
            # Exponential backoff
            backoff_delay = min(self.reconnect_delay * (2 ** (self.reconnect_count - 1)), 60)
            
            logger.info(f"Attempting to reconnect ({self.reconnect_count}/{self.max_reconnects}) in {backoff_delay}s...")
        
        self._notify_status_change(ConnectionStatus.RECONNECTING)
        
        try:
            time.sleep(backoff_delay)
            self.connect()
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            with self._lock:
                self.status = ConnectionStatus.DISCONNECTED
                self._reconnecting = False
            self._notify_status_change(ConnectionStatus.DISCONNECTED)

    def get_tick_data(self, symbol: str) -> Optional[dict]:
        """Get latest tick data for a symbol with validation."""
        with self._lock:
            tick_data = self.latest_ticks.get(symbol)
            
        if not tick_data:
            return None
        
        # Check if data is recent (within 30 seconds)
        tick_time = tick_data.get('tick_time')
        if tick_time:
            try:
                # Parse tick time (assuming format: "2023-10-01 10:00:00.123")
                tick_timestamp = time.mktime(time.strptime(tick_time.split('.')[0], "%Y-%m-%d %H:%M:%S"))
                if time.time() - tick_timestamp > 30:
                    logger.warning(f"Tick data for {symbol} is stale ({tick_time})")
                    return None
            except Exception:
                pass
        
        return tick_data
    
    def get_tick_history(self, symbol: str, limit: int = 10) -> List[dict]:
        """Get recent tick history for a symbol."""
        with self._lock:
            history = self.tick_history.get(symbol, [])
            return history[-limit:]

    def get_order_book(self, symbol: str, depth: int = 5) -> Optional[dict]:
        """Get order book data with specified depth."""
        tick_data = self.get_tick_data(symbol)
        if not tick_data:
            return None
        
        bids = tick_data.get('bids', [])[:depth]
        asks = tick_data.get('asks', [])[:depth]
        
        return {
            "code": symbol,
            "bids": bids,
            "asks": asks,
            "tick_time": tick_data.get('tick_time'),
            "last_price": tick_data.get('last_price')
        }

    def is_healthy(self) -> bool:
        """Check if the client is healthy."""
        with self._lock:
            if self.status != ConnectionStatus.CONNECTED:
                return False
            
            # Check if we've received a message recently
            if time.time() - self.last_message_time > self.message_timeout:
                return False
            
            return True

    def get_health_status(self) -> dict:
        """Get detailed health status."""
        current_time = time.time()
        
        with self._lock:
            return {
                "status": self.status.value,
                "connected": self.status == ConnectionStatus.CONNECTED,
                "reconnect_count": self.reconnect_count,
                "total_reconnects": self._total_reconnects,
                "last_message_time": self.last_message_time,
                "last_heartbeat": self.last_heartbeat,
                "last_connection_time": self._last_connection_time,
                "message_count": self._message_count,
                "error_count": self._error_count,
                "connection_quality": round(self._connection_quality, 2),
                "subscribed_symbols": len(self.subscribed_symbols),
                "latest_ticks_count": len(self.latest_ticks),
                "tick_history_sizes": {k: len(v) for k, v in self.tick_history.items()},
                "is_healthy": self.is_healthy(),
                "time_since_last_message": round(current_time - self.last_message_time, 1),
                "time_since_last_heartbeat": round(current_time - self.last_heartbeat, 1)
            }

# Example usage (for testing):
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    TOKEN = "085d4dda8f5195556c0dc5c9ebc7d6ac-c-app"
    client = AllTickClient(TOKEN)
    
    def print_tick(data):
        if 'data' in data and 'code' in data['data']:
             tick = data['data']
             print(f"Tick: {tick['code']} Price: {tick.get('bids', [{}])[0].get('price')} Time: {tick.get('tick_time')}")
        else:
            print("Msg:", data)

    client.add_callback(print_tick)
    client.connect()
    time.sleep(2)
    client.subscribe(["700.HK", "AAPL.US"])
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        client.disconnect()
