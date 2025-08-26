import time
from dataclasses import dataclass, field

@dataclass
class AppState:
    """A simple dataclass to hold the shared state of the application."""
    # We use time.time() which is wall-clock time. For checking intervals, it's sufficient.
    last_spot_ws_message_time: float = field(default_factory=time.time)
    last_futures_ws_message_time: float = field(default_factory=time.time)
    last_open_interest_update_time: float = field(default_factory=time.time)
    last_depth_snapshot_update_time: float = field(default_factory=time.time)
    db_connected: bool = False

    def is_healthy(self, max_delay_seconds: int = 300) -> (bool, str):
        """
        Checks if all components are healthy.
        A component is considered stale if its last update time is older than max_delay_seconds.
        """
        now = time.time()
        if not self.db_connected:
            return False, "Database is not connected."

        if (now - self.last_spot_ws_message_time) > max_delay_seconds:
            return False, "Spot WebSocket seems stale."

        if (now - self.last_futures_ws_message_time) > max_delay_seconds:
            return False, "Futures WebSocket seems stale."

        if (now - self.last_open_interest_update_time) > max_delay_seconds:
            return False, "Open Interest poller seems stale."

        # Depth snapshot interval is 1 min, so we give it a bit more buffer
        if (now - self.last_depth_snapshot_update_time) > (max_delay_seconds + 60):
             return False, "Depth Snapshot poller seems stale."

        return True, "All components are healthy."


# A single global instance of the application state
app_state = AppState()
