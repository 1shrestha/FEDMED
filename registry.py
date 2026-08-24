"""In-memory node registry for Week 1.

Week 3 note: this gets backed by Postgres once we add server-restart
recovery. For now it's a simple thread-safe dict so we can prove out
the registration/heartbeat/status flow end to end.
"""

import threading
import time

import node_service_pb2 as node_pb2

HEARTBEAT_SUSPECT_AFTER_SEC = 10   # no heartbeat in this long -> SUSPECT
HEARTBEAT_DEAD_AFTER_SEC = 25      # no heartbeat in this long -> DEAD


class NodeRecord:
    def __init__(self, node_id, hospital_name, address, dataset_size):
        self.node_id = node_id
        self.hospital_name = hospital_name
        self.address = address
        self.dataset_size = dataset_size
        self.last_heartbeat = time.time()

    def current_state(self):
        elapsed = time.time() - self.last_heartbeat
        if elapsed > HEARTBEAT_DEAD_AFTER_SEC:
            return node_pb2.DEAD
        if elapsed > HEARTBEAT_SUSPECT_AFTER_SEC:
            return node_pb2.SUSPECT
        return node_pb2.ACTIVE


class NodeRegistry:
    def __init__(self):
        self._nodes = {}
        self._lock = threading.Lock()

    def register(self, node_id, hospital_name, address, dataset_size):
        with self._lock:
            self._nodes[node_id] = NodeRecord(
                node_id, hospital_name, address, dataset_size
            )

    def heartbeat(self, node_id):
        with self._lock:
            if node_id not in self._nodes:
                return False
            self._nodes[node_id].last_heartbeat = time.time()
            return True

    def snapshot(self):
        """Returns a list of NodeRecord for status reporting."""
        with self._lock:
            return list(self._nodes.values())

    def active_node_ids(self):
        with self._lock:
            return [
                n.node_id for n in self._nodes.values()
                if n.current_state() == node_pb2.ACTIVE
            ]
