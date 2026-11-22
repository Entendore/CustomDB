import os
import json
import time
import re
import random
import threading
from typing import Any, Dict, List
from threading import Lock


class InMemoryDocumentDB:
    def __init__(self):
        self.collections: Dict[str, List[Dict[str, Any]]] = {}

    def create_collection(self, name: str):
        if name not in self.collections:
            self.collections[name] = []

    def insert(self, collection: str, document: Dict[str, Any]):
        if collection not in self.collections:
            raise ValueError(f"Collection '{collection}' does not exist.")
        self.collections[collection].append(document)

    def find(self, collection: str, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        if collection not in self.collections:
            return []
        return [doc for doc in self.collections[collection] if self._match(doc, query)]

    def update(self, collection: str, query: Dict[str, Any], update: Dict[str, Any]):
        for doc in self.find(collection, query):
            doc.update(update)

    def delete(self, collection: str, query: Dict[str, Any]):
        if collection in self.collections:
            self.collections[collection] = [
                doc for doc in self.collections[collection] if not self._match(doc, query)
            ]

    def aggregate_avg(self, collection: str, field: str) -> float:
        docs = self.collections.get(collection, [])
        values = [doc[field] for doc in docs if field in doc]
        return sum(values) / len(values) if values else 0.0

    def _match(self, doc: Dict[str, Any], query: Dict[str, Any]) -> bool:
        for key, value in query.items():
            if isinstance(value, dict):
                if "$gt" in value and not doc.get(key, None) > value["$gt"]:
                    return False
                if "$lt" in value and not doc.get(key, None) < value["$lt"]:
                    return False
                if "$regex" in value and not re.search(value["$regex"], str(doc.get(key, ""))):
                    return False
            elif doc.get(key) != value:
                return False
        return True


class DurableDocumentDB(InMemoryDocumentDB):
    def __init__(self, node_id: str):
        super().__init__()
        self.node_id = node_id
        self._wal_file = f"{node_id}.wal"
        self._last_op_id = 0
        self._lock = Lock()
        self._replay_wal()

    def _write_wal(self, op: Dict[str, Any]):
        with self._lock:
            self._last_op_id += 1
            op["op_id"] = self._last_op_id
            with open(self._wal_file, "a") as f:
                f.write(json.dumps(op) + "\n")

    def _replay_wal(self):
        if not os.path.exists(self._wal_file):
            return
        with open(self._wal_file, "r") as f:
            for line in f:
                op = json.loads(line)
                if op["op"] == "create_collection":
                    self.create_collection(op["collection"])
                elif op["op"] == "insert":
                    super().insert(op["collection"], op["document"])
                elif op["op"] == "update":
                    self.update(op["collection"], op["filter"], op["update"])
                elif op["op"] == "delete":
                    self.delete(op["collection"], op["filter"])

    def create_collection(self, name: str):
        super().create_collection(name)
        self._write_wal({"op": "create_collection", "collection": name})

    def insert(self, collection: str, document: Dict[str, Any]):
        super().insert(collection, document)
        self._write_wal({"op": "insert", "collection": collection, "document": document})

    def update(self, collection: str, query: Dict[str, Any], update: Dict[str, Any]):
        super().update(collection, query, update)
        self._write_wal({"op": "update", "collection": collection, "filter": query, "update": update})

    def delete(self, collection: str, query: Dict[str, Any]):
        super().delete(collection, query)
        self._write_wal({"op": "delete", "collection": collection, "filter": query})


class Node:
    current_leader: str = None
    leader_lock = Lock()

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.db = DurableDocumentDB(node_id)
        self.peers: Dict[str, "Node"] = {}
        self.is_alive = True
        self.last_heartbeat = time.time()
        self.heartbeat_interval = 1.0
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()

    def _heartbeat_loop(self):
        while self.is_alive:
            time.sleep(self.heartbeat_interval)
            if self.is_leader():
                self.last_heartbeat = time.time()
                for peer in self.peers.values():
                    peer.receive_heartbeat(self.node_id)
            else:
                if time.time() - self.last_heartbeat > 3 * self.heartbeat_interval:
                    print(f"[{self.node_id}] Leader {Node.current_leader} missed heartbeat. Starting election.")
                    self.start_election()

    def receive_heartbeat(self, from_node: str):
        if from_node == Node.current_leader:
            self.last_heartbeat = time.time()

    def start_election(self):
        with Node.leader_lock:
            new_leader = random.choice(list(self.peers.keys()) + [self.node_id])
            Node.current_leader = new_leader
            print(f"[{self.node_id}] Elected new leader: {Node.current_leader}")

    def is_leader(self):
        with Node.leader_lock:
            return Node.current_leader == self.node_id

    def add_peer(self, peer: "Node"):
        self.peers[peer.node_id] = peer

    def replicate(self, op: Dict[str, Any]):
        for peer in self.peers.values():
            try:
                if op["op"] == "create_collection":
                    peer.db.create_collection(op["collection"])
                elif op["op"] == "insert":
                    peer.db.insert(op["collection"], op["document"])
                elif op["op"] == "update":
                    peer.db.update(op["collection"], op["filter"], op["update"])
                elif op["op"] == "delete":
                    peer.db.delete(op["collection"], op["filter"])
            except Exception as e:
                print(f"[{self.node_id}] Replication to {peer.node_id} failed: {e}")


def example_usage():
    node1 = Node("node1")
    node2 = Node("node2")
    node3 = Node("node3")

    # Form the cluster
    for n1 in [node1, node2, node3]:
        for n2 in [node1, node2, node3]:
            if n1 != n2:
                n1.add_peer(n2)

    with Node.leader_lock:
        Node.current_leader = "node3"
    print(f"Initial leader is: {Node.current_leader}")

    leader = {"node1": node1, "node2": node2, "node3": node3}[Node.current_leader]

    # Example operations
    if leader.is_leader():
        print("[Leader] Creating collection and inserting users")
        leader.db.create_collection("users")
        leader.replicate({"op": "create_collection", "collection": "users"})

        for user in [
            {"id": 1, "name": "Alice", "age": 30},
            {"id": 2, "name": "Bob", "age": 25},
            {"id": 3, "name": "Charlie", "age": 35},
        ]:
            leader.db.insert("users", user)
            leader.replicate({"op": "insert", "collection": "users", "document": user})

    time.sleep(2)

    print("\nQuery from node2 (age > 26):")
    for doc in node2.db.find("users", {"age": {"$gt": 26}}):
        print(doc)

    print("\nAggregation from node1 (avg age):")
    print(node1.db.aggregate_avg("users", "age"))

    # Shutdown simulation
    for node in [node1, node2, node3]:
        node.is_alive = False
        node.heartbeat_thread.join()


if __name__ == "__main__":
    example_usage()
