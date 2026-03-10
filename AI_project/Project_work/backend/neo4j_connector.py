from neo4j import GraphDatabase
import logging

# ── Connection settings ───────────────────────────────────────────────────────
URI      = "bolt://localhost:7687"   # change if hosted remotely
USERNAME = "neo4j"
PASSWORD = "12345678"           # change to your Neo4j password

# ── Logger ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("neo4j_connection")


# ---------------------------------------------------------------------------
# Neo4j Connection Manager
# ---------------------------------------------------------------------------

class Neo4jConnection:
    """
    Manages a single Neo4j driver instance.
    Use as a context manager (with statement) for automatic disconnect,
    or call connect() / disconnect() manually.

    Usage (context manager — recommended):
        with Neo4jConnection() as conn:
            results = conn.query("MATCH (n) RETURN count(n) AS total")
            print(results)

    Usage (manual):
        conn = Neo4jConnection()
        conn.connect()
        results = conn.query("MATCH (n) RETURN count(n) AS total")
        conn.disconnect()
    """

    def __init__(self, uri=URI, username=USERNAME, password=PASSWORD):
        self.uri      = uri
        self.username = username
        self.password = password
        self._driver  = None

    # ── Connect ──────────────────────────────────────────────────────────────

    def connect(self):
        """Open the driver and verify connectivity."""
        if self._driver is not None:
            log.info("Already connected.")
            return self

        try:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            # Verify the connection is live
            self._driver.verify_connectivity()
            log.info(f"Connected to Neo4j at {self.uri}")
        except Exception as e:
            self._driver = None
            log.error(f"Connection failed: {e}")
            raise
        return self

    # ── Disconnect ───────────────────────────────────────────────────────────

    def disconnect(self):
        """Close the driver and release all connections."""
        if self._driver is None:
            log.info("Already disconnected.")
            return

        try:
            self._driver.close()
            self._driver = None
            log.info("Disconnected from Neo4j.")
        except Exception as e:
            log.error(f"Error during disconnect: {e}")
            raise

    # ── Context manager support ──────────────────────────────────────────────

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        # Return False so exceptions propagate normally
        return False

    # ── Query helpers ────────────────────────────────────────────────────────

    def query(self, cypher: str, parameters: dict = None) -> list:
        """
        Run a read query and return results as a list of dicts.

        Example:
            results = conn.query(
                "MATCH (p:Paper) RETURN p.paper_id AS id, p.title AS title"
            )
            for row in results:
                print(row["id"], row["title"])
        """
        self._check_connected()
        parameters = parameters or {}
        with self._driver.session() as session:
            result = session.run(cypher, parameters)
            return [record.data() for record in result]

    def execute(self, cypher: str, parameters: dict = None):
        """
        Run a write query (CREATE / MERGE / SET / DELETE).

        Example:
            conn.execute(
                "MERGE (p:Paper {paper_id: $pid}) SET p.title = $title",
                {"pid": "P01", "title": "My Paper"}
            )
        """
        self._check_connected()
        parameters = parameters or {}
        with self._driver.session() as session:
            session.run(cypher, parameters)

    def execute_write(self, func, *args, **kwargs):
        """
        Run a transactional write function.

        Example:
            def create_nodes(tx, data):
                tx.run("MERGE (:Paper {paper_id: $id})", {"id": data})
            conn.execute_write(create_nodes, "P01")
        """
        self._check_connected()
        with self._driver.session() as session:
            session.execute_write(func, *args, **kwargs)

    def is_connected(self) -> bool:
        """Return True if the driver is active."""
        return self._driver is not None

    def _check_connected(self):
        if self._driver is None:
            raise ConnectionError(
                "Not connected to Neo4j. Call connect() first."
            )


# ---------------------------------------------------------------------------
# Convenience functions for quick one-off use
# ---------------------------------------------------------------------------

def get_connection() -> Neo4jConnection:
    """Return an already-connected Neo4jConnection instance."""
    conn = Neo4jConnection()
    conn.connect()
    return conn


def test_connection():
    """Quick test — prints node count and disconnects."""
    with Neo4jConnection() as conn:
        result = conn.query("MATCH (n) RETURN count(n) AS total")
        total  = result[0]["total"] if result else 0
        print(f"Connection successful. Total nodes in graph: {total}")


# ---------------------------------------------------------------------------
# Main — demonstrates both usage patterns
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 55)
    print("Pattern 1: Context manager (auto disconnect)")
    print("=" * 55)
    with Neo4jConnection() as conn:
        print(f"Connected : {conn.is_connected()}")

        # Sample queries
        papers = conn.query(
            "MATCH (p:Paper) RETURN p.paper_id AS id, p.title AS title"
        )
        print(f"Papers in graph : {len(papers)}")
        for p in papers:
            print(f"  {p['id']} — {p['title']}")

        entities = conn.query(
            "MATCH (e:Entity) RETURN e.type AS type, count(e) AS count "
            "ORDER BY count DESC LIMIT 5"
        )
        print(f"\nTop entity types:")
        for e in entities:
            print(f"  {e['type']:<20} {e['count']}")

    print(f"After 'with' block — driver closed automatically.")

    print()
    print("=" * 55)
    print("Pattern 2: Manual connect / disconnect")
    print("=" * 55)
    conn = Neo4jConnection()
    conn.connect()
    print(f"Connected : {conn.is_connected()}")

    result = conn.query("MATCH (n) RETURN count(n) AS total")
    print(f"Total nodes : {result[0]['total']}")

    conn.disconnect()
    print(f"Connected : {conn.is_connected()}")