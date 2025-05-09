from neo4j import GraphDatabase
import json
import argparse
import sys

class Neo4jLoader:

    def __init__(self, uri, user, password):
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            # Test the connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            print("[✓] Successfully connected to Neo4j database")
        except Exception as e:
            print(f"[✗] Failed to connect to Neo4j database: {str(e)}")
            print("\nPlease make sure that:")
            print("1. Neo4j is installed on your machine")
            print("2. Neo4j service is running")
            print("3. The connection details (URI, username, password) are correct")
            print("\nTo start Neo4j:")
            print("- On Windows: Open Neo4j Desktop and start your database")
            print("- On Linux: Run 'sudo service neo4j start'")
            print("- On macOS: Run 'brew services start neo4j'")
            sys.exit(1)

    def close(self):
        self.driver.close()

    def load_triples(self, data):
        triples = data.get('triples', [])
        with self.driver.session() as session:
            for triple in triples:
                try:
                    session.execute_write(self._merge_triple, triple)
                except Exception as e:
                    print(f"[✗] Failed to load triple: {triple}")
                    print(f"Error: {str(e)}")

    @staticmethod
    def _merge_triple(tx, triple):
        subj = triple['subject']
        pred = triple['predicate']
        obj = triple['object']

        # Create nodes with their types and properties
        tx.run(f"""
            MERGE (s:{subj['type']} {{id: $subj_id, text: $subj_text}})
            MERGE (o:{obj['type']} {{id: $obj_id, text: $obj_text}})
            MERGE (s)-[r:{pred.upper()}]->(o)
        """, 
        subj_id=subj['text'],
        subj_text=subj['text'],
        obj_id=obj['text'],
        obj_text=obj['text'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--triples", required=True, help="Path to mapped triples JSON")
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j URI")
    parser.add_argument("--user", default="neo4j", help="Neo4j Username")
    parser.add_argument("--password", required=True, help="Neo4j Password")
    args = parser.parse_args()

    try:
        with open(args.triples) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[✗] Could not find file: {args.triples}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"[✗] Invalid JSON format in file: {args.triples}")
        sys.exit(1)

    loader = Neo4jLoader(args.uri, args.user, args.password)
    loader.load_triples(data)
    loader.close()

    print(f"[✓] Loaded {len(data.get('triples', []))} triples into Neo4j.")
