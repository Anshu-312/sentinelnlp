from neo4j import GraphDatabase

uri = "bolt://localhost:7687"
user = "neo4j"
password = "sentinel"

driver = GraphDatabase.driver(uri, auth=(user, password))

with driver.session() as session:
    result = session.run("RETURN 'Connected to Neo4j!' AS message")
    print(result.single()["message"])

driver.close()
