import streamlit as st
from neo4j import GraphDatabase
import pandas as pd
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
from scripts.inference_pipeline.ner_predictor import run_ner
from scripts.inference_pipeline.re_predictor import run_re
from scripts.inference_pipeline.export_utils import triples_to_rdf, triples_to_jsonld

# Neo4j config
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "sentinel"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Streamlit UI
st.set_page_config(page_title="SentinelNLP", layout="wide")
st.title("🔐 SentinelNLP")

# Upload report
uploaded_file = st.file_uploader("📤 Upload CTI Report", type=["txt"])
if uploaded_file:
    raw_text = uploaded_file.read().decode("utf-8")
    st.subheader("📄 Raw Report Content")
    st.code(raw_text, language='text')

    # Extract entities and relations
    st.subheader("🔎 Extracted Entities")
    entities = run_ner(raw_text)
    if entities:
        st.json(entities)
    else:
        st.warning("No entities detected in the text.")

    st.subheader("🔗 Extracted Relations")
    # Create entity pairs for relation extraction
    if len(entities) >= 2:
        # Create pairs of adjacent entities
        entity_pairs = [(entities[i]["text"], entities[i+1]["text"]) 
                       for i in range(len(entities)-1)]
        relations = run_re(entity_pairs)
        if relations:
            st.json(relations)
            
            # Generate triples from adjacent entities and their relations
            triples = []
            for i, (e1, e2, rel) in enumerate(relations):
                if rel != "no_relation":  # Only include meaningful relations
                    triples.append((e1, rel, e2))
            
            if triples:
                # Export section
                st.subheader("📤 Export Triples")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Export as RDF"):
                        try:
                            path = triples_to_rdf(triples)
                            st.success(f"RDF saved to {path}")
                        except Exception as e:
                            st.error(f"Error exporting RDF: {str(e)}")
                with col2:
                    if st.button("Export as JSON-LD"):
                        try:
                            path = triples_to_jsonld(triples)
                            st.success(f"JSON-LD saved to {path}")
                        except Exception as e:
                            st.error(f"Error exporting JSON-LD: {str(e)}")
            else:
                st.warning("No meaningful relations detected between entities.")
        else:
            st.warning("No relations detected between entities.")
    else:
        st.warning("Need at least 2 entities to extract relations.")

# Custom Cypher query
st.subheader("🔎 Query the Graph")
query = st.text_area("Enter Cypher Query", "MATCH (a)-[r]->(b) RETURN a, r, b LIMIT 25")
if st.button("Run Query"):
    with driver.session() as session:
        result = session.run(query)
        rows = [dict(record) for record in result]
        st.write(pd.DataFrame(rows))

# Visualize graph
st.subheader("🌐 Knowledge Graph View")

def visualize_neo4j_graph():
    with driver.session() as session:
        result = session.run("MATCH (a)-[r]->(b) RETURN a.id AS source, type(r) AS rel, b.id AS target LIMIT 100")
        edges = [(r["source"], r["target"], r["rel"]) for r in result]

    net = Network(height="500px", bgcolor="#ffffff", font_color="black")
    G = nx.Graph()
    for src, tgt, rel in edges:
        G.add_node(src, label=src)
        G.add_node(tgt, label=tgt)
        G.add_edge(src, tgt, label=rel)

    net.from_nx(G)
    net.save_graph("graph.html")
    HtmlFile = open("graph.html", "r", encoding="utf-8")
    source_code = HtmlFile.read()
    components.html(source_code, height=550, scrolling=True)

if st.button("Show Graph"):
    visualize_neo4j_graph()
