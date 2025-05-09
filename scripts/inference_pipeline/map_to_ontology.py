import json
import os
from fuzzywuzzy import fuzz
from collections import defaultdict

# Load ontology objects
def load_ontology_objects(json_files):
    ontology = defaultdict(list)
    for file in json_files:
        if not os.path.exists(file):
            print(f"⚠️ File not found: {file}")
            continue
        with open(file, 'r') as f:
            data = json.load(f)
            if 'entity_types' in data:
                for entity_type in data['entity_types']:
                    ontology['entities'].append(entity_type)
            elif 'relation_types' in data:
                for relation_type in data['relation_types']:
                    ontology['relations'].append(relation_type)
    return ontology

# Match a string to ontology
def map_entity(entity_text, entity_type):
    # For now, we'll just return a structured format with the entity and its type
    return {
        "text": entity_text,
        "type": entity_type
    }

# Main mapping function
def map_triples(triples_data, entity_labels, ontology):
    mapped_triples = []
    # Extract triples from JSON-LD format
    triples = triples_data.get('@graph', [])
    # Get the first entity label mapping (since it's a list with one item)
    entity_map = entity_labels[0] if entity_labels else {}
    
    for s, p, o in triples:
        subj = map_entity(s, entity_map.get(s, "UNKNOWN"))
        obj = map_entity(o, entity_map.get(o, "UNKNOWN"))
        mapped_triple = {
            "subject": subj,
            "predicate": p,
            "object": obj
        }
        mapped_triples.append(mapped_triple)
    return mapped_triples

# Entry point
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--triples", required=True, help="Path to extracted triples file (JSON)")
    parser.add_argument("--entity_labels", required=True, help="Path to entity label mapping JSON")
    parser.add_argument("--ontology_files", nargs='+', required=True, help="List of ontology json files")
    parser.add_argument("--output", default="mapped_triples.json", help="Output file")
    args = parser.parse_args()

    with open(args.triples) as f:
        triples = json.load(f)

    with open(args.entity_labels) as f:
        entity_labels = json.load(f)

    ontology = load_ontology_objects(args.ontology_files)

    print(f"[•] Mapping {len(triples.get('@graph', []))} triples...")
    mapped = map_triples(triples, entity_labels, ontology)

    with open(args.output, "w") as f:
        json.dump({
            "triples": mapped,
            "metadata": {
                "num_triples": len(mapped),
                "ontology_files": args.ontology_files
            }
        }, f, indent=2)

    print(f"[✓] Mapped triples saved to {args.output}")
