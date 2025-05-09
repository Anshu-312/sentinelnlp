import json
from rdflib import Graph, Namespace, URIRef, Literal, RDF, XSD
from typing import List, Tuple, Union, Dict, Any
import os

def validate_triple(triple: Tuple[str, str, str]) -> bool:
    """Validate a triple tuple."""
    if not isinstance(triple, tuple) or len(triple) != 3:
        return False
    s, p, o = triple
    return all(isinstance(x, str) and x.strip() for x in (s, p, o))

def triples_to_rdf(triples: List[Tuple[str, str, str]], 
                  file_path: str = "output.rdf",
                  namespace: str = "http://example.org/cyber#") -> str:
    """
    Convert triples to RDF format and save to file.
    
    Args:
        triples: List of (subject, predicate, object) tuples
        file_path: Output file path
        namespace: RDF namespace URI
    
    Returns:
        Path to the saved file
    """
    if not triples:
        raise ValueError("No triples provided")
    
    # Validate all triples
    invalid_triples = [t for t in triples if not validate_triple(t)]
    if invalid_triples:
        raise ValueError(f"Invalid triples found: {invalid_triples}")
    
    g = Graph()
    NS = Namespace(namespace)
    g.bind("cyber", NS)

    for s, p, o in triples:
        # Create URIs for subjects and predicates
        subject = URIRef(NS[s])
        predicate = URIRef(NS[p])
        
        # Try to create appropriate object type
        try:
            # Check if object is a number
            if o.isdigit():
                obj = Literal(int(o), datatype=XSD.integer)
            elif o.replace('.', '', 1).isdigit():  # Check if float
                obj = Literal(float(o), datatype=XSD.float)
            else:
                obj = URIRef(NS[o])
        except:
            obj = URIRef(NS[o])
        
        g.add((subject, predicate, obj))

    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Use absolute path for output file
    abs_file_path = os.path.join(output_dir, os.path.basename(file_path))
    
    # Save the graph
    g.serialize(destination=abs_file_path, format='xml')
    print(f"[✓] RDF file saved to {abs_file_path}")
    return abs_file_path

def triples_to_jsonld(triples: List[Tuple[str, str, str]], 
                     file_path: str = "output.jsonld",
                     context: Dict[str, Any] = None) -> str:
    """
    Convert triples to JSON-LD format and save to file.
    
    Args:
        triples: List of (subject, predicate, object) tuples
        file_path: Output file path
        context: Custom JSON-LD context
    
    Returns:
        Path to the saved file
    """
    if not triples:
        raise ValueError("No triples provided")
    
    # Validate all triples
    invalid_triples = [t for t in triples if not validate_triple(t)]
    if invalid_triples:
        raise ValueError(f"Invalid triples found: {invalid_triples}")
    
    # Default context if none provided
    if context is None:
        context = {
            "@vocab": "http://example.org/cyber#",
            "@base": "http://example.org/cyber/"
        }
    
    # Group triples by subject
    subjects = {}
    for s, p, o in triples:
        if s not in subjects:
            subjects[s] = {"@id": s}
        subjects[s][p] = o
    
    # Create JSON-LD structure
    data = {
        "@context": context,
        "@graph": list(subjects.values())
    }
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Use absolute path for output file
    abs_file_path = os.path.join(output_dir, os.path.basename(file_path))
    
    # Save to file
    with open(abs_file_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"[✓] JSON-LD file saved to {abs_file_path}")
    return abs_file_path

if __name__ == "__main__":
    # Test the functions
    test_triples = [
        ("APT29", "uses", "Mimikatz"),
        ("Emotet", "exploits", "CVE-2021-34527"),
        ("Emotet", "targets", "Microsoft")
    ]
    
    try:
        # Test RDF export
        rdf_path = triples_to_rdf(test_triples, "test_output.rdf")
        
        # Test JSON-LD export
        jsonld_path = triples_to_jsonld(test_triples, "test_output.jsonld")
        
    except Exception as e:
        print(f"[✗] Error during export: {str(e)}")
