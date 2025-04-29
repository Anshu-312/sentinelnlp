# SentinelNLP Ontology Directory

This directory contains the OWL/RDF ontology files and related documentation for the SentinelNLP project.

## Purpose

The ontologies in this directory serve as the knowledge foundation for the SentinelNLP system. They define:

- Concepts/classes for entity types that can be recognized
- Properties/relationships between entity types
- Semantic constraints and rules for knowledge validation
- Domain-specific terminology and hierarchies

## Directory Structure

```
ontology/
├── core/                   # Core ontology files
│   ├── sentinelnlp_core.owl   # Base ontology with fundamental concepts
│   └── nlp_elements.owl       # NLP-specific constructs
│
├── domains/                # Domain-specific ontologies
│   ├── general/            # General domain ontologies
│   ├── cybersecurity/      # Cybersecurity domain
│   ├── financial/          # Financial domain
│   └── medical/            # Medical domain
│
├── mappings/               # Ontology alignment and mapping files
│   ├── dbpedia_mapping.ttl # Mappings to DBpedia
│   └── wikidata_mapping.ttl# Mappings to Wikidata
│
├── schemas/                # Validation schemas
│   └── shacl/              # SHACL constraint definitions
│
└── tools/                  # Ontology management scripts
    ├── convert.py          # Format conversion utilities
    ├── validate.py         # Ontology validation
    └── visualize.py        # Ontology visualization tools
```

## Ontology Development

### Guidelines

When developing or extending ontologies for SentinelNLP:

1. Maintain compatibility with the core ontology
2. Follow standard OWL/RDF best practices
3. Add comprehensive annotations for better entity matching
4. Document all classes, properties, and axioms
5. Validate against SHACL constraints before committing

### Tools

Recommended tools for ontology development:

- [Protégé](https://protege.stanford.edu/) - Ontology editor
- [WebVOWL](http://vowl.visualdataweb.org/webvowl.html) - Visualization
- [ROBOT](http://robot.obolibrary.org/) - Command line ontology tool

## Core Ontology Elements

The core SentinelNLP ontology defines the following top-level classes:

- `Entity` - Base class for all recognized entities
  - `PhysicalEntity` - Tangible objects
  - `AbstractEntity` - Concepts, ideas, etc.
  - `Agent` - Entities capable of actions
    - `Person`
    - `Organization`
  - `Location`
  - `Event`
  - `Information`

And key relationship types:

- `hasRelationWith` - Base object property
  - `partOf`
  - `locatedIn`
  - `participatesIn`
  - `createdBy`
  - `associatedWith`

## Usage in SentinelNLP

The system uses these ontologies to:

1. Guide entity recognition by mapping extracted entities to ontology classes
2. Validate relationships based on domain constraints
3. Infer new knowledge through ontological reasoning
4. Provide structured query capabilities over extracted information

## External Ontology Integration

SentinelNLP supports integration with external ontologies:

1. Place external ontology files in the appropriate domain directory
2. Create mapping files in the `mappings/` directory
3. Update configuration to include the new ontologies

## License Information

Unless otherwise specified, ontologies in this directory are licensed under the same terms as the main project. External ontologies may have their own licensing terms, which are included in their respective files. 