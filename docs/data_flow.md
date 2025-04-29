# SentinelNLP Data Flow

This document describes the flow of data through the SentinelNLP framework.

## Data Flow Diagram

```
┌───────────────────────────────────────────────────────────────────────┐
│                                                                       │
│                          Data Processing                              │
│                                                                       │
│  ┌──────────────┐     ┌──────────────┐      ┌──────────────────────┐  │
│  │              │     │              │      │                      │  │
│  │  Raw Data    │────►│  Cleaning &  │─────►│  Structured Datasets │  │
│  │  Sources     │     │  Organization│      │                      │  │
│  │              │     │              │      │                      │  │
│  └──────────────┘     └──────────────┘      └──────────┬───────────┘  │
│                                                        │              │
└───────────────────────────────────────────────────────┼───────────────┘
                                                        │
                                                        ▼
┌───────────────────────────────────────────────────────────────────────┐
│                                                                       │
│                          NLP Pipeline                                 │
│                                                                       │
│  ┌──────────────┐     ┌──────────────┐      ┌──────────────────────┐  │
│  │              │     │              │      │                      │  │
│  │    Text      │────►│   Entity &   │─────►│ Structured Entities  │  │
│  │Preprocessing │     │  Relation    │      │  & Relationships     │  │
│  │              │     │  Extraction  │      │                      │  │
│  └──────────────┘     └──────┬───────┘      └──────────┬───────────┘  │
│                              │                         │              │
└──────────────────────────────┼─────────────────────────┼───────────────┘
                              │                         │
                              ▼                         │
┌──────────────────────────────────────────┐            │
│                                          │            │
│             Cyber Ontology               │            │
│                                          │            │
│  ┌──────────────────────────────────┐    │            │
│  │                                  │    │            │
│  │  Entity Types & Relationships    │◄───┘            │
│  │                                  │                 │
│  └───────────────┬──────────────────┘                 │
│                  │                                    │
└──────────────────┼────────────────────────────────────┘
                  │                                    
                  │                                    
                  ▼                                    
┌─────────────────────────────────────────────────────┐
│                                                     │
│                 Knowledge Graph                     │
│                                                     │
│  ┌──────────────┐      ┌─────────────────────────┐  │
│  │              │      │                         │  │
│  │  Knowledge   │◄─────┤ Entity & Relationship   │◄─┘
│  │  Storage     │      │ Mapping                 │
│  │              │      │                         │
│  └──────┬───────┘      └─────────────────────────┘
│         │                                         │
└─────────┼─────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────┐
│                                                     │
│                 Application API                     │
│                                                     │
│  ┌──────────────┐      ┌─────────────────────────┐  │
│  │              │      │                         │  │
│  │  API         │◄─────┤ Query Processing &      │  │
│  │  Endpoints   │      │ Response Formatting     │  │
│  │              │      │                         │  │
│  └──────┬───────┘      └─────────────────────────┘  │
│         │                                           │
└─────────┼───────────────────────────────────────────┘
          │
          ▼
    ┌──────────────┐
    │              │
    │  External    │
    │ Applications │
    │              │
    └──────────────┘
```

## Detailed Data Flows

### 1. Data Ingestion & Preprocessing

```
Raw Datasets → DataLoader → DataCleaner → Structured Datasets
```

**Data Types:**
- MITRE ATT&CK techniques and tactics
- CVE vulnerability descriptions
- APT reports and threat intelligence
- STIX/TAXII threat data

**Operations:**
- Text cleaning and normalization
- Entity standardization (names, versions, IDs)
- Data deduplication
- Format standardization (to JSONL/CSV)

### 2. NLP Processing & Entity Extraction

```
Structured Datasets → TextPreprocessor → EntityExtractor → RelationExtractor → Extracted Knowledge
```

**Entity Types:**
- Threat actors
- Malware
- Vulnerabilities
- Attack techniques
- Affected systems
- Indicators of compromise

**Operations:**
- Named entity recognition
- Entity classification
- Relation extraction
- Temporal information extraction

### 3. Ontology Alignment

```
Extracted Knowledge → OntologyManager → EntityMapper → Aligned Knowledge
```

**Alignment Operations:**
- Entity type validation
- Relationship validation
- Hierarchy placement
- Semantic enrichment

### 4. Knowledge Graph Population

```
Aligned Knowledge → GraphManager → EntityNodes & RelationshipEdges → Knowledge Graph
```

**Graph Operations:**
- Node creation
- Edge creation
- Property assignment
- Index maintenance

### 5. API Access

```
User Requests → APIServer → QueryEngine → GraphManager → QueryResults → ResponseFormatter → API Response
```

**API Operations:**
- Query parsing
- Authentication
- Result filtering
- Response formatting

## Data Transformations

### Raw Text to Structured Entities
- Text cleaning (remove HTML, normalize whitespace)
- Sentence segmentation
- Named entity recognition
- Entity classification
- Entity linking (to known identifiers)

### Entities to Knowledge Graph
- Entity mapping to ontology classes
- Relationship mapping to ontology properties
- Property extraction from context
- Temporal information extraction

### Knowledge to API Responses
- Graph query execution
- Result serialization
- JSON/GraphQL formatting
- Pagination handling 