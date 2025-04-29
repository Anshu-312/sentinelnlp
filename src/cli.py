"""
Command-line interface for the SentinelNLP package.
"""

import os
import sys
import argparse
import logging
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_data(args: argparse.Namespace) -> None:
    """
    Process cybersecurity data using the DataProcessor.
    
    Args:
        args: Command-line arguments.
    """
    try:
        from src.data_processor.data_processor import DataProcessor
    except ImportError:
        from data_processor.data_processor import DataProcessor
    
    processor = DataProcessor(output_dir=args.output_dir)
    
    if args.file:
        logger.info(f"Processing file: {args.file}")
        processor.process_file(args.file, output_name=args.output_name, split=not args.no_split)
    elif args.directory:
        logger.info(f"Processing directory: {args.directory}")
        processor.process_directory(
            args.directory, 
            pattern=args.pattern, 
            recursive=not args.no_recursive
        )
    else:
        logger.error("No input file or directory specified.")
        sys.exit(1)
    
    logger.info("Data processing complete.")

def reorganize_data(args: argparse.Namespace) -> None:
    """
    Reorganize cybersecurity datasets.
    
    Args:
        args: Command-line arguments.
    """
    try:
        from src.data_processor.reorganize_datasets import DatasetReorganizer
    except ImportError:
        from data_processor.reorganize_datasets import DatasetReorganizer
    
    reorganizer = DatasetReorganizer(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        processed_dir=args.processed_dir,
        nlp_datasets_dir=args.nlp_datasets_dir
    )
    
    reorganizer.run(clean_previous=not args.no_clean)
    
    logger.info("Dataset reorganization complete.")

def extract_entities(args: argparse.Namespace) -> None:
    """
    Extract entities from cybersecurity text.
    
    Args:
        args: Command-line arguments.
    """
    logger.info("Entity extraction not yet implemented.")
    # This will be implemented once the NLP pipeline is developed
    # from nlp_pipeline import EntityExtractor
    # extractor = EntityExtractor()
    # extractor.extract(args.input, output_path=args.output)

def extract_relationships(args: argparse.Namespace) -> None:
    """
    Extract relationships from cybersecurity text.
    
    Args:
        args: Command-line arguments.
    """
    logger.info("Relationship extraction not yet implemented.")
    # This will be implemented once the NLP pipeline is developed
    # from nlp_pipeline import RelationExtractor
    # extractor = RelationExtractor()
    # extractor.extract(args.input, output_path=args.output)

def build_knowledge_graph(args: argparse.Namespace) -> None:
    """
    Build a knowledge graph from extracted entities and relationships.
    
    Args:
        args: Command-line arguments.
    """
    logger.info("Knowledge graph construction not yet implemented.")
    # This will be implemented once the knowledge graph component is developed
    # from knowledge_graph import GraphManager
    # graph_manager = GraphManager(uri=args.uri, username=args.username, password=args.password)
    # graph_manager.build(args.input, clear_existing=args.clear)

def start_api_server(args: argparse.Namespace) -> None:
    """
    Start the API server.
    
    Args:
        args: Command-line arguments.
    """
    logger.info("API server not yet implemented.")
    # This will be implemented once the API component is developed
    # from api import APIServer
    # server = APIServer(host=args.host, port=args.port)
    # server.start()

def main(args: Optional[List[str]] = None) -> None:
    """
    Main entry point for the command-line interface.
    
    Args:
        args: Command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="SentinelNLP: Ontology-Based NLP Framework for Cyber Threat Knowledge Representation"
    )
    
    # Add version argument
    parser.add_argument(
        '--version', 
        action='version', 
        version='SentinelNLP 0.1.0'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Process data command
    process_parser = subparsers.add_parser('process', help='Process cybersecurity data')
    process_parser.add_argument('--file', type=str, help='Path to input file')
    process_parser.add_argument('--directory', type=str, help='Path to input directory')
    process_parser.add_argument('--output-dir', type=str, default='processed_data', help='Output directory')
    process_parser.add_argument('--output-name', type=str, help='Base name for output files')
    process_parser.add_argument('--pattern', type=str, default='*.*', help='Glob pattern for matching files')
    process_parser.add_argument('--no-split', action='store_true', help='Do not split data into train/val/test')
    process_parser.add_argument('--no-recursive', action='store_true', help='Do not search directories recursively')
    
    # Reorganize data command
    reorganize_parser = subparsers.add_parser('reorganize', help='Reorganize cybersecurity datasets')
    reorganize_parser.add_argument('--source-dir', type=str, default='data', help='Directory containing source data')
    reorganize_parser.add_argument('--output-dir', type=str, default='organized_data', help='Output directory')
    reorganize_parser.add_argument('--processed-dir', type=str, default='processed_data', help='Directory with processed data')
    reorganize_parser.add_argument('--nlp-datasets-dir', type=str, default='nlp_datasets', help='Directory with NLP datasets')
    reorganize_parser.add_argument('--no-clean', action='store_true', help='Do not clean previous organized datasets')
    
    # Extract entities command
    entity_parser = subparsers.add_parser('extract-entities', help='Extract entities from cybersecurity text')
    entity_parser.add_argument('--input', type=str, required=True, help='Input file or directory')
    entity_parser.add_argument('--output', type=str, help='Output file')
    entity_parser.add_argument('--model', type=str, default='default', help='Entity extraction model to use')
    
    # Extract relationships command
    relation_parser = subparsers.add_parser('extract-relationships', help='Extract relationships from cybersecurity text')
    relation_parser.add_argument('--input', type=str, required=True, help='Input file or directory')
    relation_parser.add_argument('--output', type=str, help='Output file')
    relation_parser.add_argument('--model', type=str, default='default', help='Relationship extraction model to use')
    
    # Build knowledge graph command
    kg_parser = subparsers.add_parser('build-kg', help='Build knowledge graph')
    kg_parser.add_argument('--input', type=str, required=True, help='Input entities and relationships')
    kg_parser.add_argument('--uri', type=str, default='bolt://localhost:7687', help='Neo4j URI')
    kg_parser.add_argument('--username', type=str, default='neo4j', help='Neo4j username')
    kg_parser.add_argument('--password', type=str, default='password', help='Neo4j password')
    kg_parser.add_argument('--clear', action='store_true', help='Clear existing graph before building')
    
    # Start API server command
    api_parser = subparsers.add_parser('api', help='Start API server')
    api_parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to listen on')
    api_parser.add_argument('--port', type=int, default=8000, help='Port to listen on')
    
    # Parse arguments
    parsed_args = parser.parse_args(args)
    
    # Execute command
    if parsed_args.command == 'process':
        process_data(parsed_args)
    elif parsed_args.command == 'reorganize':
        reorganize_data(parsed_args)
    elif parsed_args.command == 'extract-entities':
        extract_entities(parsed_args)
    elif parsed_args.command == 'extract-relationships':
        extract_relationships(parsed_args)
    elif parsed_args.command == 'build-kg':
        build_knowledge_graph(parsed_args)
    elif parsed_args.command == 'api':
        start_api_server(parsed_args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 