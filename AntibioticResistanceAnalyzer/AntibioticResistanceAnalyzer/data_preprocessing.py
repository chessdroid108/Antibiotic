import os
import logging
import numpy as np
import pandas as pd
from Bio import SeqIO
import json
from io import StringIO

logger = logging.getLogger(__name__)

def preprocess_genomes(genome_data_list):
    """
    Preprocess genome data from various file formats.
    
    Args:
        genome_data_list (list): List of GenomeData objects
        
    Returns:
        dict: Dictionary mapping genome IDs to sequences/features
    """
    sequences = {}
    
    for genome_data in genome_data_list:
        try:
            if genome_data.file_type == 'fasta':
                sequences.update(preprocess_fasta(genome_data.file_path, genome_data.id))
            elif genome_data.file_type == 'csv':
                sequences.update(preprocess_csv(genome_data.file_path, genome_data.id))
            elif genome_data.file_type == 'tsv':
                sequences.update(preprocess_tsv(genome_data.file_path, genome_data.id))
            elif genome_data.file_type == 'json':
                sequences.update(preprocess_json(genome_data.file_path, genome_data.id))
            else:
                logger.warning(f"Unsupported file type: {genome_data.file_type}")
        except Exception as e:
            logger.error(f"Error preprocessing {genome_data.filename}: {str(e)}")
            # Continue with other files if one fails
    
    return sequences

def preprocess_fasta(file_path, genome_id):
    """
    Preprocess FASTA file and extract sequences.
    
    Args:
        file_path (str): Path to FASTA file
        genome_id (int): ID of the genome in the database
        
    Returns:
        dict: Dictionary mapping sequence IDs to sequences
    """
    sequences = {}
    
    for record in SeqIO.parse(file_path, "fasta"):
        seq_id = f"{genome_id}_{record.id}"
        sequences[seq_id] = {
            'sequence': str(record.seq),
            'description': record.description,
            'genome_id': genome_id
        }
    
    return sequences

def preprocess_csv(file_path, genome_id):
    """
    Preprocess CSV file containing genomic data.
    
    Args:
        file_path (str): Path to CSV file
        genome_id (int): ID of the genome in the database
        
    Returns:
        dict: Dictionary mapping sequence IDs to features
    """
    df = pd.read_csv(file_path)
    return _process_tabular_data(df, genome_id)

def preprocess_tsv(file_path, genome_id):
    """
    Preprocess TSV file containing genomic data.
    
    Args:
        file_path (str): Path to TSV file
        genome_id (int): ID of the genome in the database
        
    Returns:
        dict: Dictionary mapping sequence IDs to features
    """
    df = pd.read_csv(file_path, sep='\t')
    return _process_tabular_data(df, genome_id)

def preprocess_json(file_path, genome_id):
    """
    Preprocess JSON file containing genomic data.
    
    Args:
        file_path (str): Path to JSON file
        genome_id (int): ID of the genome in the database
        
    Returns:
        dict: Dictionary mapping sequence IDs to features
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    sequences = {}
    
    # Process JSON based on expected structure
    if isinstance(data, list):
        # List of records
        for i, record in enumerate(data):
            seq_id = f"{genome_id}_{i}"
            if 'sequence' in record:
                sequences[seq_id] = {
                    'sequence': record['sequence'],
                    'genome_id': genome_id
                }
                # Add other fields from the record
                for key, value in record.items():
                    if key != 'sequence':
                        sequences[seq_id][key] = value
            else:
                # If no sequence field, treat entire record as features
                sequences[seq_id] = {
                    'features': record,
                    'genome_id': genome_id
                }
    elif isinstance(data, dict):
        # Dictionary of records
        for key, record in data.items():
            seq_id = f"{genome_id}_{key}"
            if isinstance(record, str):
                # Assume string values are sequences
                sequences[seq_id] = {
                    'sequence': record,
                    'genome_id': genome_id
                }
            else:
                # Process nested dictionary
                sequences[seq_id] = {
                    'genome_id': genome_id
                }
                if 'sequence' in record:
                    sequences[seq_id]['sequence'] = record['sequence']
                else:
                    sequences[seq_id]['features'] = record
    
    return sequences

def _process_tabular_data(df, genome_id):
    """
    Process tabular data (CSV/TSV) into a structured dictionary.
    
    Args:
        df (pandas.DataFrame): DataFrame containing genomic data
        genome_id (int): ID of the genome in the database
        
    Returns:
        dict: Dictionary mapping sequence IDs to features
    """
    sequences = {}
    
    # Check for sequence column
    sequence_column = next((col for col in df.columns if col.lower() == 'sequence'), None)
    
    # Check for ID column
    id_column = next((col for col in df.columns if col.lower() in ['id', 'sequence_id', 'genome_id']), None)
    
    for i, row in df.iterrows():
        # Generate sequence ID
        if id_column:
            seq_id = f"{genome_id}_{row[id_column]}"
        else:
            seq_id = f"{genome_id}_{i}"
        
        sequences[seq_id] = {
            'genome_id': genome_id
        }
        
        # Add sequence if available
        if sequence_column:
            sequences[seq_id]['sequence'] = row[sequence_column]
        
        # Add all other columns as features
        for col in df.columns:
            if col != sequence_column:
                sequences[seq_id][col] = row[col]
    
    return sequences

def find_mutations(resistant_sequences, non_resistant_sequences):
    """
    Find mutations by comparing resistant and non-resistant sequences.
    
    Args:
        resistant_sequences (dict): Dictionary of resistant sequences
        non_resistant_sequences (dict): Dictionary of non-resistant sequences
        
    Returns:
        list: List of identified mutations
    """
    mutations = []
    
    # Extract sequences for comparison
    resistant_seqs = {}
    for seq_id, data in resistant_sequences.items():
        if 'sequence' in data:
            resistant_seqs[seq_id] = data['sequence']
    
    non_resistant_seqs = {}
    for seq_id, data in non_resistant_sequences.items():
        if 'sequence' in data:
            non_resistant_seqs[seq_id] = data['sequence']
    
    # Find reference sequence (use longest sequence as reference)
    if resistant_seqs and non_resistant_seqs:
        all_seqs = list(resistant_seqs.values()) + list(non_resistant_seqs.values())
        reference_seq = max(all_seqs, key=len)
        
        # Compare each sequence to reference
        for seq_id, sequence in {**resistant_seqs, **non_resistant_seqs}.items():
            is_resistant = seq_id in resistant_sequences
            
            # Identify mutations (simplistic approach - assumes sequences are aligned)
            for i, (ref_base, seq_base) in enumerate(zip(reference_seq, sequence)):
                if ref_base != seq_base:
                    mutation = f"pos_{i}_{ref_base}_{seq_base}"
                    mutations.append({
                        'mutation': mutation,
                        'position': i,
                        'reference': ref_base,
                        'alternate': seq_base,
                        'is_resistant': is_resistant,
                        'sequence_id': seq_id
                    })
    
    return mutations

def encode_mutations(resistant_sequences, non_resistant_sequences):
    """
    Encode mutations as binary features for machine learning.
    
    Args:
        resistant_sequences (dict): Dictionary of resistant sequences
        non_resistant_sequences (dict): Dictionary of non-resistant sequences
        
    Returns:
        tuple: (feature_matrix, labels, mutation_mapping)
    """
    # Find all mutations
    mutations = find_mutations(resistant_sequences, non_resistant_sequences)
    
    # Create a set of unique mutations
    unique_mutations = set()
    for mut in mutations:
        unique_mutations.add(mut['mutation'])
    
    # Create mapping from mutation to index
    mutation_mapping = list(unique_mutations)
    
    # Create feature matrix
    all_sequence_ids = list(resistant_sequences.keys()) + list(non_resistant_sequences.keys())
    n_samples = len(all_sequence_ids)
    n_features = len(mutation_mapping)
    
    X = np.zeros((n_samples, n_features), dtype=int)
    y = np.zeros(n_samples, dtype=int)
    
    # Fill in the feature matrix
    for i, seq_id in enumerate(all_sequence_ids):
        # Set label (1 for resistant, 0 for non-resistant)
        if seq_id in resistant_sequences:
            y[i] = 1
        
        # Set features (1 if mutation is present, 0 otherwise)
        for mut in mutations:
            if mut['sequence_id'] == seq_id:
                j = mutation_mapping.index(mut['mutation'])
                X[i, j] = 1
    
    return X, y, mutation_mapping
