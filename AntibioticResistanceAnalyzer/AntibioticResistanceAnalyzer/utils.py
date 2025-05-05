import os
import logging
import scipy.stats as stats
import numpy as np
import zipfile
import tempfile
from werkzeug.utils import secure_filename

logger = logging.getLogger(__name__)

def allowed_file(filename, allowed_extensions):
    """
    Check if a file has an allowed extension.
    
    Args:
        filename (str): Filename to check
        allowed_extensions (set): Set of allowed extensions
        
    Returns:
        bool: True if file is allowed, False otherwise
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def calculate_p_values(feature_vector, labels):
    """
    Calculate p-values for association between features and labels.
    
    Args:
        feature_vector (numpy.ndarray): Binary feature vector
        labels (numpy.ndarray): Binary labels (0/1)
        
    Returns:
        tuple: (p_value, odds_ratio)
    """
    try:
        # Create contingency table
        # [resistant with mutation, non-resistant with mutation]
        # [resistant without mutation, non-resistant without mutation]
        
        resistant_with = np.sum(np.logical_and(feature_vector == 1, labels == 1))
        resistant_without = np.sum(np.logical_and(feature_vector == 0, labels == 1))
        nonresistant_with = np.sum(np.logical_and(feature_vector == 1, labels == 0))
        nonresistant_without = np.sum(np.logical_and(feature_vector == 0, labels == 0))
        
        contingency_table = np.array([
            [resistant_with, nonresistant_with],
            [resistant_without, nonresistant_without]
        ])
        
        # Fisher's exact test
        odds_ratio, p_value = stats.fisher_exact(contingency_table)
        
        return p_value, odds_ratio
    
    except Exception as e:
        logger.error(f"Error calculating p-value: {str(e)}")
        return 1.0, 1.0

def calculate_frequency_difference(feature_vector, labels):
    """
    Calculate frequency difference between resistant and non-resistant strains.
    
    Args:
        feature_vector (numpy.ndarray): Binary feature vector
        labels (numpy.ndarray): Binary labels (0/1)
        
    Returns:
        float: Absolute frequency difference
    """
    try:
        resistant_samples = feature_vector[labels == 1]
        non_resistant_samples = feature_vector[labels == 0]
        
        resistant_freq = np.mean(resistant_samples)
        non_resistant_freq = np.mean(non_resistant_samples)
        
        return abs(resistant_freq - non_resistant_freq)
    
    except Exception as e:
        logger.error(f"Error calculating frequency difference: {str(e)}")
        return 0.0

def save_uploaded_file(file, upload_folder, allowed_extensions):
    """
    Save an uploaded file with a secure filename.
    
    Args:
        file: File object from request.files
        upload_folder (str): Folder to save the file
        allowed_extensions (set): Set of allowed extensions
        
    Returns:
        tuple: (success, filename or error_message)
    """
    if file and allowed_file(file.filename, allowed_extensions):
        filename = secure_filename(file.filename)
        filepath = os.path.join(upload_folder, filename)
        file.save(filepath)
        return True, filepath
    else:
        return False, "Invalid file type"

def create_archive(source_dir, output_file):
    """
    Create a ZIP archive from a directory.
    
    Args:
        source_dir (str): Source directory
        output_file (str): Output ZIP file path
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(source_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, source_dir)
                    zipf.write(file_path, arcname)
        return True
    
    except Exception as e:
        logger.error(f"Error creating archive: {str(e)}")
        return False

def validate_genome_data(genome_data):
    """
    Validate genome data before analysis.
    
    Args:
        genome_data (list): List of GenomeData objects
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not genome_data:
        return False, "No genome data available"
    
    # Check for resistant and non-resistant samples
    resistant_count = sum(1 for g in genome_data if g.is_resistant is True)
    non_resistant_count = sum(1 for g in genome_data if g.is_resistant is False)
    
    if resistant_count == 0:
        return False, "No resistant samples available"
    
    if non_resistant_count == 0:
        return False, "No non-resistant samples available"
    
    return True, ""

def parse_fasta_header(header):
    """
    Parse FASTA header to extract metadata.
    
    Args:
        header (str): FASTA header string
        
    Returns:
        dict: Extracted metadata
    """
    metadata = {'full_header': header}
    
    # Extract ID (first part of header)
    if ' ' in header:
        metadata['id'] = header.split(' ')[0]
    else:
        metadata['id'] = header
    
    # Try to extract other fields (key=value format)
    parts = header.split('|')
    for part in parts:
        if '=' in part:
            key, value = part.split('=', 1)
            metadata[key.strip()] = value.strip()
    
    return metadata
