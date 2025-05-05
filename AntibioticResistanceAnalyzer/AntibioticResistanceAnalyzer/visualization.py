import logging
import numpy as np
import requests
import tempfile
import os
import json
import io
import base64
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environment
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import dendrogram, linkage

logger = logging.getLogger(__name__)

def generate_visualizations(mutations, model_performance, feature_importance):
    """
    Generate visualization data for antibiotic resistance mutations for the web interface.
    
    Args:
        mutations (list): List of identified mutations
        model_performance (dict): Model performance metrics
        feature_importance (list): Feature importance scores
        
    Returns:
        dict: Visualization data including charts and images
    """
    visualizations = {}
    
    try:
        # Generate frequency comparison data
        visualizations['frequency_comparison'] = _generate_frequency_comparison(mutations)
        
        # Generate ROC curve data and image
        roc_data = _generate_roc_curve_data(model_performance)
        visualizations['roc_curve'] = roc_data
        
        # If model performance already includes visualizations, use those
        if 'visualizations' in model_performance:
            mp_vis = model_performance['visualizations']
            if 'roc_curve' in mp_vis and mp_vis['roc_curve']:
                visualizations['roc_curve_image'] = mp_vis['roc_curve']
            if 'confusion_matrix' in mp_vis and mp_vis['confusion_matrix']:
                visualizations['confusion_matrix_image'] = mp_vis['confusion_matrix']
        
        # Generate feature importance chart data and visualization
        visualizations['feature_importance'] = _generate_feature_importance_chart(feature_importance)
        feature_imp_img = _create_feature_importance_image(feature_importance)
        if feature_imp_img:
            visualizations['feature_importance_image'] = feature_imp_img
        
        # Generate mutation distribution data
        visualizations['mutation_distribution'] = _generate_mutation_distribution(mutations)
        gene_dist_img = _create_gene_distribution_image(mutations)
        if gene_dist_img:
            visualizations['gene_distribution_image'] = gene_dist_img
        
        # Generate mutation clustering for related mutations
        cluster_img = _create_mutation_clustering_image(mutations)
        if cluster_img:
            visualizations['mutation_clustering_image'] = cluster_img
        
        # Generate p-value distribution
        pval_img = _create_pvalue_distribution_image(mutations)
        if pval_img:
            visualizations['pvalue_distribution_image'] = pval_img
        
        # Generate heatmap of mutation patterns
        heatmap_img = _create_mutation_heatmap_image(mutations)
        if heatmap_img:
            visualizations['mutation_heatmap_image'] = heatmap_img
        
        # Generate confusion matrix visualization
        if 'confusion_matrix' in model_performance:
            visualizations['confusion_matrix'] = model_performance['confusion_matrix']
        
    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")
        visualizations['error'] = str(e)
    
    return visualizations

def _generate_frequency_comparison(mutations):
    """
    Generate data for frequency comparison chart.
    
    Args:
        mutations (list): List of identified mutations
        
    Returns:
        dict: Chart data
    """
    # Select top 20 most significant mutations
    significant_mutations = [m for m in mutations if m.get('significant', False)]
    top_mutations = significant_mutations[:20] if len(significant_mutations) > 20 else significant_mutations
    
    if not top_mutations and mutations:
        # If no significant mutations, use mutations with largest frequency difference
        mutations_sorted = sorted(mutations, key=lambda x: abs(x.get('resistant_freq', 0) - x.get('non_resistant_freq', 0)), reverse=True)
        top_mutations = mutations_sorted[:20]
    
    # Prepare chart data
    labels = [m.get('mutation', f"Mutation_{i}") for i, m in enumerate(top_mutations)]
    resistant_freqs = [m.get('resistant_freq', 0) for m in top_mutations]
    non_resistant_freqs = [m.get('non_resistant_freq', 0) for m in top_mutations]
    
    return {
        'labels': labels,
        'resistant_frequencies': resistant_freqs,
        'non_resistant_frequencies': non_resistant_freqs
    }

def _generate_roc_curve_data(model_performance):
    """
    Generate ROC curve data.
    
    Args:
        model_performance (dict): Model performance metrics
        
    Returns:
        dict: ROC curve data
    """
    # For a proper ROC curve, we would need the actual predictions and true labels
    # Here we'll create a simple representation based on the AUC value
    
    roc_auc = model_performance.get('roc_auc', 0.5)
    
    # Generate approximate ROC curve points based on AUC
    # This is a very simplified approach
    if roc_auc > 0.5:
        # Better than random
        x = np.linspace(0, 1, 100)
        # Approximate ROC curve shape
        y = x ** (1 / (2 * roc_auc))
    else:
        # Random or worse
        x = np.linspace(0, 1, 100)
        y = x
    
    return {
        'fpr': x.tolist(),
        'tpr': y.tolist(),
        'auc': roc_auc
    }

def _generate_feature_importance_chart(feature_importance):
    """
    Generate feature importance chart data.
    
    Args:
        feature_importance (list): Feature importance scores
        
    Returns:
        dict: Chart data
    """
    # Select top 15 features
    top_features = feature_importance[:15] if len(feature_importance) > 15 else feature_importance
    
    # Prepare chart data
    labels = [f.get('feature', f"Feature_{i}") for i, f in enumerate(top_features)]
    values = [f.get('importance', 0) for f in top_features]
    
    return {
        'labels': labels,
        'values': values
    }

def _generate_mutation_distribution(mutations):
    """
    Generate mutation distribution data.
    
    Args:
        mutations (list): List of identified mutations
        
    Returns:
        dict: Chart data
    """
    # Count mutations by gene
    gene_counts = {}
    for mutation in mutations:
        gene = mutation.get('gene', 'Unknown')
        if gene not in gene_counts:
            gene_counts[gene] = 0
        gene_counts[gene] += 1
    
    # Sort genes by number of mutations
    sorted_genes = sorted(gene_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Select top 10 genes
    top_genes = sorted_genes[:10] if len(sorted_genes) > 10 else sorted_genes
    
    # Prepare chart data
    labels = [g[0] for g in top_genes]
    values = [g[1] for g in top_genes]
    
    return {
        'labels': labels,
        'values': values
    }

def map_mutations_to_structure(mutations):
    """
    Map mutations to 3D protein structures using PDB.
    
    Args:
        mutations (list): List of identified mutations
        
    Returns:
        dict: Structure mapping data
    """
    structure_mappings = {}
    
    try:
        # Select significant mutations for mapping
        significant_mutations = [m for m in mutations if m.get('significant', False)]
        
        # Group mutations by gene
        genes = {}
        for mutation in significant_mutations:
            gene = mutation.get('gene', 'Unknown')
            if gene not in genes:
                genes[gene] = []
            genes[gene].append(mutation)
        
        # Attempt to map each gene to a PDB structure
        for gene, gene_mutations in genes.items():
            # Skip unknown genes
            if gene == 'Unknown':
                continue
            
            # Query PDB for this gene
            pdb_data = _query_pdb_for_gene(gene)
            
            if pdb_data:
                structure_mappings[gene] = {
                    'pdb_id': pdb_data.get('pdb_id'),
                    'mutations': [
                        {
                            'mutation': m.get('mutation'),
                            'position': m.get('position'),
                            'significance': m.get('p_value', 1.0)
                        } for m in gene_mutations
                    ]
                }
    
    except Exception as e:
        logger.error(f"Error mapping mutations to structure: {str(e)}")
    
    return structure_mappings

def _create_feature_importance_image(feature_importance):
    """
    Create a bar chart visualization of feature importance.
    
    Args:
        feature_importance (list): Feature importance scores
        
    Returns:
        str: Base64-encoded image or None if failed
    """
    try:
        # Select top 15 features
        top_features = feature_importance[:15] if len(feature_importance) > 15 else feature_importance
        
        # Extract feature names and importance values
        if top_features and 'xgb_importance' in top_features[0]:
            # Use XGBoost importance if available
            labels = [f.get('name', f.get('feature', f"Feature_{i}")) for i, f in enumerate(top_features)]
            values = [f.get('xgb_importance', 0) for f in top_features]
        else:
            # Fall back to basic importance
            labels = [f.get('feature', f"Feature_{i}") for i, f in enumerate(top_features)]
            values = [f.get('importance', 0) for f in top_features]
        
        # Create horizontal bar chart
        plt.figure(figsize=(10, 8))
        y_pos = np.arange(len(labels))
        plt.barh(y_pos, values, align='center')
        plt.yticks(y_pos, labels)
        plt.xlabel('Importance')
        plt.title('Feature Importance for Antibiotic Resistance Prediction')
        plt.tight_layout()
        
        # Save to base64 string
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', dpi=100)
        plt.close()
        img_buf.seek(0)
        return base64.b64encode(img_buf.read()).decode('utf-8')
    
    except Exception as e:
        logger.error(f"Error creating feature importance image: {str(e)}")
        return None

def _create_gene_distribution_image(mutations):
    """
    Create a chart showing the distribution of mutations across genes.
    
    Args:
        mutations (list): List of identified mutations
        
    Returns:
        str: Base64-encoded image or None if failed
    """
    try:
        # Count mutations by gene
        gene_counts = {}
        for mutation in mutations:
            gene = mutation.get('gene', 'Unknown')
            if gene not in gene_counts:
                gene_counts[gene] = 0
            gene_counts[gene] += 1
        
        # Sort genes by number of mutations
        sorted_genes = sorted(gene_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Select top 10 genes
        top_genes = sorted_genes[:10] if len(sorted_genes) > 10 else sorted_genes
        
        # Prepare data for plotting
        labels = [g[0] for g in top_genes]
        values = [g[1] for g in top_genes]
        
        # Count significant mutations by gene
        sig_counts = {}
        for mutation in mutations:
            if mutation.get('significant', False):
                gene = mutation.get('gene', 'Unknown')
                if gene not in sig_counts:
                    sig_counts[gene] = 0
                sig_counts[gene] += 1
        
        # Get significant counts for the top genes
        sig_values = [sig_counts.get(g[0], 0) for g in top_genes]
        
        # Create stacked bar chart
        plt.figure(figsize=(10, 6))
        
        # Plot bars
        bars1 = plt.bar(labels, sig_values, color='r', alpha=0.7)
        bars2 = plt.bar(labels, [v - s for v, s in zip(values, sig_values)], bottom=sig_values, color='b', alpha=0.7)
        
        # Add labels and legend
        plt.xlabel('Gene')
        plt.ylabel('Number of Mutations')
        plt.title('Distribution of Mutations Across Genes')
        plt.xticks(rotation=45, ha='right')
        plt.legend([bars1[0], bars2[0]], ['Significant Mutations', 'Other Mutations'])
        plt.tight_layout()
        
        # Save to base64 string
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', dpi=100)
        plt.close()
        img_buf.seek(0)
        return base64.b64encode(img_buf.read()).decode('utf-8')
    
    except Exception as e:
        logger.error(f"Error creating gene distribution image: {str(e)}")
        return None

def _create_mutation_clustering_image(mutations):
    """
    Create a dendrogram showing clustering of related mutations.
    
    Args:
        mutations (list): List of identified mutations
        
    Returns:
        str: Base64-encoded image or None if failed
    """
    try:
        # Filter for significant mutations
        significant_mutations = [m for m in mutations if m.get('significant', False)]
        
        # Need at least 2 mutations to cluster
        if len(significant_mutations) < 2:
            return None
        
        # Extract features for clustering (frequency differences and p-values)
        X = np.array([
            [
                m.get('freq_diff', 0), 
                -np.log10(max(m.get('p_value', 1.0), 1e-10))  # -log10(p-value) for better scaling
            ] 
            for m in significant_mutations
        ])
        
        # Normalize features
        X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        
        # Perform hierarchical clustering
        Z = linkage(X_norm, 'ward')
        
        # Plot dendrogram
        plt.figure(figsize=(12, 7))
        plt.title('Hierarchical Clustering of Antibiotic Resistance Mutations')
        
        # Extract mutation names as labels
        labels = [m.get('mutation', f"Mutation_{i}") for i, m in enumerate(significant_mutations)]
        
        # Create dendrogram
        dendrogram(
            Z,
            labels=labels,
            leaf_rotation=90.,
            leaf_font_size=10.,
            color_threshold=0.7*max(Z[:,2])
        )
        
        plt.tight_layout()
        
        # Save to base64 string
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', dpi=100)
        plt.close()
        img_buf.seek(0)
        return base64.b64encode(img_buf.read()).decode('utf-8')
    
    except Exception as e:
        logger.error(f"Error creating mutation clustering image: {str(e)}")
        return None

def _create_pvalue_distribution_image(mutations):
    """
    Create a histogram of p-values for the identified mutations.
    
    Args:
        mutations (list): List of identified mutations
        
    Returns:
        str: Base64-encoded image or None if failed
    """
    try:
        # Extract p-values
        p_values = [m.get('p_value', 1.0) for m in mutations if 'p_value' in m]
        
        # Need at least a few mutations to visualize
        if len(p_values) < 5:
            return None
        
        # Create histogram
        plt.figure(figsize=(10, 6))
        
        # Log transform for better visualization
        log_p_values = -np.log10(np.clip(p_values, 1e-10, 1.0))
        
        # Plot histogram
        plt.hist(log_p_values, bins=20, alpha=0.7, color='blue')
        
        # Add significance threshold line (p=0.05)
        plt.axvline(x=-np.log10(0.05), color='red', linestyle='--', 
                   label='p=0.05 threshold')
        
        # Add labels
        plt.xlabel('-log10(p-value)')
        plt.ylabel('Number of Mutations')
        plt.title('Distribution of p-values for Identified Mutations')
        plt.legend()
        plt.tight_layout()
        
        # Save to base64 string
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', dpi=100)
        plt.close()
        img_buf.seek(0)
        return base64.b64encode(img_buf.read()).decode('utf-8')
    
    except Exception as e:
        logger.error(f"Error creating p-value distribution image: {str(e)}")
        return None

def _create_mutation_heatmap_image(mutations):
    """
    Create a heatmap showing relationships between mutations and resistance/non-resistance.
    
    Args:
        mutations (list): List of identified mutations
        
    Returns:
        str: Base64-encoded image or None if failed
    """
    try:
        # Select top 20 most significant mutations
        significant_mutations = sorted(
            [m for m in mutations if m.get('significant', False)],
            key=lambda x: x.get('p_value', 1.0)
        )
        
        top_mutations = significant_mutations[:20] if len(significant_mutations) > 20 else significant_mutations
        
        # Need at least a few mutations to visualize
        if len(top_mutations) < 3:
            return None
        
        # Prepare data for heatmap
        labels = [m.get('mutation', f"Mutation_{i}") for i, m in enumerate(top_mutations)]
        
        # Create matrix for heatmap [resistant_freq, non_resistant_freq]
        matrix = np.array([
            [m.get('resistant_freq', 0), m.get('non_resistant_freq', 0)]
            for m in top_mutations
        ])
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(matrix, cmap='YlOrRd', aspect='auto')
        
        # Add colorbar and labels
        plt.colorbar(label='Mutation Frequency')
        plt.xticks([0, 1], ['Resistant', 'Non-resistant'])
        plt.yticks(range(len(labels)), labels)
        plt.title('Mutation Frequencies in Resistant vs. Non-resistant Strains')
        plt.tight_layout()
        
        # Save to base64 string
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', dpi=100)
        plt.close()
        img_buf.seek(0)
        return base64.b64encode(img_buf.read()).decode('utf-8')
    
    except Exception as e:
        logger.error(f"Error creating mutation heatmap image: {str(e)}")
        return None

def _query_pdb_for_gene(gene):
    """
    Query PDB database for a gene.
    
    Args:
        gene (str): Gene name
        
    Returns:
        dict: PDB data if found, None otherwise
    """
    try:
        # Simplified PDB query - in a real application, this would be more sophisticated
        url = f"https://data.rcsb.org/rest/v1/holdings/entry/{gene.lower()}"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            return {
                'pdb_id': gene.lower(),
                'description': f"PDB structure for {gene}"
            }
        
        # If not found by gene name, try a more general search using JSON payload
        json_payload = {
            "query": {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_entity_source_organism.taxonomy_lineage.name",
                    "operator": "exact_match",
                    "value": gene
                }
            }
        }
        search_url = "https://search.rcsb.org/rcsbsearch/v2/query"
        search_response = requests.post(search_url, json=json_payload, timeout=5)
        
        if search_response.status_code == 200:
            results = search_response.json()
            if results.get('result_set') and len(results['result_set']) > 0:
                pdb_id = results['result_set'][0].get('identifier')
                return {
                    'pdb_id': pdb_id,
                    'description': f"PDB structure related to {gene}"
                }
    
    except Exception as e:
        logger.warning(f"Error querying PDB for gene {gene}: {str(e)}")
    
    return None
