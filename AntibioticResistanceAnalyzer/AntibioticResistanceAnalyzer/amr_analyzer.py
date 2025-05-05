import logging
import numpy as np
from app import db
from models import Result
from data_preprocessing import preprocess_genomes, encode_mutations
from ml_models import train_xgboost, train_svm, evaluate_model, apply_group_association_model
from visualization import generate_visualizations, map_mutations_to_structure
from utils import calculate_p_values, calculate_frequency_difference

logger = logging.getLogger(__name__)

class AMRAnalyzer:
    """
    Main class for analyzing antibiotic resistance mutations in bacterial genomes.
    Coordinates the entire analysis pipeline from preprocessing to visualization.
    """
    
    def __init__(self, result_id, parameters):
        """
        Initialize the analyzer with result ID and parameters.
        
        Args:
            result_id (int): ID of the Result record in the database
            parameters (dict): Analysis parameters
        """
        self.result_id = result_id
        self.parameters = parameters
        self.p_value_threshold = parameters.get('p_value_threshold', 0.05)
        self.min_frequency_diff = parameters.get('min_frequency_diff', 0.2)
        self.cv_folds = parameters.get('cv_folds', 10)
        self.feature_selection = parameters.get('feature_selection', 'xgboost')
        self.classifier = parameters.get('classifier', 'svm')
        
    def run_analysis(self, resistant_genomes, non_resistant_genomes):
        """
        Run the full analysis pipeline for antibiotic resistance mutation identification.
        
        Args:
            resistant_genomes (list): List of GenomeData objects for resistant strains
            non_resistant_genomes (list): List of GenomeData objects for non-resistant strains
        """
        logger.info(f"Starting analysis for result {self.result_id}")
        
        # Get result from database
        result = db.session.get(Result, self.result_id)
        if not result:
            raise ValueError(f"Result with ID {self.result_id} not found")
        
        try:
            # Step 1: Preprocess genomic data
            logger.info("Preprocessing genomic data")
            resistant_sequences = preprocess_genomes(resistant_genomes)
            non_resistant_sequences = preprocess_genomes(non_resistant_genomes)
            
            # Step 2: Identify mutations
            logger.info("Identifying mutations")
            mutations, X, y = self._identify_mutations(resistant_sequences, non_resistant_sequences)
            
            # Step 3: Statistical analysis
            logger.info("Performing statistical analysis")
            statistics = self._perform_statistical_analysis(mutations)
            
            # Step 4: Machine learning
            logger.info("Training machine learning models")
            model, feature_importance, model_performance = self._train_ml_model(X, y, mutations=mutations)
            
            # Step 5: Generate visualizations
            logger.info("Generating visualizations")
            # Use visualizations already generated in model_performance
            visualizations = model_performance.get('visualizations', {})
            
            # Add any additional visualizations
            additional_viz = generate_visualizations(mutations, model_performance, feature_importance)
            visualizations.update(additional_viz)
            
            # Step 6: Map mutations to 3D structures (if available)
            if self._should_map_to_structure(mutations):
                logger.info("Mapping mutations to 3D structures")
                structure_mappings = map_mutations_to_structure(mutations)
                visualizations['structure_mappings'] = structure_mappings
            
            # Extract top 10 statistically significant mutations from GAM results
            gam_results = model_performance.get('gam_results', [])
            significant_mutations = [m for m in gam_results if m.get('significant', False)]
            top_mutations = significant_mutations[:min(10, len(significant_mutations))]
            
            # Add the top mutations to the statistics
            statistics['top_resistance_mutations'] = top_mutations
            
            # Save results to database
            logger.info("Saving results to database")
            result.set_mutations(mutations)
            result.set_statistics(statistics)
            result.set_model_performance(model_performance)
            result.set_feature_importance(feature_importance)
            result.set_visualizations(visualizations)
            
            db.session.commit()
            logger.info(f"Analysis completed for result {self.result_id}")
            
        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}")
            # Save error in result
            result.set_statistics({'error': str(e)})
            db.session.commit()
            raise
    
    def _identify_mutations(self, resistant_sequences, non_resistant_sequences):
        """
        Identify mutations between resistant and non-resistant strains.
        
        Args:
            resistant_sequences (dict): Preprocessed sequences for resistant strains
            non_resistant_sequences (dict): Preprocessed sequences for non-resistant strains
            
        Returns:
            tuple: (mutations_list, feature_matrix, labels)
        """
        # Encode mutations as binary features
        X, y, mutation_mapping = encode_mutations(resistant_sequences, non_resistant_sequences)
        
        # Calculate mutation frequencies
        resistant_count = sum(1 for label in y if label == 1)
        non_resistant_count = sum(1 for label in y if label == 0)
        
        mutations_list = []
        for i, mutation in enumerate(mutation_mapping):
            # Calculate frequency of this mutation in resistant and non-resistant strains
            resistant_freq = np.sum(X[y == 1, i]) / resistant_count
            non_resistant_freq = np.sum(X[y == 0, i]) / non_resistant_count
            
            # Parse mutation information
            mutation_parts = mutation.split('_')
            if len(mutation_parts) >= 3:
                gene = mutation_parts[0]
                position = mutation_parts[1]
            else:
                gene = "Unknown"
                position = "Unknown"
            
            # Calculate p-value for this mutation
            p_value = calculate_p_values(X[:, i], y)[0]
            
            # Calculate frequency difference
            freq_diff = abs(resistant_freq - non_resistant_freq)
            
            # Determine if mutation is significant
            is_significant = p_value < self.p_value_threshold and freq_diff >= self.min_frequency_diff
            
            mutations_list.append({
                'mutation': mutation,
                'gene': gene,
                'position': position,
                'resistant_freq': float(resistant_freq),
                'non_resistant_freq': float(non_resistant_freq),
                'p_value': float(p_value),
                'freq_diff': float(freq_diff),
                'significant': is_significant
            })
        
        # Sort mutations by significance and frequency difference
        mutations_list.sort(key=lambda x: (not x['significant'], x['p_value'], -x['freq_diff']))
        
        return mutations_list, X, y
    
    def _perform_statistical_analysis(self, mutations):
        """
        Perform statistical analysis on identified mutations.
        
        Args:
            mutations (list): List of mutation dictionaries
            
        Returns:
            dict: Statistical results
        """
        # Count significant mutations
        significant_mutations = [m for m in mutations if m['significant']]
        
        # Group mutations by gene
        genes = {}
        for mutation in mutations:
            gene = mutation['gene']
            if gene not in genes:
                genes[gene] = {
                    'total': 0,
                    'significant': 0,
                    'mutations': []
                }
            
            genes[gene]['total'] += 1
            if mutation['significant']:
                genes[gene]['significant'] += 1
            genes[gene]['mutations'].append(mutation['mutation'])
        
        # Sort genes by number of significant mutations
        sorted_genes = sorted(genes.items(), key=lambda x: x[1]['significant'], reverse=True)
        top_genes = [{'gene': g[0], 'significant_mutations': g[1]['significant'], 'total_mutations': g[1]['total']} 
                    for g in sorted_genes[:10]]
        
        return {
            'total_mutations': len(mutations),
            'significant_mutations': len(significant_mutations),
            'top_genes': top_genes,
            'significance_threshold': self.p_value_threshold,
            'min_frequency_diff': self.min_frequency_diff
        }
    
    def _train_ml_model(self, X, y, mutations=None):
        """
        Train and evaluate machine learning models for antibiotic resistance prediction.
        
        Args:
            X (numpy.ndarray): Feature matrix
            y (numpy.ndarray): Labels
            mutations (list, optional): List of mutation dictionaries with mutation names
            
        Returns:
            tuple: (trained_model, feature_importance, model_performance)
        """
        # Create mutation mapping for feature interpretation
        mutation_mapping = None
        if mutations:
            mutation_mapping = [mut['mutation'] for mut in mutations]
        
        # Apply Group Association Model (GAM) to identify important mutations based on frequency
        ranked_mutations, gam_visualizations = apply_group_association_model(X, y, mutation_mapping)
        
        # Perform feature selection with XGBoost
        if self.feature_selection == 'xgboost':
            selected_features, feature_importance_scores, feature_info = train_xgboost(X, y, mutation_mapping=mutation_mapping)
        else:
            # Use all features if no feature selection is specified
            selected_features = list(range(X.shape[1]))
            feature_importance_scores = [1.0] * len(selected_features)
            feature_info = []
        
        # Select the top features
        X_selected = X[:, selected_features]
        
        # Map selected features back to mutations
        selected_mutation_mapping = []
        for idx in selected_features:
            if mutation_mapping and idx < len(mutation_mapping):
                selected_mutation_mapping.append(mutation_mapping[idx])
            else:
                selected_mutation_mapping.append(f"Feature_{idx}")
        
        # Train the classifier model
        if self.classifier == 'svm':
            model = train_svm(X_selected, y)
        else:
            # Default to SVM if classifier is not specified
            model = train_svm(X_selected, y)
        
        # Evaluate the model with cross-validation
        performance_metrics = evaluate_model(model, X_selected, y, cv=self.cv_folds, 
                                           mutation_mapping=selected_mutation_mapping)
        
        # Add GAM visualizations to the performance metrics
        if 'visualizations' not in performance_metrics:
            performance_metrics['visualizations'] = {}
        performance_metrics['visualizations'].update(gam_visualizations)
        
        # If we have ranked mutations from GAM, include those results as well
        performance_metrics['gam_results'] = ranked_mutations
        
        # Get top 10 mutations from GAM model
        top_mutations = ranked_mutations[:min(10, len(ranked_mutations))]
        
        # Log the top mutations identified
        logger.info(f"Top mutations identified: {[m['mutation'] for m in top_mutations]}")
        
        # Create feature importance list combining XGBoost and GAM results
        feature_importance = []
        if feature_info:
            # Use the richly annotated feature info from XGBoost
            feature_importance = feature_info
        else:
            # Create basic feature importance from importance scores
            for i, idx in enumerate(selected_features):
                if i < len(feature_importance_scores):
                    name = f"Feature_{idx}"
                    if mutation_mapping and idx < len(mutation_mapping):
                        name = mutation_mapping[idx]
                    
                    feature_importance.append({
                        'feature': name,
                        'importance': float(feature_importance_scores[i])
                    })
        
        # Sort by importance
        feature_importance.sort(key=lambda x: x.get('xgb_importance', x.get('importance', 0)), reverse=True)
        
        return model, feature_importance, performance_metrics
    
    def _should_map_to_structure(self, mutations):
        """
        Determine if mutations should be mapped to 3D structures.
        
        Args:
            mutations (list): List of mutation dictionaries
            
        Returns:
            bool: Whether to map mutations to structures
        """
        # Only attempt to map to structures if we have significant mutations
        significant_mutations = [m for m in mutations if m['significant']]
        return len(significant_mutations) > 0
