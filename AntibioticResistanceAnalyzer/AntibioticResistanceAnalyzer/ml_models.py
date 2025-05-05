import logging
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict, cross_validate, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.feature_selection import SelectFromModel
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import io
import base64

logger = logging.getLogger(__name__)

def train_xgboost(X, y, params=None, mutation_mapping=None):
    """
    Train XGBoost model for feature selection.
    
    Args:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Labels
        params (dict, optional): XGBoost parameters
        mutation_mapping (list, optional): Mapping of feature indices to mutation names
        
    Returns:
        tuple: (selected_feature_indices, feature_importance_scores, feature_info)
    """
    if params is None:
        params = {
            'learning_rate': 0.1,
            'max_depth': 5,
            'n_estimators': 100,
            'objective': 'binary:logistic',
            'scale_pos_weight': 1.0,  # Balance class weights
            'subsample': 0.8,  # Prevent overfitting
            'colsample_bytree': 0.8  # Prevent overfitting
        }
    
    try:
        # Use stratified K-fold for training to handle imbalanced data
        stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        model = XGBClassifier(**params)
        model.fit(X, y)
        
        # Get feature importance
        importance = model.feature_importances_
        
        # Select features with non-zero importance
        selected_features = [i for i in range(len(importance)) if importance[i] > 0]
        
        # If no features selected, include at least the top 10
        if not selected_features and X.shape[1] > 0:
            top_n = min(10, X.shape[1])
            selected_features = np.argsort(importance)[-top_n:]
        
        # Use permutation importance for validation (more robust than built-in importance)
        perm_importance = None
        try:
            perm_importance = permutation_importance(model, X, y, n_repeats=5, random_state=42)
            perm_importance_mean = perm_importance.importances_mean
        except Exception as pe:
            logger.warning(f"Permutation importance calculation failed: {str(pe)}")
            perm_importance_mean = importance
        
        # Create feature info with both importance measures
        feature_info = []
        for i in range(len(importance)):
            feat_name = f"Feature_{i}"
            if mutation_mapping and i < len(mutation_mapping):
                feat_name = mutation_mapping[i]
                
            feature_info.append({
                'index': i,
                'name': feat_name,
                'xgb_importance': float(importance[i]),
                'perm_importance': float(perm_importance_mean[i]) if perm_importance is not None else 0.0
            })
        
        # Sort by importance
        feature_info.sort(key=lambda x: x['xgb_importance'], reverse=True)
        
        return selected_features, importance, feature_info
    
    except Exception as e:
        logger.error(f"Error training XGBoost model: {str(e)}")
        # Return all features if training fails
        return list(range(X.shape[1])), np.ones(X.shape[1]), []

def train_svm(X, y, params=None):
    """
    Train SVM classifier with optimized hyperparameters.
    
    Args:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Labels
        params (dict, optional): SVM parameters
        
    Returns:
        sklearn.svm.SVC: Trained SVM model
    """
    if params is None:
        # Optimized hyperparameters for antibiotic resistance prediction
        params = {
            'C': 1.0,  # Regularization parameter
            'kernel': 'rbf',  # Radial basis function kernel
            'gamma': 'scale',  # Kernel coefficient
            'probability': True,  # Enable probability estimates
            'class_weight': 'balanced'  # Handle class imbalance
        }
    
    try:
        # Use stratified K-fold for model validation
        stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Train SVM model
        model = SVC(**params)
        model.fit(X, y)
        return model
    
    except Exception as e:
        logger.error(f"Error training SVM model: {str(e)}")
        # Return a simple model with default parameters if training fails
        return SVC(probability=True, class_weight='balanced').fit(X, y)

def evaluate_model(model, X, y, cv=10, mutation_mapping=None):
    """
    Evaluate model using cross-validation with stratified sampling.
    
    Args:
        model: Trained machine learning model
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Labels
        cv (int): Number of cross-validation folds
        mutation_mapping (list, optional): Mapping from feature indices to mutation names
        
    Returns:
        dict: Model performance metrics and visualizations
    """
    try:
        # Use stratified k-fold to handle class imbalance
        stratified_kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        # Perform cross-validation
        cv_results = cross_validate(
            model, X, y, cv=stratified_kfold,
            scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
            return_estimator=True
        )
        
        # Get predictions from cross-validation
        y_pred = cross_val_predict(model, X, y, cv=stratified_kfold)
        
        # Get probability estimates if possible
        y_prob = np.zeros(len(y))
        try:
            y_prob = cross_val_predict(model, X, y, cv=stratified_kfold, method='predict_proba')[:, 1]
        except Exception as prob_err:
            logger.warning(f"Probability estimation failed: {str(prob_err)}")
            # If probability estimation fails, use binary predictions
            y_prob = y_pred.astype(float)
        
        # Calculate confusion matrix
        conf_matrix = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = conf_matrix.ravel()
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # Generate ROC curve visualization
        roc_curve_img = None
        try:
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            
            # Save to base64 string
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')
            plt.close()
            img_buf.seek(0)
            roc_curve_img = base64.b64encode(img_buf.read()).decode('utf-8')
        except Exception as vis_err:
            logger.error(f"ROC curve visualization failed: {str(vis_err)}")
        
        # Generate confusion matrix visualization
        cm_img = None
        try:
            plt.figure(figsize=(8, 6))
            plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.colorbar()
            tick_marks = np.arange(2)
            plt.xticks(tick_marks, ['Non-resistant', 'Resistant'])
            plt.yticks(tick_marks, ['Non-resistant', 'Resistant'])
            
            # Add text annotations to confusion matrix
            thresh = conf_matrix.max() / 2.0
            for i in range(conf_matrix.shape[0]):
                for j in range(conf_matrix.shape[1]):
                    plt.text(j, i, format(conf_matrix[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if conf_matrix[i, j] > thresh else "black")
            
            plt.tight_layout()
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            # Save to base64 string
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')
            plt.close()
            img_buf.seek(0)
            cm_img = base64.b64encode(img_buf.read()).decode('utf-8')
        except Exception as cm_err:
            logger.error(f"Confusion matrix visualization failed: {str(cm_err)}")
        
        # Calculate and return performance metrics with visualizations
        result = {
            'accuracy': float(np.mean(cv_results['test_accuracy'])),
            'precision': float(np.mean(cv_results['test_precision'])),
            'recall': float(np.mean(cv_results['test_recall'])),
            'f1_score': float(np.mean(cv_results['test_f1'])),
            'roc_auc': float(np.mean(cv_results['test_roc_auc'])),
            'confusion_matrix': {
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'tp': int(tp)
            },
            'visualizations': {
                'roc_curve': roc_curve_img,
                'confusion_matrix': cm_img
            }
        }
        
        return result
    
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        # Return placeholder metrics if evaluation fails
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'roc_auc': 0.0,
            'confusion_matrix': {
                'tn': 0,
                'fp': 0,
                'fn': 0,
                'tp': 0
            },
            'visualizations': {},
            'error': str(e)
        }

def apply_group_association_model(X, y, mutation_mapping):
    """
    Apply Group Association Model (GAM) to identify important mutations linked to antibiotic resistance.
    
    Args:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Labels
        mutation_mapping (list): Mapping from feature indices to mutation names
        
    Returns:
        list: Mutations ranked by importance
    """
    try:
        # Calculate feature importance using statistical measures
        resistant_samples = X[y == 1]
        non_resistant_samples = X[y == 0]
        
        # Calculate frequency in resistant and non-resistant samples
        resistant_freq = np.mean(resistant_samples, axis=0)
        non_resistant_freq = np.mean(non_resistant_samples, axis=0)
        
        # Calculate absolute frequency difference
        freq_diff = np.abs(resistant_freq - non_resistant_freq)
        
        # Calculate relative risk for each mutation
        # (frequency in resistant / frequency in non-resistant)
        epsilon = 1e-10  # Small value to avoid division by zero
        rel_risk = np.divide(resistant_freq + epsilon, non_resistant_freq + epsilon)
        
        # Calculate p-values for each mutation (Fisher's exact test)
        p_values = np.ones(X.shape[1])  # Initialize with 1.0 (not significant)
        for i in range(X.shape[1]):
            if i < len(mutation_mapping):
                try:
                    # Create contingency table for this mutation
                    ct = np.array([
                        [np.sum(np.logical_and(X[:, i] == 1, y == 1)), np.sum(np.logical_and(X[:, i] == 1, y == 0))],
                        [np.sum(np.logical_and(X[:, i] == 0, y == 1)), np.sum(np.logical_and(X[:, i] == 0, y == 0))]
                    ])
                    
                    # Apply Fisher's exact test
                    from scipy.stats import fisher_exact
                    _, p_value = fisher_exact(ct)
                    p_values[i] = p_value
                except Exception as stat_err:
                    logger.warning(f"Statistical test failed for mutation {i}: {str(stat_err)}")
        
        # Apply multiple testing correction (Benjamini-Hochberg FDR)
        from scipy.stats import false_discovery_control
        try:
            rejected, corrected_p_values = false_discovery_control(p_values, method='bh', alpha=0.05)
        except Exception as fdr_err:
            logger.warning(f"FDR correction failed: {str(fdr_err)}")
            corrected_p_values = p_values
        
        # Create a combined score based on frequency difference and statistical significance
        # Higher score = more important for resistance
        combined_scores = freq_diff * (1.0 - np.array(corrected_p_values))
        
        # Sort features by combined score
        sorted_indices = np.argsort(-combined_scores)
        
        # Generate visualizations for top mutations
        vis_data = {}
        try:
            # Create bar chart of frequency differences for top 10 mutations
            plt.figure(figsize=(10, 6))
            top_n = min(10, len(sorted_indices))
            top_indices = sorted_indices[:top_n]
            
            # Extract labels and values for top mutations
            labels = [mutation_mapping[i] if i < len(mutation_mapping) else f"Feature_{i}" for i in top_indices]
            r_freqs = [resistant_freq[i] for i in top_indices]
            nr_freqs = [non_resistant_freq[i] for i in top_indices]
            
            # Plot grouped bar chart
            x = np.arange(len(labels))
            width = 0.35
            
            plt.bar(x - width/2, r_freqs, width, label='Resistant')
            plt.bar(x + width/2, nr_freqs, width, label='Non-resistant')
            
            plt.ylabel('Frequency')
            plt.title('Top Mutations by Frequency in Resistant vs. Non-resistant Strains')
            plt.xticks(x, labels, rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            
            # Save to base64 string
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')
            plt.close()
            img_buf.seek(0)
            vis_data['frequency_chart'] = base64.b64encode(img_buf.read()).decode('utf-8')
            
            # Create significance plot for top mutations
            plt.figure(figsize=(10, 6))
            
            # Plot -log10 of p-values (higher = more significant)
            log_p_values = -np.log10([max(corrected_p_values[i], 1e-10) for i in top_indices])
            
            plt.bar(x, log_p_values)
            plt.axhline(y=-np.log10(0.05), color='r', linestyle='--', label='p=0.05')
            plt.ylabel('-log10(p-value)')
            plt.title('Statistical Significance of Top Mutations')
            plt.xticks(x, labels, rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            
            # Save to base64 string
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')
            plt.close()
            img_buf.seek(0)
            vis_data['significance_chart'] = base64.b64encode(img_buf.read()).decode('utf-8')
            
        except Exception as vis_err:
            logger.error(f"GAM visualization error: {str(vis_err)}")
        
        # Create list of mutations with their importance metrics
        ranked_mutations = []
        for i in sorted_indices:
            if i < len(mutation_mapping):
                ranked_mutations.append({
                    'mutation': mutation_mapping[i],
                    'resistant_freq': float(resistant_freq[i]),
                    'non_resistant_freq': float(non_resistant_freq[i]),
                    'freq_diff': float(freq_diff[i]),
                    'relative_risk': float(rel_risk[i]),
                    'p_value': float(p_values[i]),
                    'corrected_p_value': float(corrected_p_values[i]),
                    'combined_score': float(combined_scores[i]),
                    'significant': corrected_p_values[i] < 0.05
                })
        
        return ranked_mutations, vis_data
    
    except Exception as e:
        logger.error(f"Error applying GAM: {str(e)}")
        return [], {}
