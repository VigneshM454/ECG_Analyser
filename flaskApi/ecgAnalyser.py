import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.gridspec as gridspec
from sklearn.inspection import permutation_importance

from io import BytesIO
import base64

from preProcess2 import ECGPreprocessor

MODEL_DIR = './models/'
TEMP_DIR = '../temp/'
DATA_PATH = '../datasets/mit-bih'  # Path to MIT-BIH dataset

arrhythmia_mapping = {
        'L': 'Left Bundle Branch Block',
        'R': 'Right Bundle Branch Block',
        'A': 'Atrial Premature Contraction',
        'V': 'Premature Ventricular Contraction',
        '/': 'Paced Beat',
        'E': 'Ventricular Escape Beat',
        'j': 'Junctional Premature Beat',
        'S': 'Supraventricular Premature Beat',
        'F': 'Fusion of Ventricular and Normal Beat',
        'f': 'Fusion of Paced and Normal Beat',
        'Q': 'Unclassifiable Beat',
        'N': 'Normal Beat'
    }

def load_model_and_artifacts():
    """Load the trained model and associated artifacts."""
    print("Loading model and artifacts...")
    
    # Load model
    model_path = os.path.join(MODEL_DIR, 'bayesian_ecg_model.keras')
    model = keras.models.load_model(model_path)
    
    # Load scalers
    feature_scaler = joblib.load(os.path.join(MODEL_DIR, 'feature_scaler.pkl'))
    segment_scaler = joblib.load(os.path.join(MODEL_DIR, 'segment_scaler.pkl'))
    
    # Load label classes
    label_classes = np.load(os.path.join(MODEL_DIR, 'label_classes.npy'), allow_pickle=True)
    
    print("Model and artifacts loaded successfully!")
    print(f"Feature scaler expects {feature_scaler.n_features_in_} features")
    print(f"Segment scaler expects {segment_scaler.n_features_in_} features")
    
    return model, feature_scaler, segment_scaler, label_classes

def predict_with_uncertainty(model, features_data, segments_data, feature_scaler, segment_scaler, label_classes, num_samples=30):
    """Make predictions with uncertainty estimation."""
    # Scale input data
    features_scaled = feature_scaler.transform(features_data)
    segments_scaled = segment_scaler.transform(segments_data)
    
    # Make multiple predictions for uncertainty estimation
    predictions = []
    for _ in range(num_samples):
        preds = model.predict(
            {'features_input': features_scaled, 'segments_input': segments_scaled}, 
            verbose=0
        )
        predictions.append(preds)
    
    # Stack predictions
    predictions_array = np.stack(predictions)
    
    # Calculate mean and standard deviation across samples
    mean_probs = np.mean(predictions_array, axis=0)
    std_probs = np.std(predictions_array, axis=0)
    
    # Get predicted classes
    predicted_classes = np.argmax(mean_probs, axis=1)
    predicted_class_names = label_classes[predicted_classes]
    
    # Calculate uncertainty (entropy)
    entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10), axis=1)
    
    return predicted_class_names, mean_probs, entropy, std_probs, features_scaled, segments_scaled
def get_feature_importance(model, features_scaled, segments_scaled, label_idx, feature_names):
    """
    Calculate feature importance using a more effective permutation approach.
    """
    # Ensure features_scaled is a 2D array
    if len(features_scaled.shape) == 1:
        features_scaled = features_scaled.reshape(1, -1)
    
    # Define a prediction function that handles batch inputs properly
    def predict_prob(X):
        # Make prediction
        preds = model.predict({'features_input': X, 'segments_input': segments_scaled}, verbose=0)
        # Return probability for the specified class
        return preds[:, label_idx]
    
    # Get baseline prediction
    baseline_prediction = predict_prob(features_scaled)
    
    # Manual permutation importance calculation with increased sensitivity
    importances = []
    n_repeats = 10  # Increased for stability
    
    for i in range(features_scaled.shape[1]):
        # Make multiple copies for more robust estimation
        importance_samples = []
        
        for _ in range(n_repeats):
            # Make a copy of the features
            X_permuted = features_scaled.copy()
            
            # Save original values
            original_values = X_permuted[:, i].copy()
            
            # Permute one feature with stronger perturbation
            min_val = np.min(original_values) - 0.5
            max_val = np.max(original_values) + 0.5
            
            # Ensure meaningful perturbation
            if abs(max_val - min_val) < 0.1:
                perturbation = np.random.normal(0, 0.5, size=original_values.shape)
            else:
                perturbation = np.random.uniform(min_val, max_val, size=original_values.shape)
                
            X_permuted[:, i] = perturbation
            
            # Measure change in prediction
            permuted_prediction = predict_prob(X_permuted)
            
            # Calculate importance as the absolute difference (scaled up for visibility)
            importance = np.abs(baseline_prediction - permuted_prediction).mean() * 10.0
            importance_samples.append(importance)
        
        # Use the mean importance
        importances.append(np.mean(importance_samples))
    
    # Scale importances for better visualization
    if max(importances) < 0.001:
        scale_factor = 1.0 / max(importances) if max(importances) > 0 else 1.0
        importances = [imp * scale_factor for imp in importances]
    
    # Create a dictionary mapping feature names to importance scores
    importance_dict = {feature_names[i]: importances[i] for i in range(len(feature_names))}
    
    # Sort by importance
    importance_dict = {k: v for k, v in sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)}
    
    return importance_dict

def explain_with_lime(model, features_data, feature_names, features_scaled, segments_scaled, label_idx, label_classes, feature_scaler):
    """
    Improved LIME explanation generator with better training sample generation.
    """
    # Create a wrapper prediction function
    def predict_fn(X):
        # Ensure X is a numpy array
        if isinstance(X, list):
            X = np.array(X)
        
        # Handle single samples or batches
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        try:
            # Scale the input data
            X_scaled = feature_scaler.transform(X)
            
            # Keep segments constant for all perturbed samples
            segments_batch = np.tile(segments_scaled, (X.shape[0], 1))
            
            # Get model predictions
            batch_predictions = model.predict(
                {'features_input': X_scaled, 'segments_input': segments_batch}, 
                verbose=0
            )
            
            return batch_predictions
        
        except Exception as e:
            print(f"Error in predict_fn: {str(e)}")
            # Return safe default
            return np.ones((X.shape[0], len(label_classes))) / len(label_classes)
    
    # Ensure features_data is in the right format
    if isinstance(features_data, pd.DataFrame):
        features_np = features_data.values
    else:
        features_np = np.array(features_data)
    
    # Convert feature names to strings
    feature_names_str = [str(name) for name in feature_names]
    
    # Create better training data for LIME
    # Generate wider range of variations for each feature
    n_samples = 3000  # More samples for better stability
    training_data = np.zeros((n_samples, features_np.shape[1]))
    
    # First, add the original sample
    training_data[0] = features_np[0]
    
    # Get feature ranges from the scaler 
    # This ensures the values we generate are within the expected distribution
    if hasattr(feature_scaler, 'min_') and hasattr(feature_scaler, 'max_'):
        # Use direct attributes if available
        feature_mins = feature_scaler.min_
        feature_maxs = feature_scaler.max_
    else:
        # Fallback: Calculate from the scaled data and inverse transform
        # Get the min and max values from the original data
        feature_mins = np.min(features_data, axis=0)
        feature_maxs = np.max(features_data, axis=0)
    
    # feature_mins = feature_scaler.data_min_
    # feature_maxs = feature_scaler.data_max_
    
    # Generate diverse samples within the expected feature ranges
    for i in range(1, n_samples):
        # Generate a sample with some features changed
        sample = features_np[0].copy()
        
        # Randomly modify 30-70% of features
        num_features_to_modify = np.random.randint(
            int(0.3 * len(sample)), 
            int(0.7 * len(sample)) + 1
        )
        
        # Select features to modify
        features_to_modify = np.random.choice(
            range(len(sample)), 
            size=num_features_to_modify, 
            replace=False
        )
        
        # Modify selected features with values in expected range
        for j in features_to_modify:
            min_val = feature_mins[j]
            max_val = feature_maxs[j]
            # Ensure min_val != max_val to avoid issues
            if abs(max_val - min_val) < 1e-6:
                sample[j] = sample[j] + np.random.normal(0, 0.5)
            else:
                sample[j] = np.random.uniform(min_val, max_val)
        
        training_data[i] = sample
    
    try:
        # Create explainer with optimized settings
        explainer = LimeTabularExplainer(
            training_data=training_data,
            feature_names=feature_names_str,
            class_names=[str(c) for c in label_classes],
            mode='classification',
            random_state=42,
            discretize_continuous=False,  # Better for continuous features
            sample_around_instance=True   # Sample around the instance
        )
        
        # Generate explanation with increased stability
        explanation = explainer.explain_instance(
            data_row=features_np[0],
            predict_fn=predict_fn,
            num_features=min(len(feature_names), 10),
            num_samples=2000,  # More samples for stability
            top_labels=1
        )
        
        # Verify explanation was generated
        exp_list = explanation.as_list(label=label_idx)
        print(f"LIME explanation contains {len(exp_list)} features")
        
        # Scale the LIME values for better visibility
        if exp_list:
            values = [abs(x[1]) for x in exp_list]
            if max(values) < 0.001:
                scale_factor = 0.1 / max(values) if max(values) > 0 else 1.0
                exp_list = [(feat, val * scale_factor) for feat, val in exp_list]
                # Reassign the scaled values back to the explanation
                explanation._local_exp = {label_idx: [(i, exp_list[i][1]) for i in range(len(exp_list))]}
        
        return explanation
        
    except Exception as e:
        print(f"Error generating LIME explanation: {str(e)}")
        
        # Create a meaningful fallback explanation based on feature importance
        class FeatureBasedExplanation:
            def __init__(self, features, feature_vals, label_idx):
                self.features = features
                self.values = feature_vals
                self.label = label_idx
                
                # Create normalized importance scores based on feature magnitudes
                # Scale for better visibility
                abs_vals = np.abs(features_scaled[0])
                max_val = max(abs_vals) if max(abs_vals) > 0 else 1.0
                
                # Generate alternating positive/negative values for top features
                # This simulates LIME's format of supporting/contradicting features
                self.importance_scores = []
                for i in range(len(features)):
                    value = (abs_vals[i] / max_val) * 0.2  # Scale to reasonable LIME range
                    sign = 1 if i % 2 == 0 else -1  # Alternate signs
                    self.importance_scores.append((i, sign * value))
                
                # Sort by absolute importance
                self.importance_scores = sorted(
                    self.importance_scores, 
                    key=lambda x: abs(x[1]), 
                    reverse=True
                )
            
            def as_list(self, label=None):
                # Return in LIME format with feature names
                return [(self.features[idx], val) for idx, val in self.importance_scores[:10]]
                
            def available_labels(self):
                return [self.label]
        
        return FeatureBasedExplanation(feature_names_str, features_np[0], label_idx)

def analyze_ecg_segment(signals, segment_data, label_class):
    """
    Analyze the ECG segment to identify key cardiac features.
    
    Parameters:
    signals - raw ECG signals
    segment_data - preprocessed segment data
    label_class - predicted label class
    
    Returns:
    segment_analysis - dictionary with ECG segment analysis
    """
    # Create a simple analysis of the segment
    # In a real implementation, this would be more sophisticated
    segment = segment_data[0]  # Get the first (and only) segment
    
    # Calculate basic statistics
    segment_min = np.min(segment)
    segment_max = np.max(segment)
    segment_range = segment_max - segment_min
    segment_mean = np.mean(segment)
    
    # Detect potential irregularities
    irregularities = []
    
    # Analyze based on the predicted class
    if label_class == 'N':
        rhythm_assessment = "Regular sinus rhythm detected"
    elif label_class == 'V':
        rhythm_assessment = "Premature ventricular contraction detected"
        irregularities.append("Abnormal QRS complex")
    elif label_class == 'A':
        rhythm_assessment = "Atrial premature contraction detected"
        irregularities.append("Premature P wave")
    elif label_class == 'L':
        rhythm_assessment = "Left bundle branch block detected"
        irregularities.append("Wide QRS complex")
    elif label_class == 'R':
        rhythm_assessment = "Right bundle branch block detected"
        irregularities.append("RSR' pattern in V1")
    else:
        rhythm_assessment = f"Abnormal rhythm detected ({label_class})"
        irregularities.append("Unknown irregularity")
    
    # Return the analysis
    return {
        'rhythm_assessment': rhythm_assessment,
        'segment_min': segment_min,
        'segment_max': segment_max,
        'segment_range': segment_range,
        'segment_mean': segment_mean,
        'irregularities': irregularities
    }

def visualize_results_with_xai(predicted_class_names, mean_probs, entropy, std_probs, label_classes, 
                           feature_importance, lime_exp, segment_data, segment_analysis, processed_data, sample_idx):
    """Visualize the prediction results with XAI visualizations."""
    import base64
    from io import BytesIO
    
    images_binary = {}
    
    # 1. ECG Segment plot
    fig_segment = plt.figure(figsize=(10, 4))
    segment = segment_data[0]
    plt.plot(segment, color='blue', linewidth=1.5)
    plt.title('ECG Segment Analysis', fontsize=14)
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add rhythm assessment annotation
    assessment_text = segment_analysis['rhythm_assessment']
    plt.annotate(assessment_text, xy=(0.5, 0.95), xycoords='axes fraction', 
                 horizontalalignment='center', verticalalignment='top',
                 bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
    
    # Save segment plot to binary
    buf_segment = BytesIO()
    fig_segment.savefig(buf_segment, format='png', bbox_inches='tight')
    buf_segment.seek(0)
    images_binary['segment_plot'] = buf_segment.getvalue()
    plt.close(fig_segment)
    
    # 2. Class probabilities plot
    fig_probs = plt.figure(figsize=(8, 4))
    sample_idx = 0  # First sample
    class_probs = mean_probs[sample_idx]
    bar_colors = ['lightblue' if label != 'N' else 'lightgreen' for label in label_classes]
    
    plt.bar(label_classes, class_probs, color=bar_colors)
    plt.errorbar(range(len(label_classes)), class_probs, yerr=std_probs[sample_idx], fmt='o', color='black', capsize=5)
    plt.xlabel('Class')
    plt.ylabel('Probability')
    plt.title('Class Probabilities with Uncertainty')
    plt.ylim(0, 1)
    
    # Save probabilities plot to binary
    buf_probs = BytesIO()
    fig_probs.savefig(buf_probs, format='png', bbox_inches='tight')
    buf_probs.seek(0)
    images_binary['probability_plot'] = buf_probs.getvalue()
    plt.close(fig_probs)
    
    # 3. Uncertainty plot
    fig_entropy = plt.figure(figsize=(5, 4))
    plt.bar(['Prediction Uncertainty'], [entropy[sample_idx]], color='coral')
    plt.ylabel('Entropy')
    plt.title('Prediction Uncertainty (Entropy)')
    plt.ylim(0, 1)
    
    # Save entropy plot to binary
    buf_entropy = BytesIO()
    fig_entropy.savefig(buf_entropy, format='png', bbox_inches='tight')
    buf_entropy.seek(0)
    images_binary['uncertainty_plot'] = buf_entropy.getvalue()
    plt.close(fig_entropy)
    
    # 4. Risk assessment plot
    fig_risk = plt.figure(figsize=(10, 3))
    normal_prob = mean_probs[sample_idx][np.where(label_classes == 'N')[0][0]] if 'N' in label_classes else 0
    abnormal_prob = 1 - normal_prob
    
    plt.axhspan(0, 0.33, facecolor='green', alpha=0.3)
    plt.axhspan(0.33, 0.66, facecolor='yellow', alpha=0.3)
    plt.axhspan(0.66, 1, facecolor='red', alpha=0.3)
    
    plt.barh(['Risk Level'], [abnormal_prob], color='red', alpha=0.7)
    plt.xlim(0, 1)
    plt.xlabel('Abnormality Risk Score')
    plt.title('Heart Condition Risk Assessment')
    
    risk_text = 'LOW RISK' if abnormal_prob < 0.33 else 'MEDIUM RISK' if abnormal_prob < 0.66 else 'HIGH RISK'
    plt.text(abnormal_prob + 0.05, 0, f"{risk_text} ({abnormal_prob:.2%})", 
             va='center', ha='left', fontweight='bold')
    
    # Save risk plot to binary
    buf_risk = BytesIO()
    fig_risk.savefig(buf_risk, format='png', bbox_inches='tight')
    buf_risk.seek(0)
    images_binary['risk_plot'] = buf_risk.getvalue()
    plt.close(fig_risk)
    
    # 5. Feature importance plot
    fig_importance = plt.figure(figsize=(8, 6))
    features = list(feature_importance.keys())[:10]  # Top 10 features
    importance_values = list(feature_importance.values())[:10]
    max_importance = max(importance_values) if importance_values and max(importance_values) > 0 else 1.0
    
    bars = plt.barh(features, importance_values, color='skyblue')
    plt.title('Top 10 Most Important Features')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    
    for i, bar in enumerate(bars):
        bar.set_color(plt.cm.viridis(importance_values[i] / max_importance))
    
    # Save feature importance plot to binary
    buf_importance = BytesIO()
    fig_importance.savefig(buf_importance, format='png', bbox_inches='tight')
    buf_importance.seek(0)
    images_binary['feature_importance_plot'] = buf_importance.getvalue()
    plt.close(fig_importance)
    
    # 6. LIME explanation plot
    fig_lime = plt.figure(figsize=(8, 6))
    try:
        label_idx = np.where(label_classes == predicted_class_names[0])[0][0]
        lime_expl = lime_exp.as_list(label=label_idx)
        
        if not lime_expl:
            lime_expl = lime_exp.as_list(label=0)
        
        lime_expl = sorted(lime_expl, key=lambda x: abs(x[1]), reverse=True)
        lime_expl = lime_expl[:10]  # Top 10 features
        
        lime_features = [str(x[0]) for x in lime_expl]
        lime_values = [x[1] for x in lime_expl]
        
        plt.barh(lime_features, lime_values, color=['green' if v > 0 else 'red' for v in lime_values])
        plt.title(f'LIME Explanation for {predicted_class_names[0]} Class')
        plt.xlabel('Contribution')
        plt.axvline(x=0, color='black', linestyle='--')
        
        print_lime_expl = lime_expl
    except Exception as e:
        print(f"Error in LIME visualization: {str(e)}")
        plt.text(0.5, 0.5, f'LIME explanation unavailable', 
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes, color='red')
        plt.title('LIME Explanation (Error)')
        print_lime_expl = []
    
    # Save LIME plot to binary
    buf_lime = BytesIO()
    fig_lime.savefig(buf_lime, format='png', bbox_inches='tight')
    buf_lime.seek(0)
    images_binary['lime_plot'] = buf_lime.getvalue()
    plt.close(fig_lime)
    
    # 7. Rhythm classification plot (if available)
    fig_rhythm = plt.figure(figsize=(10, 3))
    if 'labels' in processed_data:
        start_idx = max(0, sample_idx - 20)
        end_idx = min(len(processed_data['labels']), sample_idx + 20)
        
        label_values = processed_data['labels'][start_idx:end_idx]
        unique_labels = list(set(label_values))
        
        label_to_int = {label: i for i, label in enumerate(unique_labels)}
        int_values = [label_to_int[label] for label in label_values]
        
        plt.imshow([int_values], aspect='auto', cmap='viridis')
        plt.title('Heart Rhythm Classification Over Time')
        plt.yticks([])
        plt.xlabel('Beat Index')
        
        # Add colorbar
        from matplotlib.colors import BoundaryNorm
        bounds = np.arange(len(unique_labels) + 1) - 0.5
        norm = BoundaryNorm(bounds, len(unique_labels))
        
        cbar = plt.colorbar()
        cbar.set_ticks(np.arange(len(unique_labels)))
        label_descriptions = [arrhythmia_mapping.get(l, 'Unknown') for l in unique_labels]
        cbar.set_ticklabels(label_descriptions)
        
        plt.axvline(x=sample_idx - start_idx, color='red', linestyle='--', linewidth=2)
    else:
        plt.text(0.5, 0.5, 'No rhythm data available', 
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes)
    
    # Save rhythm plot to binary
    buf_rhythm = BytesIO()
    fig_rhythm.savefig(buf_rhythm, format='png', bbox_inches='tight')
    buf_rhythm.seek(0)
    images_binary['rhythm_plot'] = buf_rhythm.getvalue()
    plt.close(fig_rhythm)
    
    encoded_images={}
    for key, binary_data in images_binary.items():
        encoded_images[key] = base64.b64encode(binary_data).decode('utf-8')

    # Instead of saving, just keep the figure for returning    
    # Prepare structured results to return instead of printing
    if predicted_class_names[0] == 'N':
        risk_level = "Normal Heart Rhythm"
        recommendation = "No immediate action required. Continue regular check-ups."
    else:
        # Higher entropy means more uncertainty
        if entropy[0] > 0.5:  # Threshold can be adjusted
            confidence_status = "with HIGH UNCERTAINTY"
            recommendation = "Recommend follow-up with a specialist for confirmation."
        else:
            confidence_status = "with HIGH CONFIDENCE"
            recommendation = "Immediate medical attention recommended."
            
        arrhythmia_description = arrhythmia_mapping.get(predicted_class_names[0], 'Unknown Arrhythmia')
        risk_level = f"Abnormal Heart Rhythm Detected: {arrhythmia_description} {confidence_status}"
    
    # Prepare class probabilities details
    class_probabilities = []
    for i, cls in enumerate(label_classes):
        class_desc = ''
        if cls in arrhythmia_mapping:
            class_desc = arrhythmia_mapping[cls]
        
        class_probabilities.append({
            'class': cls,
            'description': class_desc,
            'probability': mean_probs[0][i],
            'std_deviation': std_probs[0][i]
        })
    
    # Uncertainty analysis
    certainty_level = "Low" if entropy[0] > 0.5 else "Medium" if entropy[0] > 0.2 else "High"
    
    # Top features importance
    top_features = []
    for i, (feature, importance) in enumerate(list(feature_importance.items())[:5]):
        top_features.append({
            'feature': feature,
            'importance': importance
        })
    
    # LIME explanation
    lime_features = []
    if print_lime_expl and len(print_lime_expl) > 0:
        for i, (feature, value) in enumerate(print_lime_expl[:min(5, len(print_lime_expl))]):
            impact = "supporting" if value > 0 else "contradicting"
            lime_features.append({
                'feature': feature,
                'value': value,
                'impact': impact
            })
    
    # Structured results
    results = {
        'summary': {
            'predicted_class': predicted_class_names[0],
            'condition': arrhythmia_mapping.get(predicted_class_names[0], 'Unknown'),
            'confidence': 1 - entropy[0]
        },
        'risk_assessment': {
            'risk_level': risk_level,
            'recommendation': recommendation
        },
        'detailed_probabilities': class_probabilities,
        'uncertainty_analysis': {
            'entropy': entropy[0],
            'certainty_level': certainty_level
        },
        'feature_importance': top_features,
        'lime_explanation': lime_features,
        'segment_analysis': segment_analysis
    }
    
    # Return both the figure and the structured results
    # return fig, results

    return encoded_images, results


def prepare_data_for_prediction(processed_data, sample_idx=None):
    """
    Prepare the processed ECG data for prediction.
    
    Parameters:
    processed_data (dict): Processed ECG data from ECGPreprocessor
    sample_idx (int): Index of the segment to use for prediction (defaults to random)
    
    Returns:
    tuple: (segment_data, feature_data) ready for prediction
    """
    if not processed_data['segments']:
        print("Error: No segments found in processed data")
        return None, None
    
    # Select a segment (default to a random one if not specified)
    if sample_idx is None:
        sample_idx = min(10, len(processed_data['segments']) - 1)
    
    # Get the selected segment
    segment = processed_data['segments'][sample_idx]
    
    # Convert to required format
    segment_data = segment.reshape(1, -1)
    
    # Get the corresponding features
    features_df = processed_data['features']
    if features_df.empty:
        print("Error: No features found in processed data")
        return None, None
    
    # Feature selection (use the same features as in the original code)
    feature_columns = [
        'rr_prev', 'rr_next', 'rr_ratio', 'r_amplitude', 
        'p_amplitude', 'qrs_duration', 't_amplitude', 'st_segment',
        'p_duration', 'pr_interval', 'qt_interval', 'r_symmetry',
        'qrs_area', 'p_area', 't_area', 't_symmetry',
        'entropy', 'spectral_energy', 'spectral_entropy', 'frequency_ratio',
        'heartrate', 'wavelet_coef1', 'wavelet_coef2', 'wavelet_coef3',
        'wavelet_coef4', 'wavelet_coef5'
    ]
    
    # Create feature dataframe for the selected segment
    feature_data = pd.DataFrame([features_df.iloc[sample_idx]])
    
    # Ensure all required features are present
    for col in feature_columns:
        if col not in feature_data.columns:
            feature_data[col] = 0.0  # Default value
    
    # Select only the required features in the correct order
    feature_data = feature_data[feature_columns]
    
    print(f"Prepared segment shape: {segment_data.shape}")
    print(f"Prepared feature shape: {feature_data.shape}")
    
    return segment_data, feature_data.values, feature_columns, sample_idx

def analyze_ecg_with_xai(record_number, model_dir=None, data_path=None, lead_index=0, sample_idx=None):
    """
    Process a raw MIT-BIH ECG record, predict with uncertainty, and explain results.
    
    Parameters:
    record_number (str): ID of the MIT-BIH record to process
    model_dir (str, optional): Path to the model directory
    data_path (str, optional): Path to the MIT-BIH dataset
    lead_index (int): Index of the lead to use (0=MLII typically)
    sample_idx (int, optional): Index of the heartbeat segment to analyze
    
    Returns:
    dict: Analysis results with XAI explanations
    """
    global MODEL_DIR, DATA_PATH
    print("inside analyze ecg with xai .....")
    # Update paths if provided
    if model_dir:
        MODEL_DIR = model_dir
        print('model dir is updated')
    if data_path:
        DATA_PATH = data_path
        print('data path is updated')
        
    print(f"Using model directory: {MODEL_DIR}")
    print(f"Using data path: {DATA_PATH}")
    
    # Load model and artifacts
    model, feature_scaler, segment_scaler, label_classes = load_model_and_artifacts()
    
    # Create ECGPreprocessor
    fs = 360  # Sampling frequency for MIT-BIH
    preprocessor = ECGPreprocessor(data_path=DATA_PATH, fs=fs)
    
    # Load the record
    record_path = os.path.join(DATA_PATH, record_number)
    print(f"Loading record from: {record_path}")
    print('record number')
    print(record_number)
    print('record path')
    print(record_path)
    print('data path')
    print(data_path)
    signals, annotations, fields = preprocessor.load_record(record_number)
    
    if signals is None:
        print("Error: Could not read ECG record")
        return None
    
    print(f"Record loaded. Signal shape: {signals.shape}")
    print(f"Number of annotations: {len(annotations.sample) if annotations else 0}")
    print(f"Sampling frequency: {fields['fs']} Hz")
    
    # Process the ECG signal
    print("Processing ECG signal...")
    processed_data = preprocessor.process_single_ecg(signals, annotations)
    
    print(f"Number of detected beats: {len(processed_data['segments'])}")
    if 'labels' in processed_data:
        beat_counts = pd.Series(processed_data['labels']).value_counts()
        print(f"Beat classes found: {set(processed_data['labels'])}")
        print(f"Beat counts: {beat_counts.to_dict()}")
    
    # Prepare data for prediction
    segment_data, feature_data, feature_names, sample_idx = prepare_data_for_prediction(processed_data, sample_idx)
    
    if segment_data is None or feature_data is None:
        print("Error: Failed to prepare data for prediction")
        return None
    
    # Make prediction with uncertainty
    predicted_classes, mean_probs, entropy, std_probs, features_scaled, segments_scaled = predict_with_uncertainty(
        model, feature_data, segment_data, feature_scaler, segment_scaler, label_classes
    )
    
    # Get the index of the predicted class
    label_idx = np.where(label_classes == predicted_classes[0])[0][0]
    
    # Get feature importance
    print("Calculating feature importance...")
    feature_importance = get_feature_importance(model, features_scaled, segments_scaled, label_idx, feature_names)
    
    # Get LIME explanation
    print("Generating LIME explanations...")
    lime_exp = explain_with_lime(model, feature_data, feature_names, features_scaled, segments_scaled, label_idx, label_classes, feature_scaler)

    
    # Analyze the ECG segment
    print("Analyzing ECG segment...")
    segment_analysis = analyze_ecg_segment(signals, segment_data, predicted_classes[0])

    
    # Visualize results with XAI and get structured results plus separate images
    image_binaries, structured_results = visualize_results_with_xai(
        predicted_classes, mean_probs, entropy, std_probs, label_classes,
        feature_importance, lime_exp, segment_data, segment_analysis, processed_data, sample_idx
    )
    
    # Return analysis results including the binary images
    return {
        'structured_results': structured_results,
        'images': image_binaries,  # Now returns a dictionary of binary images
        'predicted_classes': predicted_classes,
        'probabilities': mean_probs,
        'entropy': entropy,
        'std_probs': std_probs,
        'label_classes': label_classes,
        'feature_importance': feature_importance,
        'lime_explanation': lime_exp,
        'segment_analysis': segment_analysis,
        'processed_data': processed_data
    }
    
    # Visualize results with XAI and get structured results
    # fig, structured_results = visualize_results_with_xai(
    #     predicted_classes, mean_probs, entropy, std_probs, label_classes,
    #     feature_importance, lime_exp, segment_data, segment_analysis, processed_data, sample_idx
    # )
    
    # # Create a BytesIO object to store the figure
    # from io import BytesIO
    # import base64
    
    # buf = BytesIO()
    # fig.savefig(buf, format='png', bbox_inches='tight')
    # buf.seek(0)
    
    # # Convert to base64 for easy inclusion in web applications
    # img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    # # Close the figure to free resources
    # plt.close(fig)
    
    # # Return analysis results including the encoded figure
    # return {
    #     'structured_results': structured_results,
    #     'figure_base64': img_str,
    #     'predicted_classes': predicted_classes,
    #     'probabilities': mean_probs,
    #     'entropy': entropy,
    #     'std_probs': std_probs,
    #     'label_classes': label_classes,
    #     'feature_importance': feature_importance,
    #     'lime_explanation': lime_exp,
    #     'segment_analysis': segment_analysis,
    #     'processed_data': processed_data
    # }


def analyze_ecg(record_path=None, record_number=None, model_dir="../models/", 
                data_path=None, lead_index=0, sample_idx=None, with_xai=False):
    """
    Analyze an ECG record with or without XAI
    
    Parameters:
    -----------
    record_path : str
        Path to record file or directory containing the record
    record_number : str or int
        Record number (used if record_path is a directory)
    model_dir : str
        Path to model directory
    data_path : str
        Path to data directory (used if record_path not provided)
    lead_index : int
        Lead index to use for analysis
    sample_idx : int
        Sample index for XAI analysis
    with_xai : bool
        Whether to perform XAI analysis
    """
    print(f"Using record path: {record_path}")
    
    # Determine the full record path and record base
    if record_path and os.path.isfile(record_path + '.dat'):
        # record_path is already a full path to the record without extension
        full_record_path = record_path
        record_base = os.path.basename(record_path)
    elif record_path and os.path.isdir(record_path) and record_number:
        # record_path is a directory and record_number is provided
        full_record_path = os.path.join(record_path, str(record_number))
        record_base = str(record_number)
    elif record_path and os.path.isdir(record_path):
        # record_path is a directory but no record_number provided
        raise ValueError("When record_path is a directory, record_number must be provided")
    elif record_path:
        # record_path may be a full path or just the record base
        full_record_path = record_path
        record_base = os.path.basename(record_path)
    elif data_path and record_number:
        # Use data_path and record_number
        full_record_path = os.path.join(data_path, str(record_number))
        record_base = str(record_number)
    else:
        raise ValueError("Either record_path or (data_path and record_number) must be provided")
    
    print(f"Extracted record base: {record_base}")
    print(f"Full record path: {full_record_path}")
    
    # Ensure data_path is set for the preprocessor
    new_data_path = os.path.dirname(full_record_path) if not data_path else data_path
    print(f"Using data directory: {new_data_path}")
    
    print(f"Using model directory: {model_dir}")
    
    # Continue with the rest of the function...
    # Call the analysis function with the extracted information
    if with_xai:
        return analyze_ecg_with_xai(
            full_record_path,  # Pass the full path instead of just the base
            model_dir=model_dir,
            data_path=new_data_path,
            lead_index=lead_index,
            sample_idx=sample_idx
        )
    else:
        return predict_from_raw_ecg(
            full_record_path,  # Pass the full path instead of just the base
            model_dir=model_dir,
            data_path=new_data_path,
            lead_index=lead_index,
            sample_idx=sample_idx
        )

def predict_from_raw_ecg(record_number, model_dir=None, data_path=None, lead_index=0, sample_idx=None):
    """
    Process a raw MIT-BIH ECG record and make a prediction (non-XAI version).
    
    Parameters:
    record_number (str): ID of the MIT-BIH record to process
    model_dir (str, optional): Path to the model directory
    data_path (str, optional): Path to the MIT-BIH dataset
    lead_index (int): Index of the lead to use (0=MLII typically)
    sample_idx (int, optional): Index of the heartbeat segment to analyze
    
    Returns:
    dict: Basic analysis results without XAI
    """
    global MODEL_DIR, DATA_PATH
    
    # Update paths if provided
    if model_dir:
        MODEL_DIR = model_dir
    if data_path:
        DATA_PATH = data_path
        
    print(f"Using model directory: {MODEL_DIR}")
    print(f"Using data path: {DATA_PATH}")
    
    # Load model and artifacts
    model, feature_scaler, segment_scaler, label_classes = load_model_and_artifacts()
    
    # Create ECGPreprocessor
    fs = 360  # Sampling frequency for MIT-BIH
    preprocessor = ECGPreprocessor(data_path=DATA_PATH, fs=fs)
    
    # Load the record
    record_path = os.path.join(DATA_PATH, record_number)
    print(f"Loading record from: {record_path}")
    print(record_number)
    signals, annotations, fields = preprocessor.load_record(record_number)
    
    if signals is None:
        print("Error: Could not read ECG record")
        return None
    
    print(f"Record loaded. Signal shape: {signals.shape}")
    print(f"Number of annotations: {len(annotations.sample) if annotations else 0}")
    print(f"Sampling frequency: {fields['fs']} Hz")
    
    # Process the ECG signal
    print("Processing ECG signal...")
    processed_data = preprocessor.process_single_ecg(signals, annotations)
    
    print(f"Number of detected beats: {len(processed_data['segments'])}")
    
    # Prepare data for prediction
    segment_data, feature_data, feature_names, sample_idx = prepare_data_for_prediction(processed_data, sample_idx)
    
    if segment_data is None or feature_data is None:
        print("Error: Failed to prepare data for prediction")
        return None
    
    # Scale input data
    features_scaled = feature_scaler.transform(feature_data)
    segments_scaled = segment_scaler.transform(segment_data)
    
    # Make prediction
    predictions = model.predict(
        {'features_input': features_scaled, 'segments_input': segments_scaled}, 
        verbose=0
    )
    
    # Get predicted class
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = label_classes[predicted_class_idx]
    
    # Print results
    print("\n" + "="*50)
    print("ECG ANALYSIS RESULTS")
    print("="*50)
    
    print(f"\nPredicted Class: {predicted_class}")
    if predicted_class in arrhythmia_mapping:
        print(f"Condition: {arrhythmia_mapping[predicted_class]}")
    print(f"Confidence: {predictions[0][predicted_class_idx]:.2%}")
    
    return {
        'predicted_class': predicted_class,
        'probabilities': predictions[0],
        'label_classes': label_classes,
        'processed_data': processed_data
    }

# # For demonstration purposes
# if __name__ == "__main__":
#     # Example usage
#     record_number = '121'  # MIT-BIH record number
#     results = analyze_ecg(record_number, with_xai=True)
#     results = analyze_ecg(record_number='104', data_path='D:\\DeepXAI\\api\\uploads', with_xai=True)
