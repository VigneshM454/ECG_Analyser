import numpy as np
import pandas as pd
import os
import wfdb
import matplotlib.pyplot as plt
from scipy import signal as sig
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.model_selection import train_test_split

class ECGPreprocessor:
    """Comprehensive ECG preprocessing for both dataset batch processing and single sample prediction"""
    
    def __init__(self, data_path=None, fs=360, model_path=None):
        """
        Initialize the ECG preprocessor
        
        Parameters:
        -----------
        data_path : str, optional
            Path to ECG dataset directory (required for batch processing)
        fs : int
            Sampling frequency in Hz (default 360 for MIT-BIH)
        model_path : str, optional
            Path to directory containing the fitted scaler
        """
        self.data_path = data_path
        self.fs = fs
        self.model_path = model_path
        
        # ECG beat annotations
        self.beat_annotations = {
            'N': 'Normal',
            'L': 'Left bundle branch block',
            'R': 'Right bundle branch block',
            'A': 'Atrial premature',
            'a': 'Aberrated atrial premature',
            'J': 'Nodal premature',
            'S': 'Supraventricular premature',
            'V': 'Premature ventricular contraction',
            'F': 'Fusion of ventricular and normal',
            'e': 'Atrial escape',
            'j': 'Nodal escape',
            'E': 'Ventricular escape',
            '/': 'Paced',
            'f': 'Fusion of paced and normal',
            'Q': 'Unclassifiable'
        }
        
        self.common_classes = ['N', 'L', 'R', 'A', 'V']
        self.scaler = StandardScaler()
        self.feature_scaler_fitted = False
        self.label_mapping = None
        
        # Try to load scaler if model_path is provided
        if model_path and os.path.exists(os.path.join(model_path, 'scaler.pkl')):
            self.load_scaler(os.path.join(model_path, 'scaler.pkl'))
        
        if data_path:
            self.record_names = self._get_record_names()
    
    def _get_record_names(self):
        """Get list of record names in the database"""
        record_names = [f for f in os.listdir(self.data_path) if f.endswith('.dat')]
        return sorted(list(set([os.path.splitext(f)[0] for f in record_names])))
    
    def load_record(self, record_name):
        """Load a single record from the dataset"""
        print('from load_record',self.data_path)
        record_path = os.path.join(self.data_path, record_name)
        signals, fields = wfdb.rdsamp(record_path)
        annotations = wfdb.rdann(record_path, 'atr')
        return signals, annotations, fields


    # def load_record(self, record_name):
    #     """Load a single record from the dataset"""
    #     print(".........................")
    #     print('from load_record', self.data_path, record_name)
    #     # Check if record_name is already a full path
    #     if os.path.exists(record_name + '.dat'):
    #         record_path = record_name
    #     else:
    #         # Otherwise join with data_path
    #         record_path = os.path.join(self.data_path, record_name)
    #     # Add debug print to see what path is being used
    #     print(f"Attempting to load record from: {record_path}")
        
    #     signals, fields = wfdb.rdsamp(record_path)
    #     annotations = wfdb.rdann(record_path, 'atr')
    #     return signals, annotations, fields

    # def load_record(self, record_name):
    #     """Load a single record from the dataset"""
    #     print(".........................")
    #     print('from load_record', self.data_path, record_name)
        
    #     # Extract the base name without path or extension
    #     base_name = os.path.splitext(os.path.basename(record_name))[0]
        
    #     # Check if the record exists directly
    #     if os.path.exists(record_name + '.dat'):
    #         record_path = record_name
    #     else:
    #         # Otherwise join with data_path
    #         record_path = os.path.join(self.data_path, base_name)
        
    #     # Add debug print to see what path is being used
    #     print(f"Attempting to load record from: {record_path}")
        
    #     try:
    #         signals, fields = wfdb.rdsamp(record_path)
    #         annotations = wfdb.rdann(record_path, 'atr')
    #         return signals, annotations, fields
    #     except Exception as e:
    #         print(f"Error loading record {record_path}: {str(e)}")
    #         # Try to find what files actually exist
    #         print(f"Looking for files with pattern: {record_path}.*")
    #         import glob
    #         print("Existing files:", glob.glob(record_path + ".*"))
    #         raise

    # def load_record(self, record_name):
    #     """Load a single record from the dataset"""
    #     base_name = os.path.splitext(os.path.basename(record_name))[0]
    #     dat_file = os.path.join(self.data_path, f"{base_name}.dat")
    #     hea_file = os.path.join(self.data_path, f"{base_name}.hea")
        
    #     if not os.path.exists(dat_file):
    #         raise FileNotFoundError(f"Data file not found: {dat_file}")
    #     if not os.path.exists(hea_file):
    #         raise FileNotFoundError(f"Header file not found: {hea_file}")
        
    #     # Read using the full paths
    #     signals, fields = wfdb.rdsamp(record_name=base_name, pn_dir=self.data_path)
    #     annotations = wfdb.rdann(record_name=base_name, extension='atr', pn_dir=self.data_path)
    #     return signals, annotations, fields
    
    # def load_record(self, record_name):
    #     wfdb.io._cache.clear_cache()
    #     """Load a single record from the dataset"""
    #     print("hey im execured")
    #     print(".........................")
    #     print('from load_record', self.data_path, record_name)
    #     print(os.path)
    #     # Extract just the base name (454) without path or extension
    #     base_name = os.path.splitext(os.path.basename(record_name))[0]
    #     print(base_name)
    #     # Debug: Print what files actually exist
    #     print(f"Checking for files with base: {base_name}")
    #     import glob
    #     existing_files = glob.glob(os.path.join(self.data_path, f"{base_name}.*"))
    #     print(f"Existing files: {existing_files}")
        
    #     try:
    #         # Explicitly use pn_dir parameter to specify the directory
    #         signals, fields = wfdb.rdsamp(record_name=base_name, pn_dir=self.data_path)
    #         annotations = wfdb.rdann(record_name=base_name, extension='atr', pn_dir=self.data_path)
    #         return signals, annotations, fields
    #     except Exception as e:
    #         print(f"Error loading record {base_name}: {str(e)}")
    #         raise

    # def load_record(self, record_name):
    #     """Load a single record from the dataset"""
    #     print("Debug - Starting load_record")
        
    #     # Clear any WFDB cache first
    #     # wfdb.io._cache.clear_cache()
        
    #     # Normalize paths for Windows
    #     self.data_path = os.path.normpath(self.data_path)
    #     record_name = os.path.normpath(record_name)
        
    #     # Extract base name
    #     base_name = os.path.splitext(os.path.basename(record_name))[0]
    #     print(f"Debug - Base name: {base_name}")
        
    #     # Verify files exist
    #     required_files = [f"{base_name}.hea", f"{base_name}.dat"]
    #     missing_files = [f for f in required_files if not os.path.exists(os.path.join(self.data_path, f))]
        
    #     if missing_files:
    #         raise FileNotFoundError(f"Missing required files: {missing_files}")
        
    #     # Debug: Print header file contents
    #     hea_path = os.path.join(self.data_path, f"{base_name}.hea")
    #     with open(hea_path, 'r') as f:
    #         print(f"Header file contents:\n{f.read()}")
        
    #     try:
    #         print(f"Attempting to read record {base_name} from {self.data_path}")
            
    #         # Method 1: Try with pn_dir first
    #         try:
    #             signals, fields = wfdb.rdsamp(record_name=base_name, pn_dir=self.data_path)
    #             annotations = wfdb.rdann(record_name=base_name, extension='atr', pn_dir=self.data_path)
    #             return signals, annotations, fields
    #         except Exception as e:
    #             print(f"Method 1 failed: {str(e)}")
                
    #             # Method 2: Try with full path
    #             signals, fields = wfdb.rdsamp(os.path.join(self.data_path, base_name))
    #             annotations = wfdb.rdann(os.path.join(self.data_path, base_name), 'atr')
    #             return signals, annotations, fields
                
    #     except Exception as e:
    #         print(f"Final error loading record: {str(e)}")
    #         print(f"WFDB searched for: {os.path.join(self.data_path, '100.dat')}")
    #         raise RuntimeError(f"Could not load record {base_name}. WFDB is incorrectly looking for 100.dat") from e
            
    def filter_signal(self, ecg_signal, lowcut=0.5, highcut=50, order=5):
        """Bandpass filter the ECG signal"""
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = sig.butter(order, [low, high], btype='band')
        return sig.filtfilt(b, a, ecg_signal)
    
    def detect_r_peaks(self, ecg_signal, lead=0):
        """
        Detect R-peaks in ECG signal using Pan-Tompkins algorithm
        
        Parameters:
        -----------
        ecg_signal : ndarray
            ECG signal (samples x leads)
        lead : int
            Which lead to use for R-peak detection
            
        Returns:
        --------
        r_peaks : ndarray
            Array of R-peak sample locations
        """
        # Select lead for detection
        signal = ecg_signal[:, lead] if len(ecg_signal.shape) > 1 else ecg_signal
        
        # Filter to emphasize QRS complex
        b, a = sig.butter(2, [5/self.fs*2, 15/self.fs*2], btype='bandpass')
        filtered = sig.filtfilt(b, a, signal)
        
        # Derivative
        derivative = np.diff(filtered)
        squared = derivative ** 2
        
        # Moving window integration
        window_size = int(0.150 * self.fs)  # 150 ms window
        integrated = np.convolve(squared, np.ones(window_size)/window_size, mode='same')
        
        # Adaptive threshold
        threshold = 0.5 * np.mean(integrated)
        
        # Find peaks
        peaks, _ = sig.find_peaks(integrated, height=threshold, distance=int(0.2*self.fs))
        
        return peaks
    
    def segment_beats(self, signals, annotations=None, r_peaks=None, window=250):
        """
        Segment ECG signals into beats
        
        Parameters:
        -----------
        signals : ndarray
            ECG signals (samples x leads)
        annotations : wfdb.Annotation, optional
            Beat annotations (preferred if available)
        r_peaks : list, optional
            R-peak locations if annotations not available
        window : int
            Window size around R-peak (default 250 samples)
            
        Returns:
        --------
        segments : list of ndarray
            Extracted beat segments
        labels : list
            Beat labels (or 'N' if unknown)
        """
        segments = []
        labels = []
        
        # Create dummy annotations if only r_peaks provided
        if annotations is None and r_peaks is not None:
            annotations = type('Annotations', (), {'symbol': ['N']*len(r_peaks), 'sample': r_peaks})
        elif annotations is None:
            # Auto-detect R-peaks if neither is provided
            r_peaks = self.detect_r_peaks(signals)
            annotations = type('Annotations', (), {'symbol': ['N']*len(r_peaks), 'sample': r_peaks})
        
        for i, annot in enumerate(annotations.symbol):
            label = annot if annot in self.beat_annotations else 'N'
            sample = annotations.sample[i]
            
            start = max(0, sample - window//2)
            end = min(len(signals), sample + window//2)
            
            if end - start == window:
                segments.append(signals[start:end, :])
                labels.append(label)
        
        return segments, labels
    
    def extract_features(self, segment):
        """Extract features from a single beat segment"""
        features = {}
        
        for lead in range(segment.shape[1]):
            lead_signal = segment[:, lead]
            
            # Time domain features
            features[f'mean_lead{lead}'] = np.mean(lead_signal)
            features[f'std_lead{lead}'] = np.std(lead_signal)
            features[f'min_lead{lead}'] = np.min(lead_signal)
            features[f'max_lead{lead}'] = np.max(lead_signal)
            features[f'ptp_lead{lead}'] = np.ptp(lead_signal)
            features[f'rms_lead{lead}'] = np.sqrt(np.mean(np.square(lead_signal)))
            
            # Statistical features
            std = np.std(lead_signal)
            if std > 0:
                centered = lead_signal - np.mean(lead_signal)
                features[f'skewness_lead{lead}'] = np.mean(centered**3) / (std**3)
                features[f'kurtosis_lead{lead}'] = np.mean(centered**4) / (std**4)
            else:
                features[f'skewness_lead{lead}'] = 0
                features[f'kurtosis_lead{lead}'] = 0
            
            # Frequency domain features
            freqs, psd = sig.welch(lead_signal, fs=self.fs, nperseg=min(len(lead_signal), 128))
            features[f'psd_max_lead{lead}'] = np.max(psd)
            features[f'psd_mean_lead{lead}'] = np.mean(psd)
            
            # Energy in frequency bands
            features[f'energy_low_lead{lead}'] = np.sum(psd[(freqs >= 0.5) & (freqs <= 8)])
            features[f'energy_mid_lead{lead}'] = np.sum(psd[(freqs > 8) & (freqs <= 20)])
            features[f'energy_high_lead{lead}'] = np.sum(psd[(freqs > 20) & (freqs <= 45)])
            
            # Morphology features
            features[f'crossing_zero_lead{lead}'] = np.sum(np.diff(np.signbit(lead_signal).astype(int)) != 0)
            
            # Add QRS interval estimates
            # Simplified feature - width of peak
            over_threshold = np.where(lead_signal > 0.7 * np.max(lead_signal))[0]
            if len(over_threshold) > 1:
                features[f'qrs_width_lead{lead}'] = over_threshold[-1] - over_threshold[0]
            else:
                features[f'qrs_width_lead{lead}'] = 0
                
        return features
    
    def save_scaler(self, path):
        """Save fitted scaler for later use"""
        if not self.feature_scaler_fitted:
            raise ValueError("Scaler not fitted yet. Process data first.")
            
        import joblib
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.scaler, path)
        
    def load_scaler(self, path):
        """Load pre-fitted scaler"""
        import joblib
        self.scaler = joblib.load(path)
        self.feature_scaler_fitted = True
        
    def normalize_features(self, features_df):
        """Normalize features using fitted scaler"""
        feature_cols = [col for col in features_df.columns 
                       if col not in ['record', 'beat_idx', 'label', 'label_name', 'label_int']]
        
        if not self.feature_scaler_fitted:
            self.scaler.fit(features_df[feature_cols])
            self.feature_scaler_fitted = True
            
        features_df[feature_cols] = self.scaler.transform(features_df[feature_cols])
        return features_df
    
    def process_dataset(self, save_path='preprocessed'):
        """Process entire dataset (batch mode)"""
        os.makedirs(save_path, exist_ok=True)
        
        all_features = []
        all_segments = []
        all_labels = []
        
        print(f"Processing {len(self.record_names)} records...")
        
        for record_name in tqdm(self.record_names):
            try:
                signals, annotations, _ = self.load_record(record_name)
                
                # Filter signals
                filtered_signals = np.zeros_like(signals)
                for i in range(signals.shape[1]):
                    filtered_signals[:, i] = self.filter_signal(signals[:, i])
                
                # Segment beats
                segments, labels = self.segment_beats(filtered_signals, annotations)
                
                # Extract features
                for i, (segment, label) in enumerate(zip(segments, labels)):
                    features = self.extract_features(segment)
                    features.update({
                        'record': record_name,
                        'beat_idx': i,
                        'label': label,
                        'label_name': self.beat_annotations.get(label, 'Unknown')
                    })
                    all_features.append(features)
                    all_segments.append(segment)
                    all_labels.append(label)
                    
            except Exception as e:
                print(f"Error processing {record_name}: {str(e)}")
                continue
        
        # Create DataFrames
        df_features = pd.DataFrame(all_features).dropna()
        
        segments_dict = {
            'record': [f['record'] for f in all_features],
            'beat_idx': [f['beat_idx'] for f in all_features],
            'label': all_labels,
            'label_name': [self.beat_annotations.get(l, 'Unknown') for l in all_labels]
        }
        
        # Add flattened segments
        for i, segment in enumerate(all_segments):
            for j, val in enumerate(segment.flatten()):
                key = f'sample_{j}'
                if key not in segments_dict:
                    segments_dict[key] = [None] * len(all_segments)
                segments_dict[key][i] = val
        
        df_segments = pd.DataFrame(segments_dict).dropna()
        
        # Save raw data
        df_features.to_csv(os.path.join(save_path, 'ecg_features.csv'), index=False)
        df_segments.to_csv(os.path.join(save_path, 'ecg_segments.csv'), index=False)
        
        # Prepare balanced training data
        training_data = self._prepare_training_data(df_features, df_segments, save_path)
        
        # Save scaler for future use
        self.save_scaler(os.path.join(save_path, 'scaler.pkl'))
        
        return df_features, df_segments, training_data
    
    def _prepare_training_data(self, df_features, df_segments, save_path):
        """Prepare balanced training data from processed dataset"""
        # Balance classes
        min_class_count = df_features['label'].value_counts().min()
        if min_class_count < 10:
            top_classes = df_features['label'].value_counts().nlargest(3).index.tolist()
            df_features = df_features[df_features['label'].isin(top_classes)]
            min_class_count = df_features['label'].value_counts().min()
        
        balanced_indices = []
        for label in df_features['label'].unique():
            indices = np.random.choice(
                df_features[df_features['label'] == label].index,
                min_class_count,
                replace=False
            )
            balanced_indices.extend(indices)
        
        df_features_balanced = df_features.loc[balanced_indices].copy()
        
        # Match segments
        segment_indices = []
        for _, row in df_features_balanced.iterrows():
            idx = df_segments[(df_segments['record'] == row['record']) & 
                            (df_segments['beat_idx'] == row['beat_idx'])].index
            if len(idx) > 0:
                segment_indices.append(idx[0])
        
        df_segments_balanced = df_segments.loc[segment_indices].copy()
        
        # Normalize features
        df_features_balanced = self.normalize_features(df_features_balanced)
        
        # Create label mapping
        label_mapping = {label: i for i, label in enumerate(df_features_balanced['label'].unique())}
        self.label_mapping = label_mapping
        
        for df in [df_features_balanced, df_segments_balanced]:
            df['label_int'] = df['label'].map(label_mapping)
        
        # Save balanced data
        df_features_balanced.to_csv(os.path.join(save_path, 'features_balanced.csv'), index=False)
        df_segments_balanced.to_csv(os.path.join(save_path, 'segments_balanced.csv'), index=False)
        pd.DataFrame(label_mapping.items(), columns=['label', 'label_int']).to_csv(
            os.path.join(save_path, 'label_mapping.csv'), index=False)
        
        # Create train-test splits
        feature_cols = [c for c in df_features_balanced.columns 
                       if c not in ['record', 'beat_idx', 'label', 'label_name', 'label_int']]
        X_train, X_test, y_train, y_test = train_test_split(
            df_features_balanced[feature_cols], df_features_balanced['label_int'],
            test_size=0.2, random_state=42, stratify=df_features_balanced['label_int']
        )
        
        # Save splits
        pd.concat([X_train, y_train], axis=1).to_csv(os.path.join(save_path, 'train_features.csv'), index=False)
        pd.concat([X_test, y_test], axis=1).to_csv(os.path.join(save_path, 'test_features.csv'), index=False)
        
        return {
            'features': (X_train, X_test, y_train, y_test),
            'label_mapping': label_mapping
        }
    
    def process_single_ecg(self, ecg_signal, annotations=None, r_peaks=None):
        """
        Process a single ECG signal for real-time prediction
        
        Parameters:
        -----------
        ecg_signal : ndarray
            ECG signal (samples x leads)
        annotations : wfdb.Annotation, optional
            Beat annotations if available
        r_peaks : list, optional
            R-peak locations if annotations not available
            
        Returns:
        --------
        dict containing:
            - segments: list of beat segments
            - labels: beat labels
            - features: DataFrame of extracted features
            - filtered_signal: filtered ECG signal
        """
        # Ensure 2D array
        if len(ecg_signal.shape) == 1:
            ecg_signal = ecg_signal.reshape(-1, 1)
        
        # Filter signal
        filtered = np.zeros_like(ecg_signal)
        for i in range(ecg_signal.shape[1]):
            filtered[:, i] = self.filter_signal(ecg_signal[:, i])
        
        # Auto-detect R-peaks if not provided
        if annotations is None and r_peaks is None:
            r_peaks = self.detect_r_peaks(filtered)
        
        # Segment beats
        segments, labels = self.segment_beats(filtered, annotations, r_peaks)
        
        # Extract features
        features = []
        for i, (segment, label) in enumerate(zip(segments, labels)):
            feat = self.extract_features(segment)
            feat.update({
                'beat_idx': i,
                'label': label,
                'label_name': self.beat_annotations.get(label, 'Unknown')
            })
            features.append(feat)
        
        df_features = pd.DataFrame(features)
        
        # Normalize features if scaler was fitted
        if self.feature_scaler_fitted:
            feature_cols = [c for c in df_features.columns 
                          if c not in ['beat_idx', 'label', 'label_name']]
            df_features[feature_cols] = self.scaler.transform(df_features[feature_cols])
        
        return {
            'segments': segments,
            'labels': labels,
            'features': df_features,
            'filtered_signal': filtered,
            'r_peaks': r_peaks if r_peaks is not None else []
        }
    
    def prepare_for_prediction(self, processed_data):
        """
        Prepare processed data for model prediction
        
        Parameters:
        -----------
        processed_data : dict
            Output from process_single_ecg()
            
        Returns:
        --------
        feature_matrix : ndarray
            Feature matrix ready for model prediction
        """
        # Get feature columns (exclude non-feature columns)
        df = processed_data['features']
        feature_cols = [c for c in df.columns 
                      if c not in ['beat_idx', 'label', 'label_name', 'label_int']]
        
        # Return as numpy array
        return df[feature_cols].values
    
    def visualize_ecg(self, ecg_signal, processed_data=None, lead=0, title="ECG Signal"):
        """
        Visualize ECG signal and processing results
        
        Parameters:
        -----------
        ecg_signal : ndarray
            Raw ECG signal
        processed_data : dict, optional
            Output from process_single_ecg()
        lead : int
            Which lead to visualize
        title : str
            Plot title
        
        Returns:
        --------
        fig : matplotlib figure
            Figure object (can be saved or displayed)
        """
        fig = plt.figure(figsize=(15, 8))
        
        if processed_data is None:
            # Just plot raw signal
            plt.plot(ecg_signal[:, lead] if len(ecg_signal.shape) > 1 else ecg_signal)
            plt.title(f"{title} (Lead {lead})")
            plt.xlabel("Samples")
            plt.ylabel("Amplitude")
        else:
            # Plot comparison
            plt.subplot(2, 1, 1)
            raw_signal = ecg_signal[:, lead] if len(ecg_signal.shape) > 1 else ecg_signal
            plt.plot(raw_signal)
            plt.title(f"Original {title} (Lead {lead})")
            
            # Mark R peaks if available
            if 'r_peaks' in processed_data and len(processed_data['r_peaks']) > 0:
                plt.plot(processed_data['r_peaks'], raw_signal[processed_data['r_peaks']], 'ro', 
                         label='R-peaks')
                plt.legend()
            
            plt.subplot(2, 1, 2)
            filtered_signal = processed_data['filtered_signal'][:, lead] if len(processed_data['filtered_signal'].shape) > 1 else processed_data['filtered_signal']
            plt.plot(filtered_signal)
            plt.title("Filtered Signal with Detected Beats")
            
            # Mark beat locations
            segments = processed_data['segments']
            labels = processed_data['labels']
            
            # If more than 5 segments, only show the first 5
            display_count = min(5, len(segments))
            for i in range(display_count):
                segment = segments[i]
                seg_lead = segment[:, lead] if len(segment.shape) > 1 else segment
                label = labels[i]
                plt.plot(range(i*20, i*20 + len(seg_lead)), 
                         seg_lead + (i * 0.5),  # Offset for visibility
                         label=f"Beat {i+1} ({label})")
            
            plt.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_sample_beats(self, segments, labels, save_to=None):
        """
        Plot sample beats from each class
        
        Parameters:
        -----------
        segments : list of ndarray
            Beat segments
        labels : list
            Corresponding beat labels
        save_to : str, optional
            File path to save the plot
            
        Returns:
        --------
        fig : matplotlib figure
            Figure object (can be saved or displayed)
        """
        fig = plt.figure(figsize=(15, 10))
        
        # Get unique classes present
        present_classes = sorted(set(labels) & set(self.common_classes))
        if not present_classes:
            present_classes = sorted(set(labels))
        
        for i, label in enumerate(present_classes):
            # Get first 3 examples of this class
            class_segments = [seg for seg, lbl in zip(segments, labels) if lbl == label][:3]
            
            for j, segment in enumerate(class_segments):
                if len(class_segments) * len(present_classes) > 0:
                    plt.subplot(len(present_classes), min(3, len(class_segments)), i*min(3, len(class_segments)) + j + 1)
                    if len(segment.shape) > 1:
                        plt.plot(segment[:, 0])  # Plot first lead
                    else:
                        plt.plot(segment)
                    plt.title(f"{label} ({self.beat_annotations.get(label, 'Unknown')})")
        
        plt.tight_layout()
        
        if save_to:
            plt.savefig(save_to)
        
        return fig
    
    def load_label_mapping(self, path):
        """Load label mapping from CSV file"""
        df = pd.read_csv(path)
        self.label_mapping = dict(zip(df['label'], df['label_int']))
        return self.label_mapping
    
    def get_feature_importance(self, model, feature_names=None):
        """
        Get feature importance from trained model
        
        Parameters:
        -----------
        model : trained model object
            Model with feature_importances_ attribute (e.g., RandomForest)
        feature_names : list, optional
            List of feature names
            
        Returns:
        --------
        importance_df : DataFrame
            DataFrame with feature importance scores
        """
        if not hasattr(model, 'feature_importances_'):
            raise ValueError("Model does not have feature_importances_ attribute")
            
        importances = model.feature_importances_
        
        if feature_names is None:
            # Get feature names from a typical feature dataframe
            sample_features = self.extract_features(np.random.randn(250, 2))
            feature_names = list(sample_features.keys())
        
        # Sort by importance
        indices = np.argsort(importances)[::-1]
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': [feature_names[i] for i in indices],
            'Importance': importances[indices]
        })
        
        return importance_df