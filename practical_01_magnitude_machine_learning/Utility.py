"""
Utility.py - Earthquake Magnitude Prediction Utilities
Purpose: Created for MSc Seismology Course
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class EarthquakeWaveformGenerator:
    """
    Generate synthetic earthquake waveforms based on simplified seismic wave physics
    """
    
    def __init__(self, sampling_rate=100, duration=60):
        """
        Initialize the waveform generator
        
        Parameters:
        -----------
        sampling_rate : int
            Sampling rate in Hz (default: 100)
        duration : int
            Duration of waveform in seconds (default: 60)
        """
        self.sampling_rate = sampling_rate  # Hz
        self.duration = duration  # seconds
        self.n_samples = int(sampling_rate * duration)
        self.time = np.linspace(0, duration, self.n_samples)
    
    def generate_p_wave(self, magnitude, distance=50):
        """
        Generate P-wave component
        
        Parameters:
        -----------
        magnitude : float
            Earthquake magnitude
        distance : float
            Distance from source in km (default: 50)
            
        Returns:
        --------
        numpy.ndarray
            P-wave signal
        """
        # P-wave arrives first, higher frequency
        amplitude = 10**(magnitude - 3) / (distance**0.5)  # Amplitude scaling with magnitude and distance
        frequency = np.random.uniform(5, 15)  # P-wave frequency range
        
        # P-wave arrival time (simplified)
        p_arrival = distance / 6.0  # km/s average P-wave velocity
        
        if p_arrival < self.duration:
            p_start_idx = int(p_arrival * self.sampling_rate)
            p_wave = np.zeros(self.n_samples)
            
            # Create P-wave signal with exponential decay
            p_duration = min(10, self.duration - p_arrival)
            p_samples = int(p_duration * self.sampling_rate)
            
            t_p = np.linspace(0, p_duration, p_samples)
            envelope = np.exp(-t_p / 3.0)  # Exponential decay
            p_signal = amplitude * envelope * np.sin(2 * np.pi * frequency * t_p)
            
            # Add some noise and complexity
            p_signal += 0.1 * amplitude * np.random.randn(len(p_signal))
            
            end_idx = min(p_start_idx + len(p_signal), self.n_samples)
            p_wave[p_start_idx:end_idx] = p_signal[:end_idx-p_start_idx]
            
            return p_wave
        return np.zeros(self.n_samples)
    
    def generate_s_wave(self, magnitude, distance=50):
        """
        Generate S-wave component
        
        Parameters:
        -----------
        magnitude : float
            Earthquake magnitude
        distance : float
            Distance from source in km (default: 50)
            
        Returns:
        --------
        numpy.ndarray
            S-wave signal
        """
        # S-wave arrives later, lower frequency, larger amplitude
        amplitude = 1.5 * 10**(magnitude - 3) / (distance**0.5)  # S-waves typically larger
        frequency = np.random.uniform(2, 8)  # S-wave frequency range
        
        # S-wave arrival time
        s_arrival = distance / 3.5  # km/s average S-wave velocity
        
        if s_arrival < self.duration:
            s_start_idx = int(s_arrival * self.sampling_rate)
            s_wave = np.zeros(self.n_samples)
            
            # Create S-wave signal with exponential decay
            s_duration = min(20, self.duration - s_arrival)
            s_samples = int(s_duration * self.sampling_rate)
            
            t_s = np.linspace(0, s_duration, s_samples)
            envelope = np.exp(-t_s / 5.0)  # Slower decay than P-wave
            s_signal = amplitude * envelope * np.sin(2 * np.pi * frequency * t_s)
            
            # Add some noise and complexity
            s_signal += 0.15 * amplitude * np.random.randn(len(s_signal))
            
            end_idx = min(s_start_idx + len(s_signal), self.n_samples)
            s_wave[s_start_idx:end_idx] = s_signal[:end_idx-s_start_idx]
            
            return s_wave
        return np.zeros(self.n_samples)
    
    def generate_waveform(self, magnitude, distance=None):
        """
        Generate complete synthetic seismic waveform
        
        Parameters:
        -----------
        magnitude : float
            Earthquake magnitude
        distance : float, optional
            Distance from source in km (random if None)
            
        Returns:
        --------
        tuple
            (waveform, distance) - waveform array and actual distance used
        """
        if distance is None:
            distance = np.random.uniform(10, 200)  # Random distance 10-200 km
        
        # Generate P and S waves
        p_wave = self.generate_p_wave(magnitude, distance)
        s_wave = self.generate_s_wave(magnitude, distance)
        
        # Combine waves
        waveform = p_wave + s_wave
        
        # Add background noise
        noise_level = 0.05 * np.max(np.abs(waveform)) if np.max(np.abs(waveform)) > 0 else 0.01
        noise = noise_level * np.random.randn(self.n_samples)
        waveform += noise
        
        return waveform, distance

def generate_dataset(n_samples=1000, random_seed=42):
    """
    Generate synthetic earthquake dataset
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate (default: 1000)
    random_seed : int
        Random seed for reproducibility (default: 42)
        
    Returns:
    --------
    tuple
        (waveforms, magnitudes, distances) - arrays of generated data
    """
    np.random.seed(random_seed)
    tf.keras.utils.set_random_seed(random_seed)
    
    generator = EarthquakeWaveformGenerator(sampling_rate=100, duration=60)
    
    waveforms = []
    magnitudes = []
    distances = []
    
    print(f"Generating {n_samples} synthetic earthquake waveforms...")
    
    for i in range(n_samples):
        # Generate random magnitude between 4.0 and 7.0
        magnitude = np.random.uniform(4.0, 7.0)
        
        # Generate waveform
        waveform, distance = generator.generate_waveform(magnitude)
        
        waveforms.append(waveform)
        magnitudes.append(magnitude)
        distances.append(distance)
        
        if (i + 1) % 200 == 0:
            print(f"Generated {i + 1}/{n_samples} waveforms")
    
    return np.array(waveforms), np.array(magnitudes), np.array(distances)

def extract_features(waveforms):
    """
    Extract features from waveforms for ML model
    
    Parameters:
    -----------
    waveforms : numpy.ndarray
        Array of waveforms (n_samples, waveform_length)
        
    Returns:
    --------
    numpy.ndarray
        Feature matrix (n_samples, n_features)
    """
    features = []
    
    print(f"Extracting features from {len(waveforms)} waveforms...")
    
    for i, waveform in enumerate(waveforms):
        # Time domain features
        max_amplitude = np.max(np.abs(waveform))
        rms = np.sqrt(np.mean(waveform**2))
        zero_crossings = len(np.where(np.diff(np.signbit(waveform)))[0])
        
        # Frequency domain features (simplified)
        fft = np.fft.fft(waveform)
        freq_magnitude = np.abs(fft[:len(fft)//2])
        
        # Dominant frequency
        freqs = np.fft.fftfreq(len(waveform), 1/100)[:len(fft)//2]
        dominant_freq = freqs[np.argmax(freq_magnitude)]
        
        # Spectral centroid
        spectral_centroid = np.sum(freqs * freq_magnitude) / np.sum(freq_magnitude)
        
        # Energy in different frequency bands
        low_freq_energy = np.sum(freq_magnitude[freqs < 5])
        mid_freq_energy = np.sum(freq_magnitude[(freqs >= 5) & (freqs < 15)])
        high_freq_energy = np.sum(freq_magnitude[freqs >= 15])
        
        feature_vector = [
            max_amplitude, rms, zero_crossings, dominant_freq, 
            spectral_centroid, low_freq_energy, mid_freq_energy, high_freq_energy
        ]
        
        features.append(feature_vector)
        
        if (i + 1) % 200 == 0:
            print(f"Processed {i + 1}/{len(waveforms)} waveforms")
    
    return np.array(features)

def get_feature_names():
    """
    Get the names of extracted features
    
    Returns:
    --------
    list
        List of feature names
    """
    return ['Max Amplitude', 'RMS', 'Zero Crossings', 'Dominant Freq', 
            'Spectral Centroid', 'Low Freq Energy', 'Mid Freq Energy', 'High Freq Energy']

def create_ANN_model(input_shape, random_seed=42):
    """
    Create neural network model for magnitude prediction
    
    Parameters:
    -----------
    input_shape : int
        Number of input features
    random_seed : int
        Random seed for reproducibility (default: 42)
        
    Returns:
    --------
    tensorflow.keras.Model
        Compiled neural network model
    """
    tf.keras.utils.set_random_seed(random_seed)
    
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(input_shape,), name='hidden_layer_1'),
        keras.layers.Dropout(0.3, name='dropout_1'),
        keras.layers.Dense(32, activation='relu', name='hidden_layer_2'),
        keras.layers.Dropout(0.2, name='dropout_2'),
        keras.layers.Dense(16, activation='relu', name='hidden_layer_3'),
        keras.layers.Dense(1, activation='linear', name='output_layer')  # Linear output for regression
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model

def create_1d_cnn_model(input_shape, model_complexity='medium', random_seed=42):
    """
    Create 1D CNN model for magnitude prediction
    
    Parameters:
    -----------
    input_shape : tuple
        Input shape (timesteps, features)
    model_complexity : str
        Model complexity ('simple', 'medium', 'complex')
    random_seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    tensorflow.keras.Model
        Compiled 1D CNN model
    """
    tf.keras.utils.set_random_seed(random_seed)
    
    model = keras.Sequential(name=f'EarthquakeCNN_{model_complexity}')
    
    if model_complexity == 'simple':
        # Simple CNN architecture
        model.add(layers.Conv1D(32, 3, activation='relu', input_shape=input_shape, name='conv1d_1'))
        model.add(layers.MaxPooling1D(2, name='maxpool_1'))
        model.add(layers.Conv1D(64, 3, activation='relu', name='conv1d_2'))
        model.add(layers.MaxPooling1D(2, name='maxpool_2'))
        model.add(layers.Conv1D(128, 3, activation='relu', name='conv1d_3'))
        model.add(layers.GlobalAveragePooling1D(name='global_avg_pool'))
        model.add(layers.Dense(32, activation='relu', name='dense_1'))
        model.add(layers.Dropout(0.3, name='dropout'))
        model.add(layers.Dense(1, activation='linear', name='output'))
        
    elif model_complexity == 'medium':
        # Medium complexity CNN architecture
        model.add(layers.Conv1D(32, 15, activation='relu', input_shape=input_shape, name='conv1d_1'))
        model.add(layers.BatchNormalization(name='batch_norm_1'))
        model.add(layers.MaxPooling1D(2, name='maxpool_1'))
        
        model.add(layers.Conv1D(64, 11, activation='relu', name='conv1d_2'))
        model.add(layers.BatchNormalization(name='batch_norm_2'))
        model.add(layers.MaxPooling1D(2, name='maxpool_2'))
        
        model.add(layers.Conv1D(128, 7, activation='relu', name='conv1d_3'))
        model.add(layers.BatchNormalization(name='batch_norm_3'))
        model.add(layers.MaxPooling1D(2, name='maxpool_3'))
        
        model.add(layers.Conv1D(128, 5, activation='relu', name='conv1d_4'))
        model.add(layers.GlobalAveragePooling1D(name='global_avg_pool'))
        
        model.add(layers.Dense(128, activation='relu', name='dense_1'))
        model.add(layers.Dropout(0.4, name='dropout_1'))
        model.add(layers.Dense(64, activation='relu', name='dense_2'))
        model.add(layers.Dropout(0.3, name='dropout_2'))
        model.add(layers.Dense(1, activation='linear', name='output'))
        
    elif model_complexity == 'complex':
        # Complex CNN architecture with residual-like connections
        # Input layer
        inputs = layers.Input(shape=input_shape, name='input')
        
        # First conv block
        x = layers.Conv1D(64, 15, padding='same', activation='relu', name='conv1d_1')(inputs)
        x = layers.BatchNormalization(name='batch_norm_1')(x)
        x = layers.MaxPooling1D(2, name='maxpool_1')(x)
        
        # Second conv block with skip connection
        residual = x
        x = layers.Conv1D(128, 11, padding='same', activation='relu', name='conv1d_2')(x)
        x = layers.BatchNormalization(name='batch_norm_2')(x)
        x = layers.Conv1D(128, 11, padding='same', activation='relu', name='conv1d_3')(x)
        x = layers.BatchNormalization(name='batch_norm_3')(x)
        
        # Skip connection (adjust dimensions if needed)
        if residual.shape[-1] != x.shape[-1]:
            residual = layers.Conv1D(128, 1, padding='same', name='residual_conv')(residual)
        x = layers.Add(name='residual_add_1')([x, residual])
        x = layers.MaxPooling1D(2, name='maxpool_2')(x)
        
        # Third conv block
        x = layers.Conv1D(256, 7, padding='same', activation='relu', name='conv1d_4')(x)
        x = layers.BatchNormalization(name='batch_norm_4')(x)
        x = layers.MaxPooling1D(2, name='maxpool_3')(x)
        
        # Fourth conv block
        x = layers.Conv1D(512, 5, padding='same', activation='relu', name='conv1d_5')(x)
        x = layers.BatchNormalization(name='batch_norm_5')(x)
        
        # Global pooling and dense layers
        x = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)
        x = layers.Dense(256, activation='relu', name='dense_1')(x)
        x = layers.Dropout(0.5, name='dropout_1')(x)
        x = layers.Dense(128, activation='relu', name='dense_2')(x)
        x = layers.Dropout(0.4, name='dropout_2')(x)
        x = layers.Dense(64, activation='relu', name='dense_3')(x)
        x = layers.Dropout(0.3, name='dropout_3')(x)
        outputs = layers.Dense(1, activation='linear', name='output')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='EarthquakeCNN_complex')
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def evaluate_model(y_true, y_pred):
    """
    Evaluate model performance
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True values
    y_pred : numpy.ndarray
        Predicted values
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'rmse': np.sqrt(mse)
    }
    
    return metrics

def plot_waveform_examples(waveforms, magnitudes, distances, n_examples=4):
    """
    Plot example waveforms
    
    Parameters:
    -----------
    waveforms : numpy.ndarray
        Array of waveforms
    magnitudes : numpy.ndarray
        Array of magnitudes
    distances : numpy.ndarray
        Array of distances
    n_examples : int
        Number of examples to plot (default: 4)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    time = np.linspace(0, 60, len(waveforms[0]))
    
    indices = np.linspace(0, len(waveforms)-1, n_examples, dtype=int)
    
    for i, (ax, idx) in enumerate(zip(axes.flat, indices)):
        ax.plot(time, waveforms[idx])
        ax.set_title(f'Waveform {idx}: Magnitude {magnitudes[idx]:.2f}, Distance {distances[idx]:.1f} km')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_prediction_results(y_true, y_pred, title="Magnitude Prediction Results", save_title = None):
    """
    Plot prediction results
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True values
    y_pred : numpy.ndarray
        Predicted values
    title : str
        Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter plot of predicted vs actual
    ax1.scatter(y_true, y_pred, alpha=0.6)
    ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    ax1.set_xlabel('Actual Magnitude')
    ax1.set_ylabel('Predicted Magnitude')
    ax1.set_title(f'{title} - Scatter Plot')
    ax1.grid(True, alpha=0.3)
    
    # Add R² score to the plot
    r2 = r2_score(y_true, y_pred)
    ax1.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax1.transAxes, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Residuals plot
    residuals = y_pred - y_true
    ax2.scatter(y_true, residuals, alpha=0.6)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Actual Magnitude')
    ax2.set_ylabel('Residuals (Predicted - Actual)')
    ax2.set_title(f'{title} - Residuals')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_title:
        plt.savefig(save_title+'.png', dpi=300)
    plt.show()

def plot_training_history(history):
    """
    Plot training history
    
    Parameters:
    -----------
    history : tensorflow.keras.callbacks.History
        Training history object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(history.history['mae'], label='Training MAE')
    ax2.plot(history.history['val_mae'], label='Validation MAE')
    ax2.set_title('Model MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean Absolute Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def analyze_feature_importance(features, magnitudes, feature_names):
    """
    Analyze feature importance using correlation
    
    Parameters:
    -----------
    features : numpy.ndarray
        Feature matrix
    magnitudes : numpy.ndarray
        Target values (magnitudes)
    feature_names : list
        List of feature names
    """
    # Calculate correlation between features and magnitude
    correlations = []
    for i, feature_name in enumerate(feature_names):
        corr = np.corrcoef(features[:, i], magnitudes)[0, 1]
        correlations.append(abs(corr))
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    indices = np.argsort(correlations)[::-1]
    plt.bar(range(len(correlations)), [correlations[i] for i in indices])
    plt.xticks(range(len(correlations)), [feature_names[i] for i in indices], rotation=45)
    plt.title('Feature Importance (Absolute Correlation with Magnitude)')
    plt.ylabel('Absolute Correlation')
    plt.tight_layout()
    plt.show()
    
    # Print sorted correlations
    print("Feature Importance Ranking:")
    for i, idx in enumerate(indices):
        print(f"{i+1:2d}. {feature_names[idx]:18s}: {correlations[idx]:.3f}")

def plot_data_distribution(magnitudes, distances):
    """
    Plot data distribution
    
    Parameters:
    -----------
    magnitudes : numpy.ndarray
        Array of magnitudes
    distances : numpy.ndarray
        Array of distances
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Magnitude distribution
    ax1.hist(magnitudes, bins=30, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Magnitude')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Earthquake Magnitudes')
    ax1.grid(True, alpha=0.3)
    
    # Distance distribution
    ax2.hist(distances, bins=30, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Distance (km)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Source Distances')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Magnitude statistics: Mean={magnitudes.mean():.2f}, Std={magnitudes.std():.2f}")
    print(f"Distance statistics: Mean={distances.mean():.1f}, Std={distances.std():.1f}")

def print_model_summary(model):
    """
    Print a detailed model summary
    
    Parameters:
    -----------
    model : tensorflow.keras.Model
        The neural network model
    """
    print("Neural Network Architecture:")
    print("=" * 50)
    model.summary()
    print("\nModel Configuration:")
    print(f"- Input features: {model.input_shape[1]}")
    print(f"- Hidden layers: {len([l for l in model.layers if 'dense' in l.name.lower()]) - 1}")
    print(f"- Total parameters: {model.count_params():,}")
    print(f"- Optimizer: {model.optimizer.__class__.__name__}")
    print(f"- Loss function: {model.loss}")
