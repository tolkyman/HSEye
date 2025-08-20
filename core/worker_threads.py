"""
PCA Worker Thread - PCA calculation with progress bar
Performs PCA operations in background and updates progress
"""

import time
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from core.spectral_analysis import PCAAnalysis

class PCAWorkerThread(QThread):
    """
    Thread that performs PCA calculation in background
    Sends progress signals
    """
    
    # Signal definitions
    progress_updated = pyqtSignal(int, str)  # (progress_value, status_message)
    pca_completed = pyqtSignal(object, object, float, object)  # (pca_result, variance_ratios, total_variance, component_info)
    error_occurred = pyqtSignal(str)  # error_message
    
    def __init__(self, dataset, n_components, is_auto=False):
        super().__init__()
        self.dataset = dataset
        self.n_components = n_components
        self.is_auto = is_auto
        
    def run(self):
        """Main thread execution function"""
        try:
            # 1. Initialize
            self.progress_updated.emit(5, "Starting PCA calculation...")
            time.sleep(0.1)  # Brief pause for UI update
            
            if self.is_auto:
                # Auto PCA - find optimal component count
                self.progress_updated.emit(15, "Finding optimal component count...")
                optimal_components, achieved_variance = PCAAnalysis.find_optimal_components(
                    self.dataset, variance_threshold=0.95
                )
                self.n_components = optimal_components
                self.progress_updated.emit(30, f"Optimal: {optimal_components} components")
                time.sleep(0.2)
            
            # 2. Data preparation
            self.progress_updated.emit(40, "Preparing data...")
            if self.isInterruptionRequested():
                return
                
            height, width, bands = self.dataset.shape
            data_2d = self.dataset.reshape(-1, bands)
            self.progress_updated.emit(50, f"Data reshaped: {data_2d.shape}")
            time.sleep(0.1)
            
            # 3. PCA calculation
            self.progress_updated.emit(60, "Computing PCA...")
            if self.isInterruptionRequested():
                return
                
            from sklearn.decomposition import PCA
            pca = PCA(n_components=self.n_components)
            
            # PCA fit operation (longest step)
            self.progress_updated.emit(70, "Fitting PCA model...")
            if self.isInterruptionRequested():
                return
                
            pca_2d = pca.fit_transform(data_2d)
            
            # 4. Data transformation
            self.progress_updated.emit(85, "Organizing results...")
            pca_result = pca_2d.reshape(height, width, self.n_components)
            
            # 5. Variance information
            variance_ratios = pca.explained_variance_ratio_
            total_variance = np.sum(variance_ratios)
            
            # 6. Detailed information
            self.progress_updated.emit(95, "Preparing component information...")
            component_info = PCAAnalysis.get_component_info(self.dataset, self.n_components)
            
            # 7. Completed
            self.progress_updated.emit(100, f"PCA completed: {self.n_components} components")
            
            # Send results
            self.pca_completed.emit(pca_result, variance_ratios, total_variance, component_info)
            
        except Exception as e:
            self.error_occurred.emit(f"PCA calculation error: {str(e)}")

class AutoPCAWorkerThread(QThread):
    """
    Auto PCA specialized thread
    """
    
    progress_updated = pyqtSignal(int, str)
    pca_completed = pyqtSignal(object, object, float, object, int)  # +optimal_components
    error_occurred = pyqtSignal(str)
    
    def __init__(self, dataset, variance_threshold=0.95):
        super().__init__()
        self.dataset = dataset
        self.variance_threshold = variance_threshold
        
    def run(self):
        try:
            # 1. Find optimal components
            self.progress_updated.emit(10, "Analyzing optimal component count...")
            optimal_components, achieved_variance = PCAAnalysis.find_optimal_components(
                self.dataset, self.variance_threshold
            )
            
            self.progress_updated.emit(25, f"Optimal: {optimal_components} components (%.1f%% variance)" % (achieved_variance * 100))
            time.sleep(0.2)
            
            # 2. Calculate PCA
            self.progress_updated.emit(40, "Computing PCA...")
            height, width, bands = self.dataset.shape
            data_2d = self.dataset.reshape(-1, bands)
            
            self.progress_updated.emit(55, "Fitting PCA model...")
            from sklearn.decomposition import PCA
            pca = PCA(n_components=optimal_components)
            pca_2d = pca.fit_transform(data_2d)
            
            self.progress_updated.emit(80, "Preparing results...")
            pca_result = pca_2d.reshape(height, width, optimal_components)
            
            # Information
            variance_ratios = pca.explained_variance_ratio_
            total_variance = np.sum(variance_ratios)
            component_info = PCAAnalysis.get_component_info(self.dataset, optimal_components)
            
            self.progress_updated.emit(100, f"Auto PCA completed: {optimal_components} components")
            
            # Send results
            self.pca_completed.emit(pca_result, variance_ratios, total_variance, component_info, optimal_components)
            
        except Exception as e:
            self.error_occurred.emit(f"Auto PCA error: {str(e)}")

class NDVIWorkerThread(QThread):
    """
    NDVI calculation thread (bonus)
    """
    
    progress_updated = pyqtSignal(int, str)
    ndvi_completed = pyqtSignal(object, int, int)  # (ndvi_result, red_band, nir_band)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, dataset, red_band, nir_band):
        super().__init__()
        self.dataset = dataset
        self.red_band = red_band
        self.nir_band = nir_band
        
    def run(self):
        try:
            self.progress_updated.emit(20, "Calculating NDVI...")
            
            from core.spectral_analysis import SpectralIndices
            
            # Extract bands
            red_data = self.dataset[:, :, self.red_band]
            nir_data = self.dataset[:, :, self.nir_band]
            
            self.progress_updated.emit(60, "Applying NDVI formula...")
            
            # Calculate NDVI
            ndvi_result = SpectralIndices.calculate_ndvi(nir_data, red_data)
            
            self.progress_updated.emit(100, "NDVI completed")
            
            self.ndvi_completed.emit(ndvi_result, self.red_band, self.nir_band)
            
        except Exception as e:
            self.error_occurred.emit(f"NDVI calculation error: {str(e)}")
