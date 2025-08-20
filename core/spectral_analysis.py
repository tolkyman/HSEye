"""
Spectral Analysis Module
NDVI, PCA, Classification and other hyperspectral analyses
"""

import numpy as np
from typing import Tuple, Optional

class SpectralIndices:

    @staticmethod
    def calculate_ndvi(nir_band: np.ndarray, red_band: np.ndarray, epsilon: float = 1e-7) -> np.ndarray:
        """
        Calculate NDVI (Normalized Difference Vegetation Index)
        NDVI = (NIR - Red) / (NIR + Red)
        
        Args:
            nir_band: Near Infrared band data
            red_band: Red band data
            epsilon: Small value to prevent division by zero
            
        Returns:
            NDVI matrix with values between -1 and 1
        """
        nir = nir_band.astype(np.float32)  # Convert to numpy float32
        red = red_band.astype(np.float32)  # Convert to numpy float32

        denominator = nir + red
        denominator[denominator == 0] = epsilon

        ndvi = (nir - red) / denominator

        ndvi = np.clip(ndvi, -1, 1)  # Clip values to [-1, 1]

        return ndvi
    
    @staticmethod
    def find_best_bands_for_ndvi(wavelengths: Optional[np.ndarray]) -> Tuple[int, int]:
        """
        Find the best Red and NIR bands for NDVI calculation
        
        Args:
            wavelengths: Array of wavelength values in nanometers
            
        Returns:
            Tuple of (red_band_index, nir_band_index)
        """
        
        if wavelengths is None:
            return 30, 120
        
        red_target = 650
        red_idx = np.argmin(np.abs(wavelengths - red_target))  

        nir_target = 850
        nir_mask = wavelengths > 700

        if np.any(nir_mask):
            nir_candidates = wavelengths[nir_mask]
            nir_local_idx = np.argmin(np.abs(nir_candidates - nir_target))
            nir_idx = np.where(nir_mask)[0][nir_local_idx]
        else:
            nir_idx = len(wavelengths) - 1  # Last index if no NIR band found

        return red_idx, nir_idx


class PCAAnalysis:
    """Principal Component Analysis operations"""
    
    @staticmethod
    def run_pca(data, n_components):
        """
        Manual PCA - user specified component count
        
        Input: (559, 320, 168) hyperspectral image
        Output: (559, 320, 5) PCA image + variance information
        """
        # Input validation
        if len(data.shape) != 3:
            raise ValueError(f"Expected 3D data, got {len(data.shape)}D")
        
        height, width, bands = data.shape
        
        if height <= 0 or width <= 0 or bands <= 0:
            raise ValueError(f"Invalid data dimensions: {height}x{width}x{bands}")
            
        if n_components <= 0 or n_components > bands:
            raise ValueError(f"Invalid component count: {n_components} (data has {bands} bands)")
        
        # 3D → 2D transformation (required for PCA)
        data_2d = data.reshape(-1, bands)  # (178880, 168)
        
        # Calculate PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)
        pca_2d = pca.fit_transform(data_2d)  # (178880, 5)
        
        # 2D → 3D back transformation
        pca_result = pca_2d.reshape(height, width, n_components)
        
        # Variance information
        variance_ratios = pca.explained_variance_ratio_
        total_variance = np.sum(variance_ratios)
        
        # Diagnostic information for unusual cases
        if n_components > 20 and total_variance > 0.98:
            print(f"⚠️  WARNING: {n_components} components explain {total_variance:.1%} variance")
            print(f"   This might indicate:")
            print(f"   • Overfitting to noise")
            print(f"   • High spectral redundancy")
            print(f"   • Consider using fewer components")
            
            # Show variance drop-off analysis
            significant_components = np.sum(variance_ratios > 0.01)  # >1% variance
            print(f"   • Components with >1% variance: {significant_components}")
            
            if len(variance_ratios) >= 10:
                avg_last_10 = np.mean(variance_ratios[-10:])
                print(f"   • Average variance of last 10 components: {avg_last_10:.3%}")
        
        return pca_result, variance_ratios, total_variance
    
    @staticmethod
    def find_optimal_components(data, variance_threshold=0.95):
        """
        Automatic component finding - how many components for 95% variance?
        
        Example: 168 bands typically need 8-12 components for 95% variance
        """
        # Input validation
        if len(data.shape) != 3:
            raise ValueError(f"Expected 3D data, got {len(data.shape)}D")
        
        height, width, bands = data.shape
        
        if height <= 0 or width <= 0 or bands <= 0:
            raise ValueError(f"Invalid data dimensions: {height}x{width}x{bands}")
            
        if variance_threshold <= 0 or variance_threshold > 1:
            raise ValueError(f"Invalid variance threshold: {variance_threshold} (must be 0-1)")
        
        data_2d = data.reshape(-1, bands)
        
        # Maximum components to test (don't test too many, it's slow)
        max_components = min(bands // 2, 50)
        
        from sklearn.decomposition import PCA
        pca = PCA(n_components=max_components)
        pca.fit(data_2d)
        
        # Cumulative variance: [0.4, 0.65, 0.80, 0.90, 0.95, 0.97...]
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        
        # Find first component count that reaches 95%
        optimal_idx = np.argmax(cumulative_variance >= variance_threshold)
        optimal_components = optimal_idx + 1
        
        return optimal_components, cumulative_variance[optimal_idx]
    
    @staticmethod
    def get_component_info(data, n_components):
        """
        PCA detailed information - show how important each component is
        """
        height, width, bands = data.shape
        data_2d = data.reshape(-1, bands)
        
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)
        pca.fit(data_2d)
        
        # Detailed information for each component
        component_info = []
        cumulative = 0
        for i, ratio in enumerate(pca.explained_variance_ratio_):
            cumulative += ratio
            component_info.append({
                'component': i + 1,
                'variance': ratio,
                'cumulative': cumulative,
                'percentage': ratio * 100
            })
        
        return component_info
    
    @staticmethod
    def normalize_pca_for_display(pca_result):
        """
        Normalize PCA result for image display to 0-1 range
        """
        normalized = np.zeros_like(pca_result)
        
        for i in range(pca_result.shape[2]):
            component = pca_result[:, :, i]
            
            # Check for NaN or Inf values
            if np.any(np.isnan(component)) or np.any(np.isinf(component)):
                print(f"Warning: Component {i} contains NaN or Inf values, using zeros")
                normalized[:, :, i] = 0.0
                continue
            
            # Min-max normalization
            comp_min = np.min(component)
            comp_max = np.max(component)
            
            if comp_max > comp_min and np.isfinite(comp_min) and np.isfinite(comp_max):
                normalized[:, :, i] = (component - comp_min) / (comp_max - comp_min)
            else:
                normalized[:, :, i] = 0.5  # Gray if no variation
        
        return normalized