"""
Image Processing and Visualization Module (Band Viewer & RGB Composite Only)
All other features are under construction.
"""

import numpy as np
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from typing import Optional, Tuple

class HyperspectralImageProcessor:

    def __init__(self, figure_size: Tuple[int, int] = (10, 8)):
        self.figure_size = figure_size
        self.figure = None
        self.canvas = None
        self.current_image = None
        self.current_title = ""

        # Colormap options
        self.colormaps = {
            'grayscale': 'gray',
            'viridis': 'viridis',
            'plasma': 'plasma',
            'inferno': 'inferno',
            'jet': 'jet',
            'hot': 'hot',
            'coolwarm': 'coolwarm',
            'ndvi': self._create_ndvi_colormap()
        }

    def _create_ndvi_colormap(self):
        """Create custom colormap for NDVI (-1: red, 0: yellow, 1: green)"""
        colors = ['red', 'yellow', 'lightgreen', 'darkgreen']
        n_bins = 256
        cmap = mcolors.LinearSegmentedColormap.from_list('ndvi', colors, N=n_bins)
        return cmap

    def setup_matplotlib_canvas(self) -> FigureCanvas:
        self.figure = Figure(figsize=self.figure_size)
        self.canvas = FigureCanvas(self.figure)
        return self.canvas

    def plot_single_band(self, band_data: np.ndarray, band_index: int,
                        wavelength: Optional[float] = None, 
                        colormap: str = 'viridis',
                        title_prefix: str = "Band") -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        if wavelength is not None:
            title = f"{title_prefix} {band_index} ({wavelength:.1f} nm)"
        else:
            title = f"{title_prefix} {band_index}"
        self.current_title = title
        im = ax.imshow(band_data, cmap=colormap, aspect='auto')
        self.figure.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Pixel X')
        ax.set_ylabel('Pixel Y')
        ax.grid(False)
        self.figure.tight_layout()
        if self.canvas:
            self.canvas.draw()
        self.current_image = band_data

    def plot_ndvi(self, ndvi_data: np.ndarray, 
              red_band: int, nir_band: int,
              wavelengths: Optional[np.ndarray] = None,
              show_statistics: bool = True) -> None:
        """Display NDVI map with statistics"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Create title
        if wavelengths is not None:
            red_wl = wavelengths[red_band]
            nir_wl = wavelengths[nir_band]
            title = f"NDVI Map\nRed: {red_band}({red_wl:.0f}nm), NIR: {nir_band}({nir_wl:.0f}nm)"
        else:
            title = f"NDVI Map (Red: {red_band}, NIR: {nir_band})"
        
        self.current_title = title
        
        # Display NDVI image
        im = ax.imshow(ndvi_data, cmap=self.colormaps['ndvi'], 
                    vmin=-1, vmax=1, aspect='auto')
        
        # Colorbar
        cbar = self.figure.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('NDVI Value', rotation=270, labelpad=20)
        
        # Title and labels
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Pixel X')
        ax.set_ylabel('Pixel Y')
        ax.grid(False)
        
        # Statistics
        if show_statistics:
            mean_ndvi = np.nanmean(ndvi_data)
            std_ndvi = np.nanstd(ndvi_data)
            min_ndvi = np.nanmin(ndvi_data)
            max_ndvi = np.nanmax(ndvi_data)
            
            stats_text = f"Mean: {mean_ndvi:.3f}\nStd: {std_ndvi:.3f}\nMin: {min_ndvi:.3f}\nMax: {max_ndvi:.3f}"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=9)
        
        self.figure.tight_layout()
        
        if self.canvas:
            self.canvas.draw()
        
        self.current_image = ndvi_data

    def plot_pca_results(self, pca_result, variance_ratios, total_variance):
        """
        Display PCA results as RGB composite + variance information
        Handles cases with 1, 2, or 3+ components intelligently
        """
        from .spectral_analysis import PCAAnalysis
        
        self.figure.clear()
        n_components = pca_result.shape[2]
        
        # Normalize PCA data for display
        normalized = PCAAnalysis.normalize_pca_for_display(pca_result)
        
        if n_components >= 3:
            # Standard RGB display with 3+ components
            rgb_image = normalized[:, :, :3]  # First 3 components
            
            # Main image
            ax1 = self.figure.add_subplot(121)
            ax1.imshow(rgb_image)
            ax1.set_title(f'PCA RGB Composite (PC1-PC2-PC3)\nTotal Variance: {total_variance:.1%}')
            ax1.axis('off')
            
        elif n_components == 2:
            # 2 components: Create pseudo-RGB using PC1, PC2, and zeros for blue
            pc1 = normalized[:, :, 0]  # Red channel
            pc2 = normalized[:, :, 1]  # Green channel
            zeros = np.zeros_like(pc1)  # Blue channel (empty)
            
            rgb_image = np.dstack([pc1, pc2, zeros])
            
            # Main image
            ax1 = self.figure.add_subplot(121)
            ax1.imshow(rgb_image)
            ax1.set_title(f'PCA Composite (PC1-PC2-Zero)\nTotal Variance: {total_variance:.1%}\nNote: Only 2 components available')
            ax1.axis('off')
            
        elif n_components == 1:
            # 1 component: Display as grayscale
            ax1 = self.figure.add_subplot(121)
            ax1.imshow(normalized[:, :, 0], cmap='gray')
            ax1.set_title(f'PCA Component 1 (Grayscale)\nTotal Variance: {total_variance:.1%}\nNote: Only 1 component available')
            ax1.axis('off')
        
        else:
            # No components (shouldn't happen but handle gracefully)
            ax1 = self.figure.add_subplot(121)
            ax1.text(0.5, 0.5, 'No PCA components available', 
                    ha='center', va='center', transform=ax1.transAxes, fontsize=14)
            ax1.set_title('PCA Results - No Data')
            ax1.axis('off')
        
        # Variance chart (always show regardless of component count)
        ax2 = self.figure.add_subplot(122)
        
        # Limit display to first 10 components if more than 10 exist
        max_display_components = 10
        total_components = len(variance_ratios)
        
        if total_components > max_display_components:
            # Show only first 10 components
            display_ratios = variance_ratios[:max_display_components]
            display_components = range(1, max_display_components + 1)
            chart_title = f'Component Importance\nShowing first {max_display_components} of {total_components} components'
            
            # Calculate total variance of displayed components
            displayed_variance = np.sum(display_ratios)
            chart_subtitle = f'Displayed components explain {displayed_variance:.1%} of total variance'
        else:
            # Show all components
            display_ratios = variance_ratios
            display_components = range(1, total_components + 1)
            chart_title = f'Component Importance\n({total_components} components)'
            chart_subtitle = f'Total variance: {total_variance:.1%}'
        
        # Create bar chart
        bars = ax2.bar(display_components, display_ratios * 100)
        ax2.set_xlabel('Component')
        ax2.set_ylabel('Variance Explained (%)')
        ax2.set_title(chart_title)
        
        # Add subtitle with variance info
        ax2.text(0.5, -0.15, chart_subtitle, ha='center', va='top', 
                transform=ax2.transAxes, fontsize=9, style='italic')
        
        # Add percentage labels on bars (only if not too crowded)
        if len(display_ratios) <= 8:
            # Show percentage on each bar if 8 or fewer
            for i, (bar, ratio) in enumerate(zip(bars, display_ratios)):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{ratio:.1%}', ha='center', va='bottom', fontsize=9)
        else:
            # For 9-10 components, show only significant ones (>5%)
            for i, (bar, ratio) in enumerate(zip(bars, display_ratios)):
                if ratio > 0.05:  # Only show if >5%
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                            f'{ratio:.1%}', ha='center', va='bottom', fontsize=8)
        
        # Add cumulative variance line if more than 1 component
        if len(display_ratios) > 1:
            cumulative = np.cumsum(display_ratios * 100)
            ax2_twin = ax2.twinx()
            ax2_twin.plot(display_components, cumulative, 'ro-', color='red', linewidth=2, markersize=4)
            ax2_twin.set_ylabel('Cumulative Variance (%)', color='red')
            ax2_twin.tick_params(axis='y', labelcolor='red')
            
            # Add cumulative labels (only for significant points)
            step = max(1, len(cumulative) // 5)  # Show max 5 labels
            for i in range(0, len(cumulative), step):
                ax2_twin.text(display_components[i], cumulative[i] + 1, 
                            f'{cumulative[i]:.1f}%', 
                            ha='center', va='bottom', fontsize=8, color='red')
            
            # Always show the last cumulative point
            if len(cumulative) > 1:
                last_idx = len(cumulative) - 1
                ax2_twin.text(display_components[last_idx], cumulative[last_idx] + 1, 
                            f'{cumulative[last_idx]:.1f}%', 
                            ha='center', va='bottom', fontsize=8, color='red', weight='bold')
        
        self.figure.tight_layout()
        self.canvas.draw()

    def show_single_component(self, pca_result, component_idx, variance_ratio):
        """
        Display single PCA component in grayscale
        """
        from .spectral_analysis import PCAAnalysis
        
        self.figure.clear()
        
        component = pca_result[:, :, component_idx]
        normalized = PCAAnalysis.normalize_pca_for_display(pca_result)
        
        ax = self.figure.add_subplot(111)
        im = ax.imshow(normalized[:, :, component_idx], cmap='gray')
        ax.set_title(f'PCA Component {component_idx + 1}\nVariance: {variance_ratio:.1%}')
        ax.axis('off')
        
        # Add colorbar
        self.figure.colorbar(im, ax=ax)
        self.canvas.draw()

    def show_pca_comparison(self, original_rgb, pca_rgb):
        """
        Original vs PCA comparison
        """
        self.figure.clear()
        
        ax1 = self.figure.add_subplot(121)
        ax1.imshow(original_rgb)
        ax1.set_title('Original RGB')
        ax1.axis('off')
        
        ax2 = self.figure.add_subplot(122)
        ax2.imshow(pca_rgb)
        ax2.set_title('PCA RGB')
        ax2.axis('off')
        
        self.canvas.draw()

    # All other features are under construction and not implemented.