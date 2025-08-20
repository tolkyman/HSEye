import sys
import os
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5.QtCore import Qt
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from gui.ui_hyperspectral_main import Ui_MainWindow
from core.dataset_loader import DatasetLoader, DatasetInfo
from core.spectral_analysis import SpectralIndices
from core.image_processing import HyperspectralImageProcessor
from core.worker_threads import PCAWorkerThread, AutoPCAWorkerThread, NDVIWorkerThread
from core.pca_service import PCAService


class DesignerHyperspectralGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.dataset_loader = DatasetLoader()
        self.image_processor = HyperspectralImageProcessor()
        self.dataset = None
        self.dataset_info = DatasetInfo()
        
        # Service Layer - manages business logic
        self.pca_service = PCAService()
        
        # Progress bar initial settings
        self.ui.progressBar.setVisible(False)  # Hidden at start
        self.ui.progressBar.setRange(0, 100)
        self.ui.progressBar.setValue(0)
        
        # Current worker references (managed by service)
        self.current_pca_worker = None
        self.ndvi_worker = None
        
        self.setup_canvas()
        self.setup_connections()

    def setup_canvas(self):
        canvas = self.image_processor.setup_matplotlib_canvas()
        if not self.ui.widget_2.layout():
            from PyQt5.QtWidgets import QVBoxLayout
            layout = QVBoxLayout()
            self.ui.widget_2.setLayout(layout)
        self.ui.widget_2.layout().addWidget(canvas)

    def setup_connections(self):
        self.ui.btn_dtst.clicked.connect(self.load_dataset)
        self.ui.spn_band.valueChanged.connect(self.show_current_band)
        self.ui.btn_rgbcmpst.clicked.connect(self.show_rgb_composite)
        self.ui.btn_ndvi.clicked.connect(self.calculate_ndvi)
        self.ui.btn_best.clicked.connect(self.find_best_ndvi_bands)
        # Connect PCA buttons to real functions
        self.ui.btn_runpca.clicked.connect(self.run_manual_pca)
        self.ui.btn_showpca.clicked.connect(self.show_pca_image)
        # Add auto PCA button if it exists
        if hasattr(self.ui, 'btn_autopca'):
            self.ui.btn_autopca.clicked.connect(self.run_auto_pca)
        # Disable/redirect other buttons to 'under construction' message
        for btn in [getattr(self.ui, name, None) for name in [
            'btn_train', 'btn_predict', 'save_pca', 'save_ndvi', 'save_class']]:
            if btn is not None:
                btn.clicked.connect(self.under_construction)

    def under_construction(self):
        QMessageBox.information(self, "Info", "This feature is under construction.")

    def load_dataset(self):
        # Always open dialog in the resources directory
        resources_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'resources')
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Dataset", resources_dir,
            "ENVI (*.bil);;MATLAB (*.mat);;NumPy (*.npy);;TIFF (*.tiff);;All Files (*)"
        )
        if file_path:
            try:
                self.ui.progressBar.setVisible(True)
                self.ui.progressBar.setValue(0)
                self.dataset, self.dataset_info = self.dataset_loader.load_dataset(file_path)
                if self.dataset is not None:
                    info_text = f"Dataset: {self.dataset_info.shape} - {self.dataset_info.file_format}"
                    self.ui.lbl_dtst.setText(info_text)
                    self.ui.lineEdit_rgb.clear()
                    self.ui.lineEdit_rgb.setPlaceholderText("30,60,90")
                    max_band = self.dataset_info.shape[2] - 1
                    self.ui.spn_band.setMaximum(max_band)
                    # Set reasonable range for PCA components (1 to min(50, total_bands))
                    max_pca_components = min(50, self.dataset_info.shape[2])
                    self.ui.spn_pca.setRange(1, max_pca_components)
                    self.ui.spn_pca.setValue(5)  # Default to 5 components
                    # No need to set maximum value for LineEdit fields
                    self.show_current_band()
                    self.ui.progressBar.setValue(100)
                    self.ui.progressBar.setVisible(False)
                    self.ui.lbl_status.setText("Dataset loaded successfully.")
                else:
                    QMessageBox.warning(self, "Error", "Failed to load dataset!")
                    self.ui.progressBar.setVisible(False)
            except Exception as e:
                import traceback
                print(f"Error: {str(e)}")
                traceback.print_exc()
                QMessageBox.critical(self, "Error", f"Dataset loading error: {str(e)}")
                self.ui.progressBar.setVisible(False)

    def show_progress(self, message="Operation started..."):
        """Show and start progress bar"""
        self.ui.progressBar.setVisible(True)
        self.ui.progressBar.setValue(0)
        self.ui.lbl_status.setText(message)
        QApplication.processEvents()  # Update UI
    
    def update_progress(self, value, message=""):
        """Update progress bar"""
        self.ui.progressBar.setValue(value)
        if message:
            self.ui.lbl_status.setText(message)
        QApplication.processEvents()
    
    def hide_progress(self, final_message="Operation completed"):
        """Hide progress bar"""
        self.ui.progressBar.setValue(100)
        self.ui.lbl_status.setText(final_message)
        QApplication.processEvents()
        # Hide after short delay
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(1000, lambda: self.ui.progressBar.setVisible(False))

    def disable_buttons_during_operation(self, disable=True):
        """Disable buttons during operation"""
        buttons = [self.ui.btn_runpca, self.ui.btn_ndvi, self.ui.btn_dtst, self.ui.btn_showpca]
        # Add Auto PCA button
        if hasattr(self.ui, 'btn_autopca'):
            buttons.append(self.ui.btn_autopca)
        
        for btn in buttons:
            btn.setEnabled(not disable)

    def show_current_band(self):
        if self.dataset is not None:
            try:
                band_index = self.ui.spn_band.value()
                # Double-check band index is within range
                if band_index >= self.dataset.shape[2]:
                    QMessageBox.warning(self, "Error", f"Band index {band_index} out of range")
                    return
                    
                band_data = self.dataset[:, :, band_index]
                wavelength = None
                if hasattr(self.dataset_info, 'wavelengths') and self.dataset_info.wavelengths is not None:
                    if len(self.dataset_info.wavelengths) > band_index:
                        wavelength = self.dataset_info.wavelengths[band_index]
                self.image_processor.plot_single_band(band_data, band_index, wavelength)
                self.ui.lbl_status.setText(f"Band {band_index} is displayed.")
            except Exception as e:
                import traceback
                print(f"show_current_band error: {str(e)}")
                traceback.print_exc()

    def show_rgb_composite(self):
        if self.dataset is not None:
            rgb_text = self.ui.lineEdit_rgb.text()
            if rgb_text:
                try:
                    bands = [int(x.strip()) for x in rgb_text.split(",")]
                    if len(bands) == 3:
                        # Check if band indices are valid
                        max_band_index = self.dataset.shape[2] - 1
                        for band in bands:
                            if band < 0 or band > max_band_index:
                                QMessageBox.warning(self, "Error", f"Band index {band} is out of range (0-{max_band_index})")
                                return
                        
                        red_band = self.dataset[:, :, bands[0]]
                        green_band = self.dataset[:, :, bands[1]]
                        blue_band = self.dataset[:, :, bands[2]]
                        rgb_image = np.dstack([red_band, green_band, blue_band]).astype(np.float32)
                        # Normalize each channel to 0-1
                        for i in range(3):
                            channel = rgb_image[:, :, i]
                            min_val, max_val = channel.min(), channel.max()
                            if max_val > min_val:
                                rgb_image[:, :, i] = (channel - min_val) / (max_val - min_val)
                            else:
                                rgb_image[:, :, i] = 0
                        self.image_processor.figure.clear()
                        ax = self.image_processor.figure.add_subplot(111)
                        ax.imshow(rgb_image)
                        ax.set_title(f'RGB Composite: R={bands[0]}, G={bands[1]}, B={bands[2]}')
                        ax.axis('off')
                        self.image_processor.canvas.draw()
                        self.ui.lbl_status.setText(f"RGB composite displayed: R={bands[0]}, G={bands[1]}, B={bands[2]}")
                    else:
                        QMessageBox.warning(self, "Error", "Please enter 3 band indices separated by commas.")
                except Exception as e:
                    import traceback
                    print(f"RGB Composite error: {str(e)}")
                    traceback.print_exc()
                    QMessageBox.warning(self, "Error", f"RGB format error: {str(e)}")
    def calculate_ndvi(self):
        """Calculate and display NDVI"""
        if self.dataset is not None:
            try:
                # Get wavelength values from LineEdit
                red_target = float(self.ui.lineEdit_red.text())
                nir_target = float(self.ui.lineEdit_nir.text())
                
                # Find closest bands to target wavelengths
                if self.dataset_info.wavelengths is not None:
                    # Find closest red band
                    red_idx = np.argmin(np.abs(self.dataset_info.wavelengths - red_target))
                    actual_red_wl = self.dataset_info.wavelengths[red_idx]
                    
                    # Find closest NIR band (must be above 700nm for valid NIR)
                    nir_mask = self.dataset_info.wavelengths > 700
                    if np.any(nir_mask):
                        nir_candidates = self.dataset_info.wavelengths[nir_mask]
                        nir_local_idx = np.argmin(np.abs(nir_candidates - nir_target))
                        nir_idx = np.where(nir_mask)[0][nir_local_idx]
                        actual_nir_wl = self.dataset_info.wavelengths[nir_idx]
                    else:
                        nir_idx = np.argmin(np.abs(self.dataset_info.wavelengths - nir_target))
                        actual_nir_wl = self.dataset_info.wavelengths[nir_idx]
                    
                    # Update LineEdit with actual wavelengths
                    self.ui.lineEdit_red.setText(f"{actual_red_wl:.1f}")
                    self.ui.lineEdit_nir.setText(f"{actual_nir_wl:.1f}")
                    
                    # Show which bands were actually used
                    band_info = f"Used bands: Red {actual_red_wl:.1f}nm (Band {red_idx}), NIR {actual_nir_wl:.1f}nm (Band {nir_idx})"
                    
                else:
                    # Fallback if no wavelength info - use safer defaults
                    max_band_index = self.dataset.shape[2] - 1
                    red_idx = min(30, max_band_index)  # Ensure within bounds
                    nir_idx = min(120, max_band_index)  # Ensure within bounds
                    band_info = f"Used default bands: Red Band {red_idx}, NIR Band {nir_idx}"
                
                # Validate band indices
                max_band_index = self.dataset.shape[2] - 1
                if red_idx > max_band_index or nir_idx > max_band_index:
                    QMessageBox.warning(self, "Error", f"Band indices out of range. Dataset has {max_band_index + 1} bands.")
                    return
                
                # Calculate NDVI
                ndvi = SpectralIndices.calculate_ndvi(
                    self.dataset[:, :, nir_idx], 
                    self.dataset[:, :, red_idx]
                )
                
                # Display NDVI
                self.image_processor.plot_ndvi(
                    ndvi, red_idx, nir_idx, 
                    self.dataset_info.wavelengths
                )
                
                # Show success message with band info
                QMessageBox.information(self, "NDVI Calculated", band_info)
                self.ui.lbl_status.setText(f"NDVI calculated - {band_info}")
                
            except ValueError:
                QMessageBox.warning(self, "Error", "Please enter valid wavelength values!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"NDVI calculation error: {str(e)}")
        else:
            QMessageBox.warning(self, "Warning", "Please load a dataset first!")

    def find_best_ndvi_bands(self):
        """Automatically find and set the best Red and NIR bands for NDVI"""
        if self.dataset is not None:
            try:
                # Find best bands using wavelength information
                red_idx, nir_idx = SpectralIndices.find_best_bands_for_ndvi(
                    self.dataset_info.wavelengths
                )
                
                # Get actual wavelength values and set LineEdit values
                if self.dataset_info.wavelengths is not None:
                    red_wl = self.dataset_info.wavelengths[red_idx]
                    nir_wl = self.dataset_info.wavelengths[nir_idx]
                    
                    # Set LineEdit values with actual wavelengths
                    self.ui.lineEdit_red.setText(f"{red_wl:.1f}")
                    self.ui.lineEdit_nir.setText(f"{nir_wl:.1f}")
                    
                    message = f"Best bands found!\nRed: {red_wl:.1f} nm (Band {red_idx})\nNIR: {nir_wl:.1f} nm (Band {nir_idx})"
                    self.ui.lbl_status.setText(f"Best NDVI bands: Red={red_wl:.1f}nm, NIR={nir_wl:.1f}nm")
                else:
                    # If no wavelength info, use default values
                    self.ui.lineEdit_red.setText("650")
                    self.ui.lineEdit_nir.setText("850")
                    message = f"Best bands found!\nRed: Band {red_idx}\nNIR: Band {nir_idx}\n(Using default wavelengths)"
                    self.ui.lbl_status.setText(f"Best NDVI bands: Red=Band{red_idx}, NIR=Band{nir_idx}")
                
                QMessageBox.information(self, "Best Bands Found", message)
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error finding best bands: {str(e)}")
        else:
            QMessageBox.warning(self, "Warning", "Please load a dataset first!")

    def run_manual_pca(self):
        """
        Run PCA with user-specified component count using Service Layer
        """
        if self.dataset is not None:
            try:
                # Check if any PCA is already running through service
                if self.pca_service.is_pca_running():
                    running_type = self.pca_service.get_running_worker_type()
                    QMessageBox.information(self, "Info", f"PCA calculation is already running ({running_type})...")
                    return
                
                # Get component count from user
                n_components = self.ui.spn_pca.value()
                
                # Prepare UI
                self.show_progress(f"Starting Manual PCA: {n_components} components...")
                self.disable_buttons_during_operation(True)
                
                # Start PCA through service layer
                self.current_pca_worker = self.pca_service.start_manual_pca(self.dataset, n_components)
                
                # Connect signals
                self.current_pca_worker.progress_updated.connect(self.update_progress)
                self.current_pca_worker.pca_completed.connect(self.on_pca_completed)
                self.current_pca_worker.error_occurred.connect(self.on_pca_error)
                
                # Start thread
                self.current_pca_worker.start()
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"PCA startup error: {str(e)}")
                self.disable_buttons_during_operation(False)
                self.hide_progress(f"Error: {str(e)}")
        else:
            QMessageBox.warning(self, "Warning", "Please load a dataset first!")

    def on_pca_completed(self, pca_result, variance_ratios, total_variance, component_info):
        """Called when PCA is completed"""
        try:
            # Store results
            self.pca_result = pca_result
            self.pca_variance_ratios = variance_ratios
            
            # Update UI
            self.disable_buttons_during_operation(False)
            self.hide_progress(f"PCA completed: {len(variance_ratios)} components")
            
            # Check for unusual variance patterns
            n_components = len(variance_ratios)
            warning_text = ""
            
            if n_components > 20 and total_variance > 0.98:
                warning_text = f"\n⚠️ WARNING: {n_components} components with {total_variance:.1%} variance may indicate overfitting!"
                warning_text += f"\nConsider using fewer components (typically 5-15 for hyperspectral data)."
            
            # Information message
            info_text = f"PCA completed!\n"
            info_text += f"Components: {n_components}\n"
            info_text += f"Total variance preserved: {total_variance:.1%}\n"
            
            # Add efficiency analysis
            if n_components >= 5:
                first_5_variance = np.sum(variance_ratios[:5])
                info_text += f"First 5 components explain: {first_5_variance:.1%}\n"
                
            if n_components >= 10:
                first_10_variance = np.sum(variance_ratios[:10])
                info_text += f"First 10 components explain: {first_10_variance:.1%}\n"
            
            info_text += "\nComponent breakdown:\n"
            for info in component_info[:5]:  # Show first 5
                info_text += f"PC{info['component']}: {info['percentage']:.1f}%\n"
            
            # Add warning if present
            info_text += warning_text
            
            # Choose appropriate message box type
            if warning_text:
                QMessageBox.warning(self, "PCA Results (with Warning)", info_text)
            else:
                QMessageBox.information(self, "PCA Results", info_text)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"PCA result processing error: {str(e)}")

    def on_pca_error(self, error_message):
        """Called when PCA error occurs"""
        self.disable_buttons_during_operation(False)
        self.hide_progress("PCA error!")
        QMessageBox.critical(self, "PCA Error", error_message)

    def run_auto_pca(self):
        """
        Run PCA with automatic optimal component count using Service Layer
        """
        if self.dataset is not None:
            try:
                # Check if any PCA is already running through service
                if self.pca_service.is_pca_running():
                    running_type = self.pca_service.get_running_worker_type()
                    QMessageBox.information(self, "Info", f"PCA calculation is already running ({running_type})...")
                    return
                
                # Prepare UI
                self.show_progress("Starting Auto PCA (finding optimal components)...")
                self.disable_buttons_during_operation(True)
                
                # Start Auto PCA through service layer
                self.current_pca_worker = self.pca_service.start_auto_pca(self.dataset, variance_threshold=0.95)
                
                # Connect signals
                self.current_pca_worker.progress_updated.connect(self.update_progress)
                self.current_pca_worker.pca_completed.connect(self.on_auto_pca_completed)
                self.current_pca_worker.error_occurred.connect(self.on_pca_error)
                
                # Start thread
                self.current_pca_worker.start()
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Auto PCA startup error: {str(e)}")
                self.disable_buttons_during_operation(False)
                self.hide_progress(f"Error: {str(e)}")
        else:
            QMessageBox.warning(self, "Warning", "Please load a dataset first!")

    def on_auto_pca_completed(self, pca_result, variance_ratios, total_variance, component_info, optimal_components):
        """Called when Auto PCA is completed"""
        try:
            # Store results
            self.pca_result = pca_result
            self.pca_variance_ratios = variance_ratios
            
            # Update SpinBox
            self.ui.spn_pca.setValue(optimal_components)
            
            # Update UI
            self.disable_buttons_during_operation(False)
            self.hide_progress(f"Auto PCA completed: {optimal_components} components")
            
            # Information message
            info_text = f"Auto PCA completed!\n"
            info_text += f"Optimal components: {optimal_components}\n"
            info_text += f"Achieved variance: {total_variance:.1%}\n"
            info_text += f"Target was: 95%\n\n"
            info_text += "Component breakdown:\n"
            for info in component_info[:5]:  # Show first 5
                info_text += f"PC{info['component']}: {info['percentage']:.1f}%\n"
            
            QMessageBox.information(self, "Auto PCA Results", info_text)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Auto PCA result processing error: {str(e)}")

    def show_pca_image(self):
        """
        Display PCA result as RGB composite
        """
        if hasattr(self, 'pca_result') and self.pca_result is not None:
            try:
                self.image_processor.plot_pca_results(
                    self.pca_result, 
                    self.pca_variance_ratios,
                    np.sum(self.pca_variance_ratios)
                )
                self.ui.lbl_status.setText("PCA RGB composite displayed")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"PCA display error: {str(e)}")
        else:
            QMessageBox.warning(self, "Warning", "Please run PCA first!")

    def show_component_selector(self):
        """
        Let user select which component to view
        """
        if hasattr(self, 'pca_result'):
            # Open new window for component selection
            pass  # Can be implemented later

    def closeEvent(self, event):
        """
        Handle window close event - cleanup resources
        """
        try:
            # Clean up PCA service and stop any running workers
            self.pca_service.cleanup()
        except Exception as e:
            print(f"Error during cleanup: {e}")
        finally:
            event.accept()

def main():
    app = QApplication(sys.argv)
    window = DesignerHyperspectralGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()