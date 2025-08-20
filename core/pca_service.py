"""
PCA Service Layer
Manages PCA operations and worker threads
Separates business logic from GUI
"""

from typing import Optional
from .worker_threads import PCAWorkerThread, AutoPCAWorkerThread


class PCAService:
    """
    Service layer for PCA operations
    Manages worker threads and provides clean interface to GUI
    """
    
    def __init__(self):
        self.current_worker: Optional[PCAWorkerThread] = None
        self.auto_worker: Optional[AutoPCAWorkerThread] = None
        
    def start_manual_pca(self, dataset, n_components: int) -> PCAWorkerThread:
        """
        Start manual PCA with specified component count
        
        Args:
            dataset: Hyperspectral dataset
            n_components: Number of PCA components to calculate
            
        Returns:
            PCAWorkerThread: Worker thread for PCA calculation
        """
        # Stop any existing worker
        self.stop_current_workers()
        
        # Create new worker
        self.current_worker = PCAWorkerThread(dataset, n_components, is_auto=False)
        return self.current_worker
    
    def start_auto_pca(self, dataset, variance_threshold: float = 0.95) -> AutoPCAWorkerThread:
        """
        Start automatic PCA with optimal component count
        
        Args:
            dataset: Hyperspectral dataset
            variance_threshold: Target variance to preserve (default: 95%)
            
        Returns:
            AutoPCAWorkerThread: Worker thread for auto PCA calculation
        """
        # Stop any existing worker
        self.stop_current_workers()
        
        # Create new auto worker
        self.auto_worker = AutoPCAWorkerThread(dataset, variance_threshold)
        return self.auto_worker
    
    def is_pca_running(self) -> bool:
        """
        Check if any PCA operation is currently running
        
        Returns:
            bool: True if PCA is running, False otherwise
        """
        manual_running = self.current_worker and self.current_worker.isRunning()
        auto_running = self.auto_worker and self.auto_worker.isRunning()
        return manual_running or auto_running
    
    def stop_current_workers(self):
        """
        Stop all running PCA workers
        """
        if self.current_worker and self.current_worker.isRunning():
            self.current_worker.terminate()
            self.current_worker.wait()
            
        if self.auto_worker and self.auto_worker.isRunning():
            self.auto_worker.terminate()
            self.auto_worker.wait()
    
    def get_running_worker_type(self) -> str:
        """
        Get the type of currently running worker
        
        Returns:
            str: 'manual', 'auto', or 'none'
        """
        if self.current_worker and self.current_worker.isRunning():
            return 'manual'
        elif self.auto_worker and self.auto_worker.isRunning():
            return 'auto'
        else:
            return 'none'
    
    def cleanup(self):
        """
        Clean up resources when service is destroyed
        """
        self.stop_current_workers()
        self.current_worker = None
        self.auto_worker = None
