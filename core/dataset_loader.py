
import os
import numpy as np
import scipy.io
import cv2
from typing import Optional, Tuple

try:
    import spectral.io.envi as envi
    SPECTRAL_AVAILABLE = True
except ImportError:
    SPECTRAL_AVAILABLE = False
    print("Warning: 'spectral' library not found. ENVI support disabled.")

class DatasetInfo:
    def __init__(self):
        self.shape: Optional[Tuple[int, int, int]] = None
        self.wavelengths: Optional[np.ndarray] = None
        self.wavelength_units: str = "nm"
        self.data_type: Optional[str] = None
        self.file_path: str = ""
        self.file_format: str = ""
        self.memory_size_mb: float = 0.0
        self.spectral_range: Optional[Tuple[float, float]] = None

    def __str__(self):
        if self.shape is None:
            return "Dataset Info: No data loaded"
        info = f"Dataset Info:\n"
        info += f"  Shape: {self.shape} (H×W×Bands)\n"
        info += f"  Format: {self.file_format}\n"
        info += f"  Size: {self.memory_size_mb:.1f} MB\n"
        info += f"  Data Type: {self.data_type}\n"
        if self.wavelengths is not None:
            info += f"  Spectral Range: {self.spectral_range[0]:.1f}-{self.spectral_range[1]:.1f} {self.wavelength_units}\n"
            info += f"  Bands: {len(self.wavelengths)}"
        return info

class DatasetLoader:
    def load_dataset(self, file_path: str) -> Tuple[Optional[np.ndarray], DatasetInfo]:
        info = DatasetInfo()
        info.file_path = file_path
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext == '.bil':
                return self._load_envi(file_path, info)
            elif file_ext == '.mat':
                return self._load_matlab(file_path, info)
            elif file_ext == '.npy':
                return self._load_numpy(file_path, info)
            elif file_ext in ['.tiff', '.tif']:
                return self._load_tiff(file_path, info)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
        except Exception as e:
            print(f"Dataset loading error: {str(e)}")
            return None, info

    def _load_envi(self, file_path: str, info: DatasetInfo) -> Tuple[Optional[np.ndarray], DatasetInfo]:
        if not SPECTRAL_AVAILABLE:
            raise ImportError("'spectral' library required for ENVI support")
        info.file_format = "ENVI"
        hdr_path = file_path + '.hdr'
        if not os.path.exists(hdr_path):
            base_path = os.path.splitext(file_path)[0]
            hdr_path = base_path + '.hdr'
            if not os.path.exists(hdr_path):
                raise FileNotFoundError(f"Header file not found: {hdr_path}")
        img = envi.open(hdr_path, file_path)
        metadata = img.metadata
        lines = int(metadata['lines'])
        samples = int(metadata['samples'])
        bands = int(metadata['bands'])
        
        # Validate dimensions
        if lines <= 0 or samples <= 0 or bands <= 0:
            raise ValueError(f"Invalid dataset dimensions: {lines}x{samples}x{bands}")
            
        info.shape = (lines, samples, bands)
        if 'wavelength' in metadata:
            info.wavelengths = np.array([float(w) for w in metadata['wavelength']], dtype=np.float32)
            info.spectral_range = (info.wavelengths.min(), info.wavelengths.max())
        if 'wavelength units' in metadata:
            info.wavelength_units = metadata['wavelength units']
        dtype_map = {
            '1': np.uint8, '2': np.int16, '3': np.int32, '4': np.float32,
            '5': np.float64, '12': np.uint16, '13': np.uint32, '14': np.int64, '15': np.uint64
        }
        data_type = metadata.get('data type', '4')
        np_dtype = dtype_map.get(data_type, np.float32)
        info.data_type = str(np_dtype)
        info.memory_size_mb = (info.shape[0] * info.shape[1] * info.shape[2] * np.dtype(np_dtype).itemsize) / (1024**2)
        if info.memory_size_mb > 500:
            print(f"Large file detected ({info.memory_size_mb:.1f} MB). Using memory map...")
            data = img.open_memmap()
        else:
            data = img.load()
        return data, info

    def _load_matlab(self, file_path: str, info: DatasetInfo) -> Tuple[Optional[np.ndarray], DatasetInfo]:
        info.file_format = "MATLAB"
        mat_data = scipy.io.loadmat(file_path)
        data = None
        for key, value in mat_data.items():
            if not key.startswith('__') and isinstance(value, np.ndarray) and len(value.shape) == 3:
                data = value
                break
        if data is None:
            raise ValueError("No 3D array found in MATLAB file")
        info.shape = data.shape
        info.data_type = str(data.dtype)
        info.memory_size_mb = data.nbytes / (1024**2)
        return data, info

    def _load_numpy(self, file_path: str, info: DatasetInfo) -> Tuple[Optional[np.ndarray], DatasetInfo]:
        info.file_format = "NumPy"
        data = np.load(file_path)
        if len(data.shape) != 3:
            raise ValueError(f"Expected 3D array, got {len(data.shape)}D array")
        info.shape = data.shape
        info.data_type = str(data.dtype)
        info.memory_size_mb = data.nbytes / (1024**2)
        return data, info

    def _load_tiff(self, file_path: str, info: DatasetInfo) -> Tuple[Optional[np.ndarray], DatasetInfo]:
        info.file_format = "TIFF"
        data = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if data is None:
            raise ValueError("TIFF file could not be read")
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=2)
        info.shape = data.shape
        info.data_type = str(data.dtype)
        info.memory_size_mb = data.nbytes / (1024**2)
        return data, info