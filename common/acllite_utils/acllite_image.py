"""
Image processing utilities for ACL inference
"""
import numpy as np
import acl
import constants as const
from acllite_logger import log_error, log_info


class AclLiteImage:
    """Wrapper for image data in ACL format"""

    def __init__(self, image_data=None, width=0, height=0, size=0):
        """
        Initialize AclLiteImage

        Args:
            image_data: raw image data bytes
            width: image width
            height: image height
            size: data size
        """
        self._data = None
        self._size = 0
        self._width = width
        self._height = height
        self._aligned_width = 0
        self._aligned_height = 0

        if image_data is not None:
            self._data = image_data
            self._size = size if size > 0 else len(image_data)

    def data(self):
        """Get raw image data"""
        return self._data

    def size(self):
        """Get data size"""
        return self._size

    def width(self):
        """Get image width"""
        return self._width

    def height(self):
        """Get image height"""
        return self._height

    @staticmethod
    def from_numpy(np_image):
        """Create AclLiteImage from numpy array"""
        if not isinstance(np_image, np.ndarray):
            raise ValueError("Input must be numpy array")

        h, w = np_image.shape[:2]
        size = np_image.nbytes

        # Convert numpy to pointer
        if hasattr(acl.util, 'bytes_to_ptr'):
            data_ptr = acl.util.bytes_to_ptr(np_image.tobytes())
        else:
            data_ptr = acl.util.numpy_to_ptr(np_image)

        image = AclLiteImage()
        image._data = data_ptr
        image._size = size
        image._width = w
        image._height = h
        return image
