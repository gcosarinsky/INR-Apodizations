import math
import re
import warnings
from abc import ABC, abstractmethod

import numpy as np


class KernelParametersBase(ABC):
    """
    Base class for kernel parameter management.
    - user_param_names: all params user must provide (can be float, int, tuple)
    - kernel_float_names: floats passed to kernel (user + dependents)
    - kernel_int_names: ints passed to kernel (user + dependents)
    - dependent_float_names: float params calculated internally
    - dependent_int_names: int params calculated internally
    - default_param_values: dict with default values for user parameters
    """

    user_param_names = None
    kernel_float_names = None
    kernel_int_names = None
    dependent_float_names = None
    dependent_int_names = None
    default_param_values = {}

    def __init__(self, params_dict):
        self.param_dict = params_dict  # Store original parameters dictionary provided by user

        # Ensure required class variables are defined
        required_vars = [
            "user_param_names",
            "kernel_float_names",
            "kernel_int_names",
            "dependent_float_names",
            "dependent_int_names",
        ]
        for var in required_vars:
            if getattr(self, var) is None:
                raise NotImplementedError(f"Subclass must define {var}.")

        # Check for missing user parameters
        missing = [name for name in self.user_param_names if name not in params_dict]
        missing_with_defaults = []

        # Fill missing parameters with default values if available
        for name in missing:
            if name in self.default_param_values:
                params_dict[name] = self.default_param_values[name]
                missing_with_defaults.append((name, self.default_param_values[name]))
            else:
                raise ValueError(f"Missing user parameter '{name}' and no default value is defined.")

        # Print a single warning with all missing parameters and their default values
        if missing_with_defaults:
            missing_text = ", ".join([f"'{name}': {value}" for name, value in missing_with_defaults])
            warnings.warn(
                (
                    "The following parameters were not provided and default values were used:\n"
                    f" {missing_text}"
                ),
                stacklevel=2,
            )

        # Assign user parameters as attributes
        for name in self.user_param_names:
            setattr(self, name, params_dict[name])

        # Ensure blocksize_img is a tuple
        self.blocksize_img = tuple(self.blocksize_img)

        # Calculate and set dependent parameters
        self.calculate_dependent_parameters()

    @classmethod
    def check_enum_consistency(cls, enum_code):
        enum_int_names = get_enum_names(enum_code, "IntParams")
        enum_float_names = get_enum_names(enum_code, "FloatParams")
        # convert to lowercase
        enum_int_names = [name.lower() for name in enum_int_names]
        enum_float_names = [name.lower() for name in enum_float_names]
        # check consistency with names in the class, order is important
        assert enum_int_names == cls.kernel_int_names, (
            f"Enum IntParams names {enum_int_names} do not match "
            f"kernel_int_names {cls.kernel_int_names}"
        )
        assert enum_float_names == cls.kernel_float_names, (
            f"Enum FloatParams names {enum_float_names} do not match "
            f"kernel_float_names {cls.kernel_float_names}"
        )

    @abstractmethod
    def calculate_dependent_parameters(self):
        """
        Subclasses implement calculation of dependent parameters.
        """

    def get_float_array(self):
        """Return ordered array of float parameters for kernel."""
        return np.array([float(getattr(self, name)) for name in self.kernel_float_names], dtype=np.float32)

    def get_int_array(self):
        """Return ordered array of int parameters for kernel."""
        return np.array([int(getattr(self, name)) for name in self.kernel_int_names], dtype=np.int32)

    def generate_macros(self):
        """Generate C/CUDA macros for kernel parameter passing."""
        macros = []
        for name in self.kernel_int_names:
            value = int(getattr(self, name))
            macros.append(f"#define {name.upper()} {value}")
        for name in self.kernel_float_names:
            value = float(getattr(self, name))
            macros.append(f"#define {name.upper()} {value:.6f}f")
        return "\n".join(macros) + "\n"

    def __str__(self):
        """String representation of all kernel parameters."""
        params = {
            name: getattr(self, name)
            for name in (self.user_param_names + self.dependent_float_names + self.dependent_int_names)
        }
        return "\n".join(
            [
                f"{key}: {value:.3f}" if isinstance(value, float) else f"{key}: {value}"
                for key, value in params.items()
            ]
        )


class KernelParameters2D(KernelParametersBase):
    """
    2D kernel parameter class.
    """

    # All parameters the user must provide (regardless of type)
    user_param_names = [
        "fs",
        "c1",
        "pitch",
        "f1",
        "f2",
        "bfd",
        "x_step",
        "z_step",
        "t_start",  # floats
        "taps",
        "n_batch",
        "n_elementos",
        "n_ch",
        "n_angles",
        "n_samples",  # ints
        "roi_user",
        "blocksize_img",  # tuples/arrays
    ]

    # Float and int names for kernel (includes dependents)
    kernel_float_names = [
        "fs",
        "c1",
        "pitch",
        "f1",
        "f2",
        "bfd",
        "x_step",
        "z_step",
        "x0_roi",
        "z0_roi",
        "t_start",
        "x_0",
    ]
    kernel_int_names = [
        "taps",
        "n_batch",
        "n_elementos",
        "n_ch",
        "n_angles",
        "nx",
        "nz",
        "n_samples",
    ]

    # Names calculated internally
    dependent_float_names = ["x0_roi", "z0_roi", "x_0"]  # x_0 can be user-defined
    dependent_int_names = ["nx", "nz"]

    # Default values for user parameters if not provided
    default_param_values = {
        "fs": 62.5,  # MHz
        "c1": 1.54,  # mm/us
        "pitch": 0.5,  # mm
        "f1": 1,  # MHz
        "f2": 10,  # MHz
        "bfd": 1,  # twice f#
        "x_step": 0.2,  # mm
        "z_step": 0.2,  # mm
        "t_start": 0.0,  # us
        "taps": 62,  # Number of taps for bandpass filter
        "n_batch": 0,  # Not used in this case
        "n_elementos": 128,
        "n_ch": 128,
        "n_angles": 20,
        "n_samples": 1000,
        "roi_user": [-4.0, 10.0, 1.6, 30.0],  # [xmin, xmax, zmin, zmax] in mm
        "blocksize_img": (32, 32),  # Block size for image processing
    }

    def calculate_dependent_parameters(self, decimals=3):
        """Calculate 2D-specific dependent parameters."""
        self.x0_roi = self.roi_user[0]
        self.z0_roi = self.roi_user[2]
        self.nx, self.nz, self.roi_effective = self.calculate_image_size_and_roi(
            self.roi_user, self.x_step, self.z_step, self.blocksize_img
        )
        self.gridsize_img = (
            self.nz // self.blocksize_img[0],
            self.nx // self.blocksize_img[1],
        )
        self.img_shape = (self.nz, self.nx)
        self.matrix_shape = (self.n_angles, self.n_elementos, self.n_samples)
        if "x_0" not in self.param_dict:
            self.x_0 = np.around((self.n_elementos - 1) * self.pitch / 2, decimals=2)
        else:
            self.x_0 = self.param_dict["x_0"]

    @staticmethod
    def calculate_image_size_and_roi(roi, x_step, z_step, blocksize):
        """
        Compute image size and adjust effective ROI to be a multiple of blocksize.
        """
        x0_roi, x1_roi, z0_roi, z1_roi = roi
        nx = math.ceil((x1_roi - x0_roi) / x_step)
        nz = math.ceil((z1_roi - z0_roi) / z_step)
        nx = math.ceil(nx / blocksize[1]) * blocksize[1]
        nz = math.ceil(nz / blocksize[0]) * blocksize[0]
        x1_effective = x0_roi + nx * x_step
        z1_effective = z0_roi + nz * z_step
        return nx, nz, (x0_roi, x1_effective, z0_roi, z1_effective)

    def multiply_pixels(self, factor):
        """
        Multiply image pixels while preserving ROI.
        Divides x_step and z_step by the given factor and recalculates dependents.

        Args:
            factor (int): Factor by which pixel count is multiplied.
                          Must be an integer greater than 1.
        """
        if factor <= 1 or not isinstance(factor, int):
            raise ValueError("Factor must be an integer greater than 1.")

        # Divide x_step and z_step by factor.
        self.x_step /= factor
        self.z_step /= factor

        # Recompute dependent parameters.
        self.calculate_dependent_parameters()

    def get_imshow_extent(self):
        """
        Returns extent tuple for imshow, based on effective ROI.
        Only applicable for 2D images.
        """
        x0, x1, z0, z1 = self.roi_effective
        return x0, x1, z1, z0  # Flip z for imshow convention


class KernelParameters3D(KernelParametersBase):
    """
    3D kernel parameter class.
    """

    user_param_names = [
        "fs",
        "c1",
        "pitch_x",
        "pitch_y",
        "f1",
        "f2",
        "bfd",
        "x_step",
        "y_step",
        "z_step",
        "t_start",
        "elem_x_width",
        "elem_y_width",
        "fc",  # floats
        "taps",
        "n_batch",
        "n_elementos",
        "nel_x",
        "nel_y",
        "n_ch",
        "n_waves",
        "n_samples",
        "n_reflectors",  # ints
        "roi_user",
        "blocksize_img",  # tuples/arrays
    ]

    kernel_float_names = [
        "fs",
        "c1",
        "pitch_x",
        "pitch_y",
        "f1",
        "f2",
        "bfd",
        "x_step",
        "y_step",
        "z_step",
        "x0_roi",
        "y0_roi",
        "z0_roi",
        "t_start",
        "x_0",
        "y_0",
        "elem_x_width",
        "elem_y_width",
        "fc",
    ]

    kernel_int_names = [
        "taps",
        "n_batch",
        "n_elementos",
        "nel_x",
        "nel_y",
        "n_ch",
        "n_waves",
        "nx",
        "ny",
        "nz",
        "n_samples",
        "n_reflectors",
    ]

    # x_0, y_0 can be user-defined
    dependent_float_names = ["x0_roi", "y0_roi", "z0_roi", "roi_effective", "x_0", "y_0"]
    dependent_int_names = ["nx", "ny", "nz", "gridsize_img", "img_shape"]

    # Default values for user parameters if not provided
    default_param_values = {
        "fs": 62.5,  # MHz
        "c1": 1.54,  # mm/us
        "pitch_x": 0.5,  # mm
        "pitch_y": 0.5,  # mm
        "f1": 1,  # MHz
        "f2": 10,  # MHz
        "bfd": 1,  # twice f#
        "x_step": 0.2,  # mm
        "y_step": 0.2,  # mm
        "z_step": 0.2,  # mm
        "t_start": 0.0,  # us
        "taps": 62,  # Number of taps for bandpass filter
        "n_batch": 0,  # Not used in this case
        "nel_x": 16,  # Number of elements in x direction
        "nel_y": 8,  # Number of elements in y direction
        "n_elementos": 128,
        "n_ch": 128,
        "n_waves": 5,
        "n_samples": 1000,
        "roi_user": [-4, 4, -4, 4, 1.6, 30.0],
        "blocksize_img": (8, 8, 8),  # Block size for image processing
        "n_reflectors": 0,
        "elem_x_width": 1,
        "elem_y_width": 1,
        "fc": 5,  # MHz
    }

    def calculate_dependent_parameters(self, decimals=3):
        """Calculate 3D-specific dependent parameters."""
        self.x0_roi = self.roi_user[0]
        self.y0_roi = self.roi_user[2]
        self.z0_roi = self.roi_user[4]
        self.nx, self.ny, self.nz, self.roi_effective = self.calculate_image_size_and_roi(
            self.roi_user,
            self.x_step,
            self.y_step,
            self.z_step,
            self.blocksize_img,
        )
        self.gridsize_img = (
            self.nx // self.blocksize_img[0],
            self.ny // self.blocksize_img[1],
            self.nz // self.blocksize_img[2],
        )
        self.img_shape = (self.nx, self.ny, self.nz)
        self.matrix_shape = (self.n_waves, self.n_elementos, self.n_samples)
        if "x_0" not in self.param_dict:
            self.x_0 = np.around((self.nel_x - 1) * self.pitch_x / 2, decimals=decimals)
            self.y_0 = np.around((self.nel_y - 1) * self.pitch_y / 2, decimals=decimals)
        else:
            self.x_0 = self.param_dict["x_0"]
            self.y_0 = self.param_dict["y_0"]

    @staticmethod
    def calculate_image_size_and_roi(roi, x_step, y_step, z_step, blocksize):
        """
        Compute 3D image size and adjust effective ROI to be a multiple of blocksize.
        """
        x0_roi, x1_roi, y0_roi, y1_roi, z0_roi, z1_roi = roi
        nx = math.ceil((x1_roi - x0_roi) / x_step)
        ny = math.ceil((y1_roi - y0_roi) / y_step)
        nz = math.ceil((z1_roi - z0_roi) / z_step)

        nx = math.ceil(nx / blocksize[0]) * blocksize[0]
        ny = math.ceil(ny / blocksize[1]) * blocksize[1]
        nz = math.ceil(nz / blocksize[2]) * blocksize[2]

        x1_effective = x0_roi + nx * x_step
        y1_effective = y0_roi + ny * y_step
        z1_effective = z0_roi + nz * z_step
        return nx, ny, nz, (x0_roi, x1_effective, y0_roi, y1_effective, z0_roi, z1_effective)

    def get_imshow_extent(self, view="xz"):
        """ Returns extent for imshow according to the selected view.
        Args:
            view (str): Desired view: 'xz', 'yz', or 'xy'

        Returns:
            tuple: (extent_h0, extent_h1, extent_v1, extent_v0)
                where h is horizontal and v vertical in the image
        """

        x0, x1, y0, y1, z0, z1 = self.roi_effective

        if view == "xz":
            return x0, x1, z1, z0  # Flip z for imshow convention
        if view == "yz":
            return y0, y1, z1, z0  # Flip z for imshow convention
        if view == "xy":
            # Using x as vertical axis so the view is from above the array (top-view).
            return y0, y1, x1, x0
        raise ValueError("Invalid view. Use 'xz', 'yz' or 'xy'")

    def get_voxel_size(self, axis):
        """ Returns the voxel size along the specified axis.
        Args:
            axis (int): Axis index (0 for x, 1 for y, 2 for z)

        Returns:
            float: Voxel size along the specified axis
        """
        if axis == 0:
            return self.x_step
        if axis == 1:
            return self.y_step
        if axis == 2:
            return self.z_step
        raise ValueError("Invalid axis. Use 0 for x, 1 for y, or 2 for z.")


def get_enum_names(enum_code, enum_type):
    # Regex to extract enum members between { and }
    pattern = rf"enum\s+{enum_type}\s*\{{([^}}]+)\}}"
    match = re.search(pattern, enum_code, re.MULTILINE | re.DOTALL)
    if not match:
        return []
    # Split by commas, remove comments and whitespace
    members = match.group(1).split(",")
    names = []
    for member in members:
        member = member.strip()
        # Remove trailing comments
        member = member.split("//")[0].strip()
        if member and not member.endswith("_COUNT"):  # Ignore the *_COUNT member
            names.append(member)
    return names
