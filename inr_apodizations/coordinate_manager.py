import tensorflow as tf
import numpy as np
import warnings


ALLOWED_PHYSICAL_FEATURE_COMPONENTS = (
    "z",
    "x",
    "abs_x",
    "xrel",
    "abs_xrel",
    "dist_to_edge",
)

LEGACY_FEATURE_SET_TO_COMPONENTS = {
    "distance_depth_edge": ("abs_xrel", "z", "dist_to_edge"),
    "distance_depth_center": ("abs_xrel", "z", "abs_x"),
    "distance_depth": ("abs_xrel", "z"),
    "x_rel_depth_edge": ("xrel", "z", "dist_to_edge"),
}


class CoordinateManager:
    """
    Class to manage coordinates and features for beamforming.

    Generates and stores:
    - Spatial coordinates: x, z, x_elem (in mm and scaled by D)
        - Physical features, configurable by ordered component tokens:
            - 'z': z
            - 'x': x
            - 'abs_x': |x|
            - 'xrel': x - x_elem
            - 'abs_xrel': |x - x_elem|
            - 'dist_to_edge': D/2 - |x| (legacy-compatible)

    Scaling: all coordinates are scaled by dividing by D (array aperture)

    Available formats:
    - Grid: (n_elem, nz, nx, 3) - for complete grids
    - Flat: (n_elem * nz * nx, 3) - for passing to INR
    - Points: (n_elem * n_points, 3) - for arbitrary points
    """

    def __init__(
        self,
        kp,
        physical_feature_set="distance_depth_edge",
        physical_feature_components=None,
    ):
        """
        Initializes the coordinate manager.

        Args:
            kp: KernelParameters object (2D or 3D) with system parameters.
            physical_feature_set: Legacy physical feature preset.
            physical_feature_components: Optional ordered feature token list.
                Allowed tokens: ('z', 'x', 'abs_x', 'xrel', 'abs_xrel', 'dist_to_edge').
                If provided, this takes precedence over `physical_feature_set`.

        Raises:
            ValueError: If the requested feature configuration is invalid.
        """
        self.kp = kp
        self.physical_feature_set = str(physical_feature_set)
        self.physical_feature_components = self._resolve_physical_feature_components(
            physical_feature_set=self.physical_feature_set,
            physical_feature_components=physical_feature_components,
        )
        self.physical_feature_names = self.physical_feature_components
        self.n_physical_features = len(self.physical_feature_names)

        # Verify that it is 2D (for now)
        if not hasattr(kp, 'nx') or not hasattr(kp, 'nz'):
            raise ValueError("Only 2D parameters are supported for now")

        # Extract dimensions
        self.nx = kp.nx
        self.nz = kp.nz
        self.n_elem = kp.n_elements
        self.shape = (self.n_elem, self.nz, self.nx)

        # ====================================================================
        # STEP 1: Create spatial coordinates in mm
        # ====================================================================
        self.x_coords_mm = self._create_x_coords()  # (nx,)
        self.z_coords_mm = self._create_z_coords()  # (nz,)
        self.x_elem_coords_mm = self._create_x_elem_coords()  # (n_elem,)

        # Calculate aperture parameters
        self.D_half = kp.x_0
        self.D = 2 * self.D_half
        self.x_center = 0.0  # Assuming center at x=0

        # ====================================================================
        # STEP 2: Scale coordinates by D
        # ====================================================================
        self.x_coords_scaled = self.x_coords_mm / self.D  # (nx,)
        self.z_coords_scaled = self.z_coords_mm / self.D  # (nz,)
        self.x_elem_coords_scaled = self.x_elem_coords_mm / self.D  # (n_elem,)

        # ====================================================================
        # STEP 3: Grids lazy initialization
        # ====================================================================
        # Coordinates [x, z, x_elem]
        self._coords_grid_mm = None  # (n_elem, nz, nx, 3)
        self._coords_flat_mm = None  # (n_elem * nz * nx, 3)
        self._coords_grid_scaled = None  # (n_elem, nz, nx, 3)
        self._coords_flat_scaled = None  # (n_elem * nz * nx, 3)

        # Features depend on the selected physical feature set.
        self._features_grid_mm = None  # (n_elem, nz, nx, n_physical_features)
        self._features_flat_mm = None  # (n_elem * nz * nx, n_physical_features)
        self._features_grid_scaled = None  # (n_elem, nz, nx, n_physical_features)
        self._features_flat_scaled = None  # (n_elem * nz * nx, n_physical_features)

        print(f"📐 [CoordinateManager] Initialized:")
        print(f"   Grid shape: (nz={self.nz}, nx={self.nx})")
        print(f"   Elements: {self.n_elem}")
        print(f"   Aperture D: {self.D:.2f} mm")
        print(f"   Physical feature set: {self.physical_feature_set}")
        print(f"   Physical feature components: {self.physical_feature_components}")

    def _resolve_physical_feature_components(self, physical_feature_set, physical_feature_components):
        """Resolve selected physical feature components from explicit list or legacy preset."""
        if physical_feature_components is not None:
            if not isinstance(physical_feature_components, (list, tuple)):
                raise ValueError(
                    "physical_feature_components must be a list/tuple of strings "
                    f"from {ALLOWED_PHYSICAL_FEATURE_COMPONENTS}"
                )

            resolved = tuple(str(token).strip() for token in physical_feature_components)
            if len(resolved) not in (2, 3):
                raise ValueError(
                    "physical_feature_components must contain exactly 2 or 3 tokens"
                )
            if len(set(resolved)) != len(resolved):
                raise ValueError("physical_feature_components cannot contain duplicate tokens")

            invalid_tokens = [
                token for token in resolved if token not in ALLOWED_PHYSICAL_FEATURE_COMPONENTS
            ]
            if invalid_tokens:
                raise ValueError(
                    "Invalid physical feature tokens: "
                    f"{invalid_tokens}. Allowed tokens: {ALLOWED_PHYSICAL_FEATURE_COMPONENTS}"
                )

            self.physical_feature_set = "components_list"
            return resolved

        if physical_feature_set not in LEGACY_FEATURE_SET_TO_COMPONENTS:
            raise ValueError(
                "physical_feature_set must be one of "
                f"{tuple(LEGACY_FEATURE_SET_TO_COMPONENTS.keys())} "
                "when physical_feature_components is not provided"
            )

        warnings.warn(
            "physical_feature_set is deprecated. Use physical_feature_components in YAML.",
            DeprecationWarning,
            stacklevel=2,
        )
        return LEGACY_FEATURE_SET_TO_COMPONENTS[physical_feature_set]

    # ========================================================================
    # PRIVATE METHODS: Base coordinate creation
    # ========================================================================

    def _create_x_coords(self):
        """Creates x (lateral) coordinates in mm."""
        x0, x1 = self.kp.roi_effective[0], self.kp.roi_effective[1]
        return np.arange(x0, x1, (x1 - x0) / self.nx, dtype=np.float32)

    def _create_z_coords(self):
        """Creates z (axial) coordinates in mm."""
        z0, z1 = self.kp.roi_effective[2], self.kp.roi_effective[3]
        return np.arange(z0, z1, (z1 - z0) / self.nz, dtype=np.float32)

    def _create_x_elem_coords(self):
        """Creates x coordinates of the elements in mm."""
        x_elem = np.arange(self.kp.n_elements, dtype=np.float32) * self.kp.pitch - self.kp.x_0
        return x_elem

    # ========================================================================
    # PRIVATE METHODS: Physical feature calculation
    # ========================================================================

    def _stack_physical_features(self, feature_tensors):
        """
        Stacks the selected physical features along the last axis.

        Args:
            feature_tensors: Sequence of feature tensors with shape (nz, nx).

        Returns:
            tensor: Physical features with shape (nz, nx, n_physical_features).
        """
        return tf.stack(feature_tensors, axis=-1)

    def _component_tensor_mm(self, token, x_grid, z_grid, x_elem_grid):
        """Return one physical feature tensor in millimeters for the selected token."""
        if token == "z":
            return tf.broadcast_to(z_grid, [self.n_elem, self.nz, self.nx])
        if token == "x":
            return tf.broadcast_to(x_grid, [self.n_elem, self.nz, self.nx])
        if token == "abs_x":
            return tf.broadcast_to(tf.abs(x_grid - self.x_center), [self.n_elem, self.nz, self.nx])
        if token == "xrel":
            return tf.broadcast_to(x_grid - x_elem_grid, [self.n_elem, self.nz, self.nx])
        if token == "abs_xrel":
            return tf.broadcast_to(tf.abs(x_grid - x_elem_grid), [self.n_elem, self.nz, self.nx])
        if token == "dist_to_edge":
            return tf.broadcast_to(self.D_half - tf.abs(x_grid - self.x_center), [self.n_elem, self.nz, self.nx])
        raise ValueError(f"Unsupported physical feature token: {token}")

    def _component_tensor_scaled(self, token, x_grid_mm, z_grid_scaled, x_elem_grid_mm):
        """Return one D-scaled physical feature tensor for the selected token."""
        if token == "z":
            return tf.broadcast_to(z_grid_scaled, [self.n_elem, self.nz, self.nx])
        if token == "x":
            return tf.broadcast_to(x_grid_mm / self.D, [self.n_elem, self.nz, self.nx])
        if token == "abs_x":
            abs_x_scaled = tf.abs(x_grid_mm - self.x_center) / self.D
            return tf.broadcast_to(abs_x_scaled, [self.n_elem, self.nz, self.nx])
        if token == "xrel":
            xrel_scaled = (x_grid_mm - x_elem_grid_mm) / self.D
            return tf.broadcast_to(xrel_scaled, [self.n_elem, self.nz, self.nx])
        if token == "abs_xrel":
            abs_xrel_scaled = tf.abs(x_grid_mm - x_elem_grid_mm) / self.D
            return tf.broadcast_to(abs_xrel_scaled, [self.n_elem, self.nz, self.nx])
        if token == "dist_to_edge":
            dist_to_edge_scaled = (self.D_half - tf.abs(x_grid_mm - self.x_center)) / self.D
            return tf.broadcast_to(dist_to_edge_scaled, [self.n_elem, self.nz, self.nx])
        raise ValueError(f"Unsupported physical feature token: {token}")

    # ========================================================================
    # PRIVATE METHODS: Grid construction
    # ========================================================================

    def _build_coordinates_grid_mm(self):
        """Builds grid of coordinates [x, z, x_elem] IN MM."""
        x_mm_tf = tf.constant(self.x_coords_mm, dtype=tf.float32)
        z_mm_tf = tf.constant(self.z_coords_mm, dtype=tf.float32)
        x_elem_mm_tf = tf.constant(self.x_elem_coords_mm, dtype=tf.float32)

        x_grid = tf.reshape(x_mm_tf, [1, 1, -1])
        z_grid = tf.reshape(z_mm_tf, [1, -1, 1])
        x_elem_grid = tf.reshape(x_elem_mm_tf, [-1, 1, 1])

        x_grid = tf.broadcast_to(x_grid, [self.n_elem, self.nz, self.nx])
        z_grid = tf.broadcast_to(z_grid, [self.n_elem, self.nz, self.nx])
        x_elem_grid = tf.broadcast_to(x_elem_grid, [self.n_elem, self.nz, self.nx])

        self._coords_grid_mm = tf.stack([x_grid, z_grid, x_elem_grid], axis=-1)
        self._coords_flat_mm = tf.reshape(self._coords_grid_mm, [-1, 3])

    def _build_coordinates_grid_scaled(self):
        """Builds grid of coordinates [x, z, x_elem] SCALED by D."""
        x_scaled_tf = tf.constant(self.x_coords_scaled, dtype=tf.float32)
        z_scaled_tf = tf.constant(self.z_coords_scaled, dtype=tf.float32)
        x_elem_scaled_tf = tf.constant(self.x_elem_coords_scaled, dtype=tf.float32)

        x_grid = tf.reshape(x_scaled_tf, [1, 1, -1])
        z_grid = tf.reshape(z_scaled_tf, [1, -1, 1])
        x_elem_grid = tf.reshape(x_elem_scaled_tf, [-1, 1, 1])

        x_grid = tf.broadcast_to(x_grid, [self.n_elem, self.nz, self.nx])
        z_grid = tf.broadcast_to(z_grid, [self.n_elem, self.nz, self.nx])
        x_elem_grid = tf.broadcast_to(x_elem_grid, [self.n_elem, self.nz, self.nx])

        self._coords_grid_scaled = tf.stack([x_grid, z_grid, x_elem_grid], axis=-1)
        self._coords_flat_scaled = tf.reshape(self._coords_grid_scaled, [-1, 3])

    def _build_features_grid_mm(self):
        """Builds grid of physical features IN MM."""
        x_mm_tf = tf.constant(self.x_coords_mm, dtype=tf.float32)
        z_mm_tf = tf.constant(self.z_coords_mm, dtype=tf.float32)
        x_elem_mm_tf = tf.constant(self.x_elem_coords_mm, dtype=tf.float32)

        x_grid = tf.reshape(x_mm_tf, [1, 1, -1])
        z_grid = tf.reshape(z_mm_tf, [1, -1, 1])
        x_elem_grid = tf.reshape(x_elem_mm_tf, [-1, 1, 1])

        feature_tensors = tuple(
            self._component_tensor_mm(token, x_grid, z_grid, x_elem_grid)
            for token in self.physical_feature_components
        )

        self._features_grid_mm = self._stack_physical_features(feature_tensors)
        self._features_flat_mm = tf.reshape(self._features_grid_mm, [-1, self.n_physical_features])

    def _build_features_grid_scaled(self):
        """Builds grid of physical features SCALED by D."""
        x_mm_tf = tf.constant(self.x_coords_mm, dtype=tf.float32)
        z_scaled_tf = tf.constant(self.z_coords_scaled, dtype=tf.float32)
        x_elem_mm_tf = tf.constant(self.x_elem_coords_mm, dtype=tf.float32)

        x_grid = tf.reshape(x_mm_tf, [1, 1, -1])
        z_grid = tf.reshape(z_scaled_tf, [1, -1, 1])
        x_elem_grid = tf.reshape(x_elem_mm_tf, [-1, 1, 1])

        feature_tensors = tuple(
            self._component_tensor_scaled(token, x_grid, z_grid, x_elem_grid)
            for token in self.physical_feature_components
        )

        self._features_grid_scaled = self._stack_physical_features(feature_tensors)
        self._features_flat_scaled = tf.reshape(
            self._features_grid_scaled,
            [-1, self.n_physical_features]
        )

    # ========================================================================
    # PUBLIC METHODS: Get 1D coordinates
    # ========================================================================

    def get_coordinates_1d(self, scaled=False):
        """
        Returns 1D spatial coordinates.

        Args:
            scaled: False for mm, True for scaled by D

        Returns:
            dict: {
                'x': array (nx,),
                'z': array (nz,),
                'x_elem': array (n_elem,)
            }
        """
        if scaled:
            return {
                'x': self.x_coords_scaled,
                'z': self.z_coords_scaled,
                'x_elem': self.x_elem_coords_scaled
            }
        else:
            return {
                'x': self.x_coords_mm,
                'z': self.z_coords_mm,
                'x_elem': self.x_elem_coords_mm
            }

    # ========================================================================
    # PUBLIC METHODS: Get grid/flat coordinates
    # ========================================================================

    def get_coordinates_grid(self, scaled=False):
        """
        Returns coordinates [x, z, x_elem] in grid format.

        Args:
            scaled: False for mm, True for scaled by D

        Returns:
            tensor: (n_elem, nz, nx, 3)
        """
        if scaled:
            if self._coords_grid_scaled is None:
                self._build_coordinates_grid_scaled()
            return self._coords_grid_scaled
        else:
            if self._coords_grid_mm is None:
                self._build_coordinates_grid_mm()
            return self._coords_grid_mm

    def get_coordinates_flat(self, scaled=False):
        """
        Returns coordinates [x, z, x_elem] in flat format.

        Args:
            scaled: False for mm, True for scaled by D

        Returns:
            tensor: (n_elem * nz * nx, 3)
        """
        if scaled:
            if self._coords_flat_scaled is None:
                self._build_coordinates_grid_scaled()
            return self._coords_flat_scaled
        else:
            if self._coords_flat_mm is None:
                self._build_coordinates_grid_mm()
            return self._coords_flat_mm

    # ========================================================================
    # PUBLIC METHODS: Get grid/flat features
    # ========================================================================

    def get_features_grid(self, scaled=False):
        """
        Returns the selected physical features in grid format.

        Args:
            scaled: False for mm, True for scaled by D.

        Returns:
            tensor: (n_elem, nz, nx, n_physical_features)
        """
        if scaled:
            if self._features_grid_scaled is None:
                self._build_features_grid_scaled()
            return self._features_grid_scaled
        else:
            if self._features_grid_mm is None:
                self._build_features_grid_mm()
            return self._features_grid_mm

    def get_features_flat(self, scaled=False):
        """
        Returns the selected physical features in flat format.
        This is the format to pass to ApodizationINR.

        Args:
            scaled: False for mm, True for scaled by D.

        Returns:
            tensor: (n_elem * nz * nx, n_physical_features)
        """
        if scaled:
            if self._features_flat_scaled is None:
                self._build_features_grid_scaled()
            return self._features_flat_scaled
        else:
            if self._features_flat_mm is None:
                self._build_features_grid_mm()
            return self._features_flat_mm

    # ========================================================================
    # PUBLIC METHODS: Get features/coords for specific element
    # ========================================================================

    def get_features_for_element(self, elem_idx, scaled=False):
        """
        Returns features for a specific element.

        Args:
            elem_idx: Index of the element.
            scaled: False for mm, True for scaled by D.

        Returns:
            tensor: (nz, nx, n_physical_features)
        """
        if elem_idx < 0 or elem_idx >= self.n_elem:
            raise ValueError(f"elem_idx must be between 0 and {self.n_elem - 1}")

        features_grid = self.get_features_grid(scaled=scaled)
        return features_grid[elem_idx]

    def get_coordinates_for_element(self, elem_idx, scaled=False):
        """
        Returns coordinates for a specific element.

        Args:
            elem_idx: index of the element
            scaled: False for mm, True for scaled by D

        Returns:
            tensor: (nz, nx, 3)
        """
        if elem_idx < 0 or elem_idx >= self.n_elem:
            raise ValueError(f"elem_idx must be between 0 and {self.n_elem - 1}")

        coords_grid = self.get_coordinates_grid(scaled=scaled)
        return coords_grid[elem_idx]

    # ========================================================================
    # PUBLIC METHODS: Aperture information
    # ========================================================================

    def get_aperture_info(self):
        """
        Returns aperture information.

        Returns:
            dict: {
                'D': float (mm),
                'D_half': float (mm),
                'x_center': float (mm)
            }
        """
        return {
            'D': self.D,
            'D_half': self.D_half,
            'x_center': self.x_center
        }

    def get_physical_feature_info(self):
        """
        Returns information about the selected physical feature set.

        Returns:
            dict: {
                'physical_feature_set': str,
                'physical_feature_components': tuple[str, ...],
                'physical_feature_names': tuple[str, ...],
                'n_physical_features': int
            }
        """
        return {
            'physical_feature_set': self.physical_feature_set,
            'physical_feature_components': self.physical_feature_components,
            'physical_feature_names': self.physical_feature_names,
            'n_physical_features': self.n_physical_features,
        }