import tensorflow as tf
import numpy as np


class CoordinateManager:
    """
    Class to manage coordinates and features for beamforming.

    Generates and stores:
    - Spatial coordinates: x, z, x_elem (in mm and scaled by D)
    - Physical features: |x - x_elem|, z, D/2 - |x| (in mm and scaled by D)

    Scaling: all coordinates are scaled by dividing by D (array aperture)

    Available formats:
    - Grid: (n_elem, nz, nx, 3) - for complete grids
    - Flat: (n_elem * nz * nx, 3) - for passing to INR
    - Points: (n_elem * n_points, 3) - for arbitrary points
    """

    def __init__(self, kp):
        """
        Args:
            kp: KernelParameters object (2D or 3D) with system parameters
            Type of features to generate
                - 'physical': [|x - x_elem|, z, D/2 - |x|]
                - 'coordinates': [x, z, x_elem]
        """
        self.kp = kp

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

        # Features [|x-x_elem|, z, D/2-|x|]
        self._features_grid_mm = None  # (n_elem, nz, nx, 3)
        self._features_flat_mm = None  # (n_elem * nz * nx, 3)
        self._features_grid_scaled = None  # (n_elem, nz, nx, 3)
        self._features_flat_scaled = None  # (n_elem * nz * nx, 3)

        print(f"📐 [CoordinateManager] Initialized:")
        print(f"   Grid shape: (nz={self.nz}, nx={self.nx})")
        print(f"   Elements: {self.n_elem}")
        print(f"   Aperture D: {self.D:.2f} mm")

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

    def _compute_physical_features_mm(self, X_mm, Z_mm, x_elem_mm):
        """
        Computes physical features IN MILLIMETERS.

        Args:
            X_mm: grid of x coordinates (nz, nx) in mm
            Z_mm: grid of z coordinates (nz, nx) in mm
            x_elem_mm: x coordinate of the element in mm (scalar)

        Returns:
            tuple: (dist_to_elem, depth, dist_to_edge) in mm
        """
        # Feature 1: |x - x_elem| in mm
        dist_to_elem = tf.abs(X_mm - x_elem_mm)

        # Feature 2: z in mm
        depth = Z_mm

        # Feature 3: D/2 - |x - x_center| in mm
        x_from_center = tf.abs(X_mm - self.x_center)
        dist_to_edge = self.D_half - x_from_center

        return dist_to_elem, depth, dist_to_edge

    def _compute_physical_features_scaled(self, X_mm, Z_scaled, x_elem_mm):
        """
        Computes physical features SCALED by D.

        Args:
            X_mm: grid of x coordinates (nz, nx) in mm
            Z_scaled: grid of z coordinates (nz, nx) scaled by D
            x_elem_mm: x coordinate of the element in mm (scalar)

        Returns:
            tuple: (dist_to_elem_scaled, depth_scaled, dist_to_edge_scaled)
        """
        # Feature 1: |x - x_elem| / D
        dist_to_elem = tf.abs(X_mm - x_elem_mm)
        dist_to_elem_scaled = dist_to_elem / self.D

        # Feature 2: z / D (already scaled)
        depth_scaled = Z_scaled

        # Feature 3: (D/2 - |x - x_center|) / D = 0.5 - |x - x_center| / D
        x_from_center = tf.abs(X_mm - self.x_center)
        dist_to_edge = self.D_half - x_from_center
        dist_to_edge_scaled = dist_to_edge / self.D
        return dist_to_elem_scaled, depth_scaled, dist_to_edge_scaled

    # ========================================================================
    # PRIVATE METHODS: Grid construction
    # ========================================================================

    def _build_coordinates_grid_mm(self):
        """Builds grid of coordinates [x, z, x_elem] IN MM."""
        x_mm_tf = tf.constant(self.x_coords_mm, dtype=tf.float32)
        z_mm_tf = tf.constant(self.z_coords_mm, dtype=tf.float32)
        x_elem_mm_tf = tf.constant(self.x_elem_coords_mm, dtype=tf.float32)

        Z_mm, X_mm = tf.meshgrid(z_mm_tf, x_mm_tf, indexing='ij')

        coords_list = []
        for x_elem_mm in x_elem_mm_tf:
            x_elem_grid = tf.fill(X_mm.shape, x_elem_mm)
            coords = tf.stack([X_mm, Z_mm, x_elem_grid], axis=-1)  # (nz, nx, 3)
            coords_list.append(coords)

        self._coords_grid_mm = tf.stack(coords_list, axis=0)  # (n_elem, nz, nx, 3)
        self._coords_flat_mm = tf.reshape(self._coords_grid_mm, [-1, 3])

    def _build_coordinates_grid_scaled(self):
        """Builds grid of coordinates [x, z, x_elem] SCALED by D."""
        x_scaled_tf = tf.constant(self.x_coords_scaled, dtype=tf.float32)
        z_scaled_tf = tf.constant(self.z_coords_scaled, dtype=tf.float32)
        x_elem_scaled_tf = tf.constant(self.x_elem_coords_scaled, dtype=tf.float32)

        Z_scaled, X_scaled = tf.meshgrid(z_scaled_tf, x_scaled_tf, indexing='ij')

        coords_list = []
        for x_elem_scaled in x_elem_scaled_tf:
            x_elem_grid = tf.fill(X_scaled.shape, x_elem_scaled)
            coords = tf.stack([X_scaled, Z_scaled, x_elem_grid], axis=-1)  # (nz, nx, 3)
            coords_list.append(coords)

        self._coords_grid_scaled = tf.stack(coords_list, axis=0)  # (n_elem, nz, nx, 3)
        self._coords_flat_scaled = tf.reshape(self._coords_grid_scaled, [-1, 3])

    def _build_features_grid_mm(self):
        """Builds grid of physical features IN MM."""
        x_mm_tf = tf.constant(self.x_coords_mm, dtype=tf.float32)
        z_mm_tf = tf.constant(self.z_coords_mm, dtype=tf.float32)
        x_elem_mm_tf = tf.constant(self.x_elem_coords_mm, dtype=tf.float32)

        Z_mm, X_mm = tf.meshgrid(z_mm_tf, x_mm_tf, indexing='ij')

        features_list = []
        for x_elem_mm in x_elem_mm_tf:
            feat1, feat2, feat3 = self._compute_physical_features_mm(X_mm, Z_mm, x_elem_mm)
            features = tf.stack([feat1, feat2, feat3], axis=-1)  # (nz, nx, 3)
            features_list.append(features)

        self._features_grid_mm = tf.stack(features_list, axis=0)  # (n_elem, nz, nx, 3)
        self._features_flat_mm = tf.reshape(self._features_grid_mm, [-1, 3])

    def _build_features_grid_scaled(self):
        """Builds grid of physical features SCALED by D."""
        x_mm_tf = tf.constant(self.x_coords_mm, dtype=tf.float32)
        z_scaled_tf = tf.constant(self.z_coords_scaled, dtype=tf.float32)
        x_elem_mm_tf = tf.constant(self.x_elem_coords_mm, dtype=tf.float32)

        Z_scaled, X_mm = tf.meshgrid(z_scaled_tf, x_mm_tf, indexing='ij')

        features_list = []
        for x_elem_mm in x_elem_mm_tf:
            feat1, feat2, feat3 = self._compute_physical_features_scaled(X_mm, Z_scaled, x_elem_mm)
            features = tf.stack([feat1, feat2, feat3], axis=-1)  # (nz, nx, 3)
            features_list.append(features)

        self._features_grid_scaled = tf.stack(features_list, axis=0)  # (n_elem, nz, nx, 3)
        self._features_flat_scaled = tf.reshape(self._features_grid_scaled, [-1, 3])

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
        Returns features [|x-x_elem|, z, D/2-|x|] in grid format.

        Args:
            scaled: False for mm, True for scaled by D

        Returns:
            tensor: (n_elem, nz, nx, 3)
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
        Returns features [|x-x_elem|, z, D/2-|x|] in flat format.
        This is the format to pass to ApodizationINR.

        Args:
            scaled: False for mm, True for scaled by D

        Returns:
            tensor: (n_elem * nz * nx, 3)
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
            elem_idx: index of the element
            scaled: False for mm, True for scaled by D

        Returns:
            tensor: (nz, nx, 3)
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