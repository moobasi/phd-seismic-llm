"""
================================================================================
3D SEISMIC VISUALIZATION MODULE
================================================================================

Provides volumetric 3D rendering for seismic data and fault surfaces.
Uses PyVista (VTK) when available, falls back to matplotlib 3D.

Features:
- 3D volume rendering with opacity control
- Fault surface visualization
- 2D/3D overlay comparison
- Interactive viewing controls

Author: Moses Ekene Obasi
PhD Research - University of Calabar
================================================================================
"""

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import json

# Check for 3D libraries
PYVISTA_AVAILABLE = False
try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    pass

MPL_3D_AVAILABLE = False
try:
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    MPL_3D_AVAILABLE = True
except ImportError:
    pass

# Check for PIL (for image display)
PIL_AVAILABLE = False
try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    pass


class Seismic3DViewer:
    """
    3D visualization for seismic data and fault surfaces.

    Uses PyVista if available (better quality), falls back to matplotlib 3D.
    Designed to be embedded in a Tkinter application.
    """

    def __init__(self, parent: tk.Widget, colors: Optional[Dict] = None):
        """
        Initialize 3D viewer.

        Args:
            parent: Tkinter parent widget
            colors: Color theme dictionary
        """
        self.parent = parent
        self.colors = colors or {
            'bg': '#1a1a2e',
            'surface': '#16213e',
            'overlay': '#0f3460',
            'fg': '#e6e6e6',
            'accent': '#e94560',
            'success': '#4ecca3',
            'warning': '#ffc107'
        }

        self.seismic_data: Optional[np.ndarray] = None
        self.fault_data: Optional[np.ndarray] = None
        self.geometry: Dict = {}

        # Well data for 3D display
        self.wells: Dict[str, Dict] = {}
        self.well_logs: Dict[str, Dict] = {}

        # Display settings
        self.opacity = 0.3
        self.colormap = "seismic"
        self.fault_threshold = 0.5
        self.show_faults_var = tk.BooleanVar(value=True)
        self.show_seismic_var = tk.BooleanVar(value=True)
        self.show_wells_var = tk.BooleanVar(value=True)

        # Downsampling for performance
        self.downsample_factor = 4

        # Try to load well data
        self._load_well_data()

        # Initialize viewer
        if PYVISTA_AVAILABLE:
            self._init_pyvista()
        elif MPL_3D_AVAILABLE:
            self._init_matplotlib_3d()
        else:
            self._init_fallback()

    def _init_pyvista(self):
        """Initialize PyVista 3D viewer (off-screen rendering for Tkinter)."""
        self.viewer_type = "pyvista"

        # Main frame
        main_frame = ttk.Frame(self.parent)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Control frame
        self._create_controls(main_frame)

        # Display frame for rendered image
        self.display_frame = ttk.Frame(main_frame)
        self.display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Label to show rendered image
        self.img_label = ttk.Label(self.display_frame, text="Click 'Render 3D' to generate view")
        self.img_label.pack(expand=True)

        # PyVista plotter (off-screen)
        self.plotter = None

    def _init_matplotlib_3d(self):
        """Initialize matplotlib 3D fallback."""
        self.viewer_type = "matplotlib"

        # Main frame
        main_frame = ttk.Frame(self.parent)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Control frame
        self._create_controls(main_frame)

        # Matplotlib figure
        self.fig = Figure(figsize=(10, 8), facecolor=self.colors['bg'])
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor(self.colors['surface'])

        self.canvas_frame = ttk.Frame(main_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.canvas = FigureCanvasTkAgg(self.fig, self.canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add matplotlib toolbar
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar_frame = ttk.Frame(self.canvas_frame)
        toolbar_frame.pack(fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

    def _init_fallback(self):
        """Fallback when no 3D library available."""
        self.viewer_type = "none"
        label = ttk.Label(
            self.parent,
            text="3D Visualization requires matplotlib or PyVista.\n\n"
                 "Install with:\n"
                 "  pip install matplotlib\n"
                 "or\n"
                 "  pip install pyvista",
            justify=tk.CENTER,
            font=('Segoe UI', 11)
        )
        label.pack(expand=True, pady=50)

    def _create_controls(self, parent):
        """Create control panel."""
        control_frame = ttk.LabelFrame(parent, text="3D View Controls")
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        # Row 1: Display options
        row1 = ttk.Frame(control_frame)
        row1.pack(fill=tk.X, padx=5, pady=5)

        ttk.Checkbutton(row1, text="Show Seismic",
                       variable=self.show_seismic_var).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(row1, text="Show Faults",
                       variable=self.show_faults_var).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(row1, text="Show Wells",
                       variable=self.show_wells_var).pack(side=tk.LEFT, padx=5)

        ttk.Label(row1, text="Opacity:").pack(side=tk.LEFT, padx=(20, 5))
        self.opacity_var = tk.DoubleVar(value=0.3)
        opacity_slider = ttk.Scale(row1, from_=0.0, to=1.0,
                                   variable=self.opacity_var,
                                   orient=tk.HORIZONTAL, length=100)
        opacity_slider.pack(side=tk.LEFT, padx=5)

        ttk.Label(row1, text="Downsample:").pack(side=tk.LEFT, padx=(20, 5))
        self.downsample_var = tk.IntVar(value=4)
        downsample_spin = ttk.Spinbox(row1, from_=1, to=10, width=5,
                                      textvariable=self.downsample_var)
        downsample_spin.pack(side=tk.LEFT, padx=5)

        # Row 2: Rendering and view controls
        row2 = ttk.Frame(control_frame)
        row2.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(row2, text="Render 3D",
                  command=self.render).pack(side=tk.LEFT, padx=5)
        ttk.Button(row2, text="Reset View",
                  command=self._reset_view).pack(side=tk.LEFT, padx=5)

        # View presets
        ttk.Label(row2, text="View:").pack(side=tk.LEFT, padx=(20, 5))
        ttk.Button(row2, text="Top",
                  command=lambda: self._set_view("top")).pack(side=tk.LEFT, padx=2)
        ttk.Button(row2, text="Front",
                  command=lambda: self._set_view("front")).pack(side=tk.LEFT, padx=2)
        ttk.Button(row2, text="Side",
                  command=lambda: self._set_view("side")).pack(side=tk.LEFT, padx=2)
        ttk.Button(row2, text="Isometric",
                  command=lambda: self._set_view("iso")).pack(side=tk.LEFT, padx=2)

        # Row 3: Slice selection for matplotlib
        if self.viewer_type == "matplotlib":
            row3 = ttk.Frame(control_frame)
            row3.pack(fill=tk.X, padx=5, pady=5)

            ttk.Label(row3, text="IL Skip:").pack(side=tk.LEFT)
            self.il_skip_var = tk.IntVar(value=10)
            ttk.Spinbox(row3, from_=1, to=50, width=5,
                       textvariable=self.il_skip_var).pack(side=tk.LEFT, padx=5)

            ttk.Label(row3, text="XL Skip:").pack(side=tk.LEFT, padx=(10, 0))
            self.xl_skip_var = tk.IntVar(value=10)
            ttk.Spinbox(row3, from_=1, to=50, width=5,
                       textvariable=self.xl_skip_var).pack(side=tk.LEFT, padx=5)

            ttk.Label(row3, text="Time Skip:").pack(side=tk.LEFT, padx=(10, 0))
            self.time_skip_var = tk.IntVar(value=20)
            ttk.Spinbox(row3, from_=1, to=100, width=5,
                       textvariable=self.time_skip_var).pack(side=tk.LEFT, padx=5)

    def _load_well_data(self):
        """Load well locations from well_locations.json."""
        try:
            # Try multiple possible locations
            possible_paths = [
                Path(__file__).parent / "well_locations.json",
                Path("well_locations.json"),
                Path(__file__).parent / "outputs" / "well_locations.json"
            ]

            for well_file in possible_paths:
                if well_file.exists():
                    with open(well_file, 'r') as f:
                        data = json.load(f)
                        self.wells = data.get('wells', {})
                        print(f"3D Viewer: Loaded {len(self.wells)} wells from {well_file}")
                        return

            print("3D Viewer: No well_locations.json found")
        except Exception as e:
            print(f"3D Viewer: Error loading well data: {e}")

    def load_well_logs(self, well_name: str, logs_dir: str):
        """
        Load well log data for display along well trajectory.

        Args:
            well_name: Name of the well
            logs_dir: Directory containing LAS files
        """
        try:
            import lasio
            las_path = Path(logs_dir) / f"{well_name}.las"
            if las_path.exists():
                las = lasio.read(str(las_path))
                self.well_logs[well_name] = {
                    'depth': las.index,
                    'gr': las['GR'] if 'GR' in las.keys() else None,
                    'data': las
                }
                print(f"3D Viewer: Loaded logs for {well_name}")
        except ImportError:
            print("3D Viewer: lasio not available for well log loading")
        except Exception as e:
            print(f"3D Viewer: Error loading logs for {well_name}: {e}")

    def set_seismic_data(self, data: np.ndarray, geometry: Dict):
        """
        Set seismic volume data.

        Args:
            data: 3D numpy array (inline, crossline, samples)
            geometry: Dict with il_min, il_max, xl_min, xl_max, sample_rate_ms
        """
        self.seismic_data = data
        self.geometry = geometry
        print(f"3D Viewer: Loaded seismic data {data.shape}")

    def set_fault_data(self, fault_probability: np.ndarray, threshold: float = 0.5):
        """
        Set fault probability volume.

        Args:
            fault_probability: 3D numpy array of fault probabilities (0-1)
            threshold: Threshold for fault surface extraction
        """
        self.fault_data = fault_probability
        self.fault_threshold = threshold
        print(f"3D Viewer: Loaded fault data {fault_probability.shape}")

    def render(self):
        """Render the 3D scene."""
        if self.seismic_data is None:
            return

        if self.viewer_type == "pyvista":
            self._render_pyvista()
        elif self.viewer_type == "matplotlib":
            self._render_matplotlib()

    def _render_pyvista(self):
        """Render using PyVista (off-screen, display as image)."""
        if not PYVISTA_AVAILABLE or self.seismic_data is None:
            return

        # Downsample for performance
        ds = self.downsample_var.get()
        data = self.seismic_data[::ds, ::ds, ::ds]

        # Create plotter
        plotter = pv.Plotter(off_screen=True, window_size=(800, 600))
        plotter.set_background(self.colors['bg'])

        # Add seismic volume
        if self.show_seismic_var.get():
            # Create ImageData (uniform grid)
            grid = pv.ImageData()
            grid.dimensions = np.array(data.shape) + 1
            grid.spacing = (ds, ds, ds)
            grid.cell_data["amplitude"] = data.flatten(order="F")

            # Add as volume with opacity
            opacity = self.opacity_var.get()
            plotter.add_volume(
                grid,
                scalars="amplitude",
                cmap="seismic",
                opacity="sigmoid_5",
                shade=False
            )

        # Add fault surfaces
        if self.show_faults_var.get() and self.fault_data is not None:
            fault_ds = self.fault_data[::ds, ::ds, ::ds]
            fault_surface = self._extract_fault_surface_pyvista(fault_ds)
            if fault_surface is not None and fault_surface.n_points > 0:
                plotter.add_mesh(
                    fault_surface,
                    color='yellow',
                    opacity=0.8,
                    show_edges=False
                )

        # Add wells (Petrel-style)
        if self.show_wells_var.get() and self.wells:
            self._render_wells_pyvista(plotter, data.shape, ds)

        # Add axes
        plotter.add_axes()

        # Render to image
        plotter.show(auto_close=False)
        img = plotter.screenshot(return_img=True)
        plotter.close()

        # Display in Tkinter
        self._display_image(img)

    def _extract_fault_surface_pyvista(self, fault_prob: np.ndarray):
        """Extract isosurface from fault probability volume using PyVista."""
        if not PYVISTA_AVAILABLE:
            return None

        try:
            # Create grid
            grid = pv.ImageData()
            grid.dimensions = np.array(fault_prob.shape) + 1
            grid.spacing = (1, 1, 1)
            grid.cell_data["probability"] = fault_prob.flatten(order="F")

            # Extract isosurface
            surface = grid.contour(
                isosurfaces=[self.fault_threshold],
                scalars="probability"
            )
            return surface
        except Exception as e:
            print(f"Fault surface extraction error: {e}")
            return None

    def _render_wells_pyvista(self, plotter, data_shape: tuple, downsample: int):
        """
        Render wells using PyVista (3D tubes with labels).

        Args:
            plotter: PyVista plotter instance
            data_shape: Shape of downsampled seismic data
            downsample: Downsample factor
        """
        if not PYVISTA_AVAILABLE:
            return

        well_colors = ['lime', 'cyan', 'magenta', 'orange', 'red', 'blue']
        color_idx = 0

        n_il, n_xl, n_samples = data_shape

        for well_name, well_data in self.wells.items():
            if not well_data.get('within_3d', False):
                continue

            # Get inline/crossline position
            il_approx = well_data.get('inline_approx', 0)
            xl_approx = well_data.get('xline_approx', 0)

            # Convert to volume coordinates
            il_min = self.geometry.get('il_min', 5000)
            xl_min = self.geometry.get('xl_min', 5000)

            il_pos = (il_approx - il_min) / downsample
            xl_pos = (xl_approx - xl_min) / downsample

            # Clamp to volume bounds
            il_pos = np.clip(il_pos, 0, n_il - 1)
            xl_pos = np.clip(xl_pos, 0, n_xl - 1)

            # Create well trajectory points (vertical well)
            n_points = 50
            points = np.zeros((n_points, 3))
            points[:, 0] = xl_pos  # X = crossline
            points[:, 1] = il_pos  # Y = inline
            points[:, 2] = np.linspace(0, n_samples * 0.9, n_points)  # Z = time/depth

            # Create tube for well trajectory
            try:
                well_line = pv.lines_from_points(points)
                well_tube = well_line.tube(radius=1.5)

                color = well_colors[color_idx % len(well_colors)]
                plotter.add_mesh(
                    well_tube,
                    color=color,
                    opacity=1.0,
                    label=well_name
                )

                # Add well head marker (sphere at surface)
                well_head = pv.Sphere(radius=3, center=(xl_pos, il_pos, 0))
                plotter.add_mesh(well_head, color=color)

                # Add well name label
                plotter.add_point_labels(
                    [(xl_pos, il_pos, -5)],
                    [well_name],
                    font_size=14,
                    text_color='white',
                    shape_color=color,
                    shape='rounded_rect',
                    shape_opacity=0.8
                )

                # Add depth markers
                depth_interval = n_samples // 5
                for z in range(depth_interval, n_samples, depth_interval):
                    depth_ms = z * downsample * self.geometry.get('sample_rate_ms', 4.0)
                    marker = pv.Sphere(radius=1, center=(xl_pos, il_pos, z))
                    plotter.add_mesh(marker, color='white')

                color_idx += 1

            except Exception as e:
                print(f"Error rendering well {well_name}: {e}")

    def _display_image(self, img: np.ndarray):
        """Display rendered image in Tkinter."""
        if not PIL_AVAILABLE:
            return

        pil_img = Image.fromarray(img)

        # Resize to fit frame if needed
        frame_width = self.display_frame.winfo_width()
        frame_height = self.display_frame.winfo_height()

        if frame_width > 100 and frame_height > 100:
            # Calculate aspect ratio preserving size
            img_ratio = pil_img.width / pil_img.height
            frame_ratio = frame_width / frame_height

            if img_ratio > frame_ratio:
                new_width = min(frame_width, pil_img.width)
                new_height = int(new_width / img_ratio)
            else:
                new_height = min(frame_height, pil_img.height)
                new_width = int(new_height * img_ratio)

            pil_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        tk_img = ImageTk.PhotoImage(pil_img)

        self.img_label.configure(image=tk_img)
        self.img_label.image = tk_img  # Keep reference

    def _render_matplotlib(self):
        """Render using matplotlib 3D (wireframe slices)."""
        if self.seismic_data is None:
            return

        self.ax.clear()
        self.ax.set_facecolor(self.colors['surface'])

        n_il, n_xl, n_samples = self.seismic_data.shape
        il_skip = self.il_skip_var.get() if hasattr(self, 'il_skip_var') else 10
        xl_skip = self.xl_skip_var.get() if hasattr(self, 'xl_skip_var') else 10

        # Downsample data
        ds = self.downsample_var.get()
        data = self.seismic_data[::ds, ::ds, ::ds]
        n_il_ds, n_xl_ds, n_samples_ds = data.shape

        # Normalize for coloring
        vmin, vmax = np.percentile(data, [2, 98])

        if self.show_seismic_var.get():
            # Plot selected inlines as surfaces
            for il in range(0, n_il_ds, max(1, il_skip // ds)):
                if il >= n_il_ds:
                    continue

                X = np.arange(n_xl_ds)
                Z = np.arange(n_samples_ds)
                X, Z = np.meshgrid(X, Z)
                Y = np.full_like(X, il)
                C = data[il, :, :].T

                # Normalize colors
                C_norm = np.clip((C - vmin) / (vmax - vmin + 1e-10), 0, 1)

                self.ax.plot_surface(
                    X, Y, Z,
                    facecolors=plt.cm.seismic(C_norm),
                    alpha=self.opacity_var.get() * 0.5,
                    linewidth=0,
                    antialiased=False,
                    shade=False
                )

            # Plot selected crosslines
            for xl in range(0, n_xl_ds, max(1, xl_skip // ds)):
                if xl >= n_xl_ds:
                    continue

                Y = np.arange(n_il_ds)
                Z = np.arange(n_samples_ds)
                Y, Z = np.meshgrid(Y, Z)
                X = np.full_like(Y, xl)
                C = data[:, xl, :].T

                C_norm = np.clip((C - vmin) / (vmax - vmin + 1e-10), 0, 1)

                self.ax.plot_surface(
                    X, Y, Z,
                    facecolors=plt.cm.seismic(C_norm),
                    alpha=self.opacity_var.get() * 0.5,
                    linewidth=0,
                    antialiased=False,
                    shade=False
                )

        # Plot faults if available
        if self.show_faults_var.get() and self.fault_data is not None:
            fault_ds = self.fault_data[::ds, ::ds, ::ds]
            fault_mask = fault_ds > self.fault_threshold
            fault_coords = np.where(fault_mask)

            if len(fault_coords[0]) > 0:
                # Subsample for performance
                step = max(1, len(fault_coords[0]) // 3000)
                self.ax.scatter(
                    fault_coords[1][::step],  # XL
                    fault_coords[0][::step],  # IL
                    fault_coords[2][::step],  # Sample
                    c='yellow',
                    s=2,
                    alpha=0.6,
                    label='Faults'
                )

        # Plot wells with trajectory and depth markers (Petrel-style)
        if self.show_wells_var.get() and self.wells:
            self._render_wells_matplotlib(n_samples_ds, ds)

        # Labels
        self.ax.set_xlabel('Crossline', color=self.colors['fg'], fontsize=14)
        self.ax.set_ylabel('Inline', color=self.colors['fg'], fontsize=14)
        self.ax.set_zlabel('Time (samples)', color=self.colors['fg'], fontsize=14)
        self.ax.invert_zaxis()

        # Add legend if we have items
        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            self.ax.legend(loc='upper right', fontsize=12)

        # Style axes
        self.ax.tick_params(colors=self.colors['fg'])

        self.fig.tight_layout()
        self.canvas.draw()

    def _render_wells_matplotlib(self, n_samples: int, downsample: int):
        """
        Render wells in 3D matplotlib view (Petrel-style).

        Args:
            n_samples: Number of time samples in downsampled data
            downsample: Downsample factor applied to seismic
        """
        # Well colors for different wells
        well_colors = ['lime', 'cyan', 'magenta', 'orange', 'red', 'blue']
        color_idx = 0

        for well_name, well_data in self.wells.items():
            # Only display wells within 3D volume
            if not well_data.get('within_3d', False):
                continue

            # Get inline/crossline position
            il_approx = well_data.get('inline_approx', 0)
            xl_approx = well_data.get('xline_approx', 0)

            # Convert to downsampled coordinates
            # Assume geometry starts at il_min, xl_min
            il_min = self.geometry.get('il_min', 5000)
            xl_min = self.geometry.get('xl_min', 5000)

            # Calculate relative position in volume (0 to n_il/xl)
            il_pos = (il_approx - il_min) / downsample
            xl_pos = (xl_approx - xl_min) / downsample

            # Well trajectory - vertical for now (from surface to TD)
            # In real implementation, would use deviation survey
            n_depth_points = 50
            z_trajectory = np.linspace(0, n_samples * 0.9, n_depth_points)

            # Create slight deviation for visual interest (simulated trajectory)
            il_trajectory = np.full(n_depth_points, il_pos)
            xl_trajectory = np.full(n_depth_points, xl_pos)

            # Plot well trajectory as a thick line (well stick)
            color = well_colors[color_idx % len(well_colors)]
            self.ax.plot(
                xl_trajectory, il_trajectory, z_trajectory,
                color=color, linewidth=4, label=well_name,
                solid_capstyle='round'
            )

            # Add well head marker (surface location)
            self.ax.scatter(
                [xl_pos], [il_pos], [0],
                c=color, s=150, marker='^',
                edgecolors='white', linewidths=2,
                zorder=10
            )

            # Add well name label at surface
            self.ax.text(
                xl_pos, il_pos, -5,
                well_name,
                color='white',
                fontsize=12,
                fontweight='bold',
                ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8)
            )

            # Add depth markers along well (every ~500ms TWT equivalent)
            depth_interval = n_samples // 5  # 5 depth markers
            for i, z in enumerate(range(0, n_samples, depth_interval)):
                if z > 0:
                    depth_ms = z * downsample * self.geometry.get('sample_rate_ms', 4.0)
                    # Marker point
                    self.ax.scatter(
                        [xl_pos], [il_pos], [z],
                        c='white', s=30, marker='o',
                        edgecolors=color, linewidths=1
                    )
                    # Depth label
                    self.ax.text(
                        xl_pos + 3, il_pos + 3, z,
                        f'{int(depth_ms)}ms',
                        color=self.colors['fg'],
                        fontsize=9,
                        alpha=0.8
                    )

            # Add TD marker (total depth)
            td_z = n_samples * 0.9
            self.ax.scatter(
                [xl_pos], [il_pos], [td_z],
                c=color, s=100, marker='s',
                edgecolors='white', linewidths=2
            )

            color_idx += 1

    def _reset_view(self):
        """Reset to default view."""
        if self.viewer_type == "matplotlib":
            self.ax.view_init(elev=30, azim=-60)
            self.canvas.draw()
        elif self.viewer_type == "pyvista":
            self.render()

    def _set_view(self, view: str):
        """Set predefined view angle."""
        if self.viewer_type == "matplotlib":
            views = {
                "top": (90, 0),
                "front": (0, 0),
                "side": (0, 90),
                "iso": (30, -60)
            }
            elev, azim = views.get(view, (30, -60))
            self.ax.view_init(elev=elev, azim=azim)
            self.canvas.draw()
        elif self.viewer_type == "pyvista":
            self.render()


class Overlay2D3DViewer:
    """
    Overlay viewer for combined 2D and 3D visualization.

    Shows 2D seismic lines with context from 3D volume,
    enabling comparison and quality control.
    """

    def __init__(self, parent: tk.Widget, colors: Optional[Dict] = None):
        """
        Initialize overlay viewer.

        Args:
            parent: Tkinter parent widget
            colors: Color theme dictionary
        """
        self.parent = parent
        self.colors = colors or {
            'bg': '#1a1a2e',
            'surface': '#16213e',
            'overlay': '#0f3460',
            'fg': '#e6e6e6',
            'accent': '#e94560',
            'success': '#4ecca3',
            'warning': '#ffc107'
        }

        self.seismic_3d: Optional[np.ndarray] = None
        self.seismic_2d_lines: Dict[str, np.ndarray] = {}
        self.geometry_3d: Dict = {}
        self.geometry_2d: Dict[str, Dict] = {}

        self._setup_ui()

    def _setup_ui(self):
        """Setup the overlay viewer UI."""
        if not MPL_3D_AVAILABLE:
            ttk.Label(self.parent,
                     text="Overlay viewer requires matplotlib",
                     font=('Segoe UI', 11)).pack(expand=True, pady=50)
            return

        # Control panel
        control_frame = ttk.LabelFrame(self.parent, text="2D/3D Overlay Controls")
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        row1 = ttk.Frame(control_frame)
        row1.pack(fill=tk.X, padx=5, pady=5)

        # 2D line selection
        ttk.Label(row1, text="2D Line:").pack(side=tk.LEFT, padx=5)
        self.line_var = tk.StringVar()
        self.line_combo = ttk.Combobox(
            row1, textvariable=self.line_var,
            state='readonly', width=25
        )
        self.line_combo.pack(side=tk.LEFT, padx=5)
        self.line_combo.bind('<<ComboboxSelected>>', self._on_line_selected)

        # Overlay opacity
        ttk.Label(row1, text="2D Opacity:").pack(side=tk.LEFT, padx=(20, 5))
        self.overlay_opacity = tk.DoubleVar(value=1.0)
        ttk.Scale(row1, from_=0.0, to=1.0,
                 variable=self.overlay_opacity,
                 orient=tk.HORIZONTAL, length=100,
                 command=self._update_display).pack(side=tk.LEFT, padx=5)

        # Display mode
        row2 = ttk.Frame(control_frame)
        row2.pack(fill=tk.X, padx=5, pady=5)

        self.mode_var = tk.StringVar(value="sidebyside")
        ttk.Radiobutton(row2, text="Side-by-Side",
                       variable=self.mode_var, value="sidebyside",
                       command=self._update_display).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(row2, text="Overlay",
                       variable=self.mode_var, value="overlay",
                       command=self._update_display).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(row2, text="Difference",
                       variable=self.mode_var, value="difference",
                       command=self._update_display).pack(side=tk.LEFT, padx=5)

        # 3D inline selection
        ttk.Label(row2, text="3D Inline:").pack(side=tk.LEFT, padx=(20, 5))
        self.inline_var = tk.IntVar(value=0)
        self.inline_spin = ttk.Spinbox(row2, from_=0, to=1000, width=8,
                                       textvariable=self.inline_var,
                                       command=self._update_display)
        self.inline_spin.pack(side=tk.LEFT, padx=5)

        # Canvas
        self.fig, self.axes = plt.subplots(1, 2, figsize=(14, 6),
                                           facecolor=self.colors['bg'])
        for ax in self.axes:
            ax.set_facecolor(self.colors['surface'])

        self.canvas_frame = ttk.Frame(self.parent)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.canvas = FigureCanvasTkAgg(self.fig, self.canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Toolbar
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar_frame = ttk.Frame(self.canvas_frame)
        toolbar_frame.pack(fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        # Status label
        self.status_label = ttk.Label(self.parent, text="Load 2D and 3D data to compare")
        self.status_label.pack(fill=tk.X, padx=10, pady=5)

    def set_3d_data(self, data: np.ndarray, geometry: Dict):
        """Set 3D seismic volume."""
        self.seismic_3d = data
        self.geometry_3d = geometry

        # Update inline range
        if hasattr(self, 'inline_spin'):
            n_il = data.shape[0]
            self.inline_spin.configure(from_=0, to=n_il-1)
            self.inline_var.set(n_il // 2)

        print(f"Overlay: Set 3D data {data.shape}")

    def add_2d_line(self, name: str, data: np.ndarray, geometry: Dict):
        """Add a 2D seismic line."""
        self.seismic_2d_lines[name] = data
        self.geometry_2d[name] = geometry

        # Update combo box
        if hasattr(self, 'line_combo'):
            self.line_combo['values'] = list(self.seismic_2d_lines.keys())
            if not self.line_var.get() and self.seismic_2d_lines:
                self.line_var.set(list(self.seismic_2d_lines.keys())[0])

        print(f"Overlay: Added 2D line '{name}' {data.shape}")

    def _on_line_selected(self, event=None):
        self._update_display()

    def _update_display(self, *args):
        """Update the overlay display."""
        if not MPL_3D_AVAILABLE:
            return

        line_name = self.line_var.get()
        if not line_name or line_name not in self.seismic_2d_lines:
            return

        line_2d = self.seismic_2d_lines[line_name]
        info_2d = self.geometry_2d.get(line_name, {})

        for ax in self.axes:
            ax.clear()
            ax.set_facecolor(self.colors['surface'])

        mode = self.mode_var.get()
        inline_idx = self.inline_var.get()

        # Get 3D slice if available
        slice_3d = None
        if self.seismic_3d is not None:
            if 0 <= inline_idx < self.seismic_3d.shape[0]:
                slice_3d = self.seismic_3d[inline_idx, :, :].T

        # Normalize for display
        clip_2d = np.percentile(np.abs(line_2d), 99)
        if slice_3d is not None:
            clip_3d = np.percentile(np.abs(slice_3d), 99)
        else:
            clip_3d = clip_2d

        if mode == "sidebyside":
            # Left: 2D line
            self.axes[0].imshow(line_2d.T if line_2d.ndim == 2 else line_2d,
                               cmap='seismic', aspect='auto',
                               vmin=-clip_2d, vmax=clip_2d)
            self.axes[0].set_title(f"2D Line: {line_name}",
                                  color=self.colors['fg'])
            self.axes[0].set_xlabel('Trace', color=self.colors['fg'])
            self.axes[0].set_ylabel('Time (samples)', color=self.colors['fg'])
            self.axes[0].tick_params(colors=self.colors['fg'])

            # Right: 3D inline
            if slice_3d is not None:
                self.axes[1].imshow(slice_3d, cmap='seismic', aspect='auto',
                                   vmin=-clip_3d, vmax=clip_3d)
                self.axes[1].set_title(f"3D Inline: {inline_idx}",
                                      color=self.colors['fg'])
            else:
                self.axes[1].text(0.5, 0.5, "No 3D data loaded",
                                 ha='center', va='center',
                                 transform=self.axes[1].transAxes,
                                 color=self.colors['fg'])
            self.axes[1].set_xlabel('Crossline', color=self.colors['fg'])
            self.axes[1].set_ylabel('Time (samples)', color=self.colors['fg'])
            self.axes[1].tick_params(colors=self.colors['fg'])

        elif mode == "overlay":
            if slice_3d is not None:
                # Show 3D with reduced opacity
                self.axes[0].imshow(slice_3d, cmap='seismic', aspect='auto',
                                   vmin=-clip_3d, vmax=clip_3d, alpha=0.5)
                # Overlay 2D (may need interpolation to match sizes)
                self.axes[0].imshow(line_2d.T if line_2d.ndim == 2 else line_2d,
                                   cmap='seismic', aspect='auto',
                                   vmin=-clip_2d, vmax=clip_2d,
                                   alpha=self.overlay_opacity.get())
                self.axes[0].set_title(f"Overlay: {line_name} on IL {inline_idx}",
                                      color=self.colors['fg'])
            else:
                self.axes[0].imshow(line_2d.T if line_2d.ndim == 2 else line_2d,
                                   cmap='seismic', aspect='auto',
                                   vmin=-clip_2d, vmax=clip_2d)
                self.axes[0].set_title(f"2D Line: {line_name}",
                                      color=self.colors['fg'])

            self.axes[0].set_xlabel('Trace', color=self.colors['fg'])
            self.axes[0].set_ylabel('Time (samples)', color=self.colors['fg'])
            self.axes[0].tick_params(colors=self.colors['fg'])

            # Hide second axis in overlay mode
            self.axes[1].axis('off')

        elif mode == "difference":
            if slice_3d is not None:
                # Resize 2D to match 3D if needed
                line_display = line_2d.T if line_2d.ndim == 2 else line_2d

                if line_display.shape == slice_3d.shape:
                    diff = line_display - slice_3d
                    clip_diff = np.percentile(np.abs(diff), 99)

                    self.axes[0].imshow(diff, cmap='RdBu', aspect='auto',
                                       vmin=-clip_diff, vmax=clip_diff)
                    self.axes[0].set_title(f"Difference: {line_name} - IL {inline_idx}",
                                          color=self.colors['fg'])
                else:
                    self.axes[0].text(0.5, 0.5,
                                     f"Size mismatch:\n2D: {line_display.shape}\n3D: {slice_3d.shape}",
                                     ha='center', va='center',
                                     transform=self.axes[0].transAxes,
                                     color=self.colors['fg'])
            else:
                self.axes[0].text(0.5, 0.5, "No 3D data for difference",
                                 ha='center', va='center',
                                 transform=self.axes[0].transAxes,
                                 color=self.colors['fg'])

            self.axes[0].set_xlabel('Trace', color=self.colors['fg'])
            self.axes[0].set_ylabel('Time (samples)', color=self.colors['fg'])
            self.axes[0].tick_params(colors=self.colors['fg'])
            self.axes[1].axis('off')

        self.fig.tight_layout()
        self.canvas.draw()

        # Update status
        status = f"2D: {line_name}"
        if slice_3d is not None:
            status += f" | 3D IL: {inline_idx}"
        if hasattr(self, 'status_label'):
            self.status_label.config(text=status)


# Standalone test
if __name__ == "__main__":
    root = tk.Tk()
    root.title("3D Viewer Test")
    root.geometry("1200x800")

    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True)

    # 3D tab
    frame_3d = ttk.Frame(notebook)
    notebook.add(frame_3d, text="3D Volume")
    viewer_3d = Seismic3DViewer(frame_3d)

    # Overlay tab
    frame_overlay = ttk.Frame(notebook)
    notebook.add(frame_overlay, text="2D/3D Overlay")
    overlay_viewer = Overlay2D3DViewer(frame_overlay)

    # Test with random data
    test_data = np.random.randn(100, 100, 200).astype(np.float32)
    viewer_3d.set_seismic_data(test_data, {})
    overlay_viewer.set_3d_data(test_data, {})

    root.mainloop()
