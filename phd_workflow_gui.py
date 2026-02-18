"""
================================================================================
PHD SEISMIC INTERPRETATION WORKFLOW - MAIN GUI APPLICATION
================================================================================

Unified graphical interface for the LLM-Assisted Seismic Interpretation Framework.
This is the main entry point for running the complete PhD workflow.

Features:
- Project configuration with file path selection
- All 9 processing steps with progress tracking
- Deep learning fault detection and facies classification
- Results visualization and export
- LLM-powered geological interpretation

Author: Moses Ekene Obasi
PhD Research - University of Calabar
================================================================================
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Dict, Optional, Callable, List
import threading
import subprocess
import sys
import json
import re
from datetime import datetime

# =============================================================================
# STATE MANAGEMENT
# =============================================================================

class ProcessingState:
    """Manages processing state persistence."""

    def __init__(self, state_file: Path = None):
        self.state_file = state_file or Path(__file__).parent / "processing_state.json"
        self.completed_steps: List[int] = []
        self.step_timestamps: Dict[str, str] = {}
        self.step_results: Dict[str, dict] = {}
        self.notes: List[str] = []
        self.last_updated: str = ""
        self.load()

    def load(self):
        """Load state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                self.completed_steps = data.get('completed_steps', [])
                self.step_timestamps = data.get('step_timestamps', {})
                self.step_results = data.get('step_results', {})
                self.notes = data.get('notes', [])
                self.last_updated = data.get('last_updated', '')
            except Exception as e:
                print(f"Warning: Could not load state: {e}")

    def save(self):
        """Save state to file."""
        self.last_updated = datetime.now().isoformat()
        data = {
            'completed_steps': self.completed_steps,
            'step_timestamps': self.step_timestamps,
            'step_results': self.step_results,
            'notes': self.notes,
            'last_updated': self.last_updated
        }
        try:
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save state: {e}")

    def mark_completed(self, step_num: int, results: dict = None):
        """Mark a step as completed."""
        if step_num not in self.completed_steps:
            self.completed_steps.append(step_num)
        self.step_timestamps[str(step_num)] = datetime.now().isoformat()
        if results:
            self.step_results[str(step_num)] = results
        self.save()

    def is_completed(self, step_num: int) -> bool:
        """Check if a step is completed."""
        return step_num in self.completed_steps

    def get_timestamp(self, step_num: int) -> Optional[str]:
        """Get completion timestamp for a step."""
        return self.step_timestamps.get(str(step_num))

    def reset_step(self, step_num: int):
        """Reset a step to run again."""
        if step_num in self.completed_steps:
            self.completed_steps.remove(step_num)
        if str(step_num) in self.step_timestamps:
            del self.step_timestamps[str(step_num)]
        if str(step_num) in self.step_results:
            del self.step_results[str(step_num)]
        self.save()

    def reset_all(self):
        """Reset all progress."""
        self.completed_steps = []
        self.step_timestamps = {}
        self.step_results = {}
        self.save()


# Import project configuration
try:
    from project_config import ProjectConfig, get_config, set_config, get_framework_dir
except ImportError:
    print("Error: project_config.py not found. Please ensure it exists in the same directory.")
    sys.exit(1)


# =============================================================================
# THEME AND STYLES
# =============================================================================

class ModernTheme:
    """Modern dark theme colors and fonts."""

    # Colors
    BG_DARK = "#1a1a2e"
    BG_SURFACE = "#16213e"
    BG_OVERLAY = "#0f3460"
    BG_INPUT = "#1f2940"

    FG_PRIMARY = "#e6e6e6"
    FG_SECONDARY = "#a0a0a0"
    FG_MUTED = "#6b7280"

    ACCENT = "#e94560"
    ACCENT_HOVER = "#ff6b6b"
    SUCCESS = "#4ecca3"
    WARNING = "#ffc107"
    ERROR = "#ff4757"
    INFO = "#3498db"

    # Fonts
    FONT_TITLE = ("Segoe UI", 16, "bold")
    FONT_HEADING = ("Segoe UI", 12, "bold")
    FONT_BODY = ("Segoe UI", 10)
    FONT_SMALL = ("Segoe UI", 9)
    FONT_MONO = ("Consolas", 9)


# =============================================================================
# CUSTOM WIDGETS
# =============================================================================

class PathSelector(tk.Frame):
    """Widget for selecting file or directory paths."""

    def __init__(self, parent, label: str, is_directory: bool = False,
                 file_types: List[tuple] = None, on_change: Callable = None):
        super().__init__(parent, bg=ModernTheme.BG_SURFACE)

        self.is_directory = is_directory
        self.file_types = file_types or [("All files", "*.*")]
        self.on_change = on_change
        self.path_var = tk.StringVar()

        # Label
        tk.Label(
            self, text=label, font=ModernTheme.FONT_BODY,
            bg=ModernTheme.BG_SURFACE, fg=ModernTheme.FG_SECONDARY,
            anchor='w'
        ).pack(fill=tk.X, pady=(0, 2))

        # Entry frame
        entry_frame = tk.Frame(self, bg=ModernTheme.BG_SURFACE)
        entry_frame.pack(fill=tk.X)

        # Entry
        self.entry = tk.Entry(
            entry_frame, textvariable=self.path_var,
            font=ModernTheme.FONT_BODY, bg=ModernTheme.BG_INPUT,
            fg=ModernTheme.FG_PRIMARY, insertbackground=ModernTheme.FG_PRIMARY,
            relief='flat', bd=0
        )
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=6, padx=(0, 5))

        # Browse button
        self.browse_btn = tk.Button(
            entry_frame, text="Browse", font=ModernTheme.FONT_SMALL,
            bg=ModernTheme.BG_OVERLAY, fg=ModernTheme.FG_PRIMARY,
            relief='flat', cursor='hand2', command=self._browse
        )
        self.browse_btn.pack(side=tk.RIGHT, ipadx=10, ipady=3)

        # Status indicator
        self.status_label = tk.Label(
            self, text="", font=ModernTheme.FONT_SMALL,
            bg=ModernTheme.BG_SURFACE, fg=ModernTheme.FG_MUTED
        )
        self.status_label.pack(fill=tk.X, pady=(2, 0))

        # Bind change event
        self.path_var.trace_add('write', self._on_path_change)

    def _browse(self):
        """Open file/directory dialog."""
        if self.is_directory:
            path = filedialog.askdirectory()
        else:
            path = filedialog.askopenfilename(filetypes=self.file_types)

        if path:
            self.path_var.set(path)

    def _on_path_change(self, *args):
        """Handle path change."""
        path = self.path_var.get()
        if path:
            exists = Path(path).exists()
            if exists:
                self.status_label.config(text="✓ Found", fg=ModernTheme.SUCCESS)
            else:
                self.status_label.config(text="✗ Not found", fg=ModernTheme.ERROR)
        else:
            self.status_label.config(text="", fg=ModernTheme.FG_MUTED)

        if self.on_change:
            self.on_change(path)

    def get(self) -> str:
        return self.path_var.get()

    def set(self, value: str):
        self.path_var.set(value)


class ProgressPanel(tk.Frame):
    """Panel showing real-time progress of running processing steps."""

    def __init__(self, parent):
        super().__init__(parent, bg=ModernTheme.BG_SURFACE)

        # Header
        header = tk.Frame(self, bg=ModernTheme.BG_SURFACE)
        header.pack(fill=tk.X, padx=15, pady=(15, 10))

        tk.Label(
            header, text="Processing Status",
            font=ModernTheme.FONT_HEADING, bg=ModernTheme.BG_SURFACE,
            fg=ModernTheme.ACCENT
        ).pack(side=tk.LEFT)

        self.status_indicator = tk.Label(
            header, text="● Idle", font=ModernTheme.FONT_SMALL,
            bg=ModernTheme.BG_SURFACE, fg=ModernTheme.FG_MUTED
        )
        self.status_indicator.pack(side=tk.RIGHT)

        # Current step info
        self.step_frame = tk.Frame(self, bg=ModernTheme.BG_OVERLAY)
        self.step_frame.pack(fill=tk.X, padx=15, pady=(0, 10))

        self.step_label = tk.Label(
            self.step_frame, text="No step running",
            font=ModernTheme.FONT_BODY, bg=ModernTheme.BG_OVERLAY,
            fg=ModernTheme.FG_SECONDARY, anchor='w'
        )
        self.step_label.pack(fill=tk.X, padx=10, pady=(10, 5))

        # Progress bar with percentage
        progress_row = tk.Frame(self.step_frame, bg=ModernTheme.BG_OVERLAY)
        progress_row.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            progress_row, variable=self.progress_var,
            mode='determinate', length=200
        )
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.progress_label = tk.Label(
            progress_row, text="0%", font=ModernTheme.FONT_SMALL,
            bg=ModernTheme.BG_OVERLAY, fg=ModernTheme.FG_PRIMARY, width=6
        )
        self.progress_label.pack(side=tk.RIGHT, padx=(10, 0))

        # Log output area
        log_label = tk.Label(
            self, text="Live Output:", font=ModernTheme.FONT_SMALL,
            bg=ModernTheme.BG_SURFACE, fg=ModernTheme.FG_SECONDARY, anchor='w'
        )
        log_label.pack(fill=tk.X, padx=15, pady=(5, 2))

        # Scrollable log text
        log_frame = tk.Frame(self, bg=ModernTheme.BG_INPUT)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))

        self.log_text = tk.Text(
            log_frame, font=ModernTheme.FONT_MONO,
            bg=ModernTheme.BG_INPUT, fg=ModernTheme.FG_PRIMARY,
            relief='flat', height=8, wrap=tk.WORD,
            state='disabled'
        )
        log_scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)

        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Configure text tags for colored output
        self.log_text.tag_configure('info', foreground=ModernTheme.FG_PRIMARY)
        self.log_text.tag_configure('success', foreground=ModernTheme.SUCCESS)
        self.log_text.tag_configure('error', foreground=ModernTheme.ERROR)
        self.log_text.tag_configure('warning', foreground=ModernTheme.WARNING)

    def set_running(self, step_num: int, step_name: str):
        """Set the panel to show a running step."""
        self.status_indicator.config(text="● Running", fg=ModernTheme.WARNING)
        self.step_label.config(
            text=f"Step {step_num}: {step_name}",
            fg=ModernTheme.FG_PRIMARY
        )
        self.progress_var.set(0)
        self.progress_label.config(text="0%")
        self.clear_log()

    def set_progress(self, percent: float):
        """Update progress percentage (0-100)."""
        self.progress_var.set(percent)
        self.progress_label.config(text=f"{int(percent)}%")

    def set_completed(self, step_num: int, step_name: str):
        """Mark step as completed."""
        self.status_indicator.config(text="● Completed", fg=ModernTheme.SUCCESS)
        self.step_label.config(
            text=f"Step {step_num}: {step_name} - Done!",
            fg=ModernTheme.SUCCESS
        )
        self.progress_var.set(100)
        self.progress_label.config(text="100%")

    def set_error(self, step_num: int, step_name: str):
        """Mark step as failed."""
        self.status_indicator.config(text="● Error", fg=ModernTheme.ERROR)
        self.step_label.config(
            text=f"Step {step_num}: {step_name} - Failed!",
            fg=ModernTheme.ERROR
        )

    def set_idle(self):
        """Reset to idle state."""
        self.status_indicator.config(text="● Idle", fg=ModernTheme.FG_MUTED)
        self.step_label.config(text="No step running", fg=ModernTheme.FG_SECONDARY)
        self.progress_var.set(0)
        self.progress_label.config(text="0%")

    def clear_log(self):
        """Clear the log output."""
        self.log_text.config(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state='disabled')

    def append_log(self, text: str, tag: str = 'info'):
        """Append text to the log output."""
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, text, tag)
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')


class StepCard(tk.Frame):
    """Card widget for a processing step."""

    def __init__(self, parent, step_num: int, title: str, description: str,
                 on_run: Callable = None):
        super().__init__(parent, bg=ModernTheme.BG_SURFACE, relief='flat', bd=1)

        self.step_num = step_num
        self.on_run = on_run
        self.status = "pending"  # pending, running, completed, error

        # Header
        header = tk.Frame(self, bg=ModernTheme.BG_SURFACE)
        header.pack(fill=tk.X, padx=15, pady=(15, 5))

        # Step number badge
        self.badge = tk.Label(
            header, text=str(step_num), font=ModernTheme.FONT_HEADING,
            bg=ModernTheme.BG_OVERLAY, fg=ModernTheme.FG_PRIMARY,
            width=3, height=1
        )
        self.badge.pack(side=tk.LEFT, padx=(0, 10))

        # Title
        tk.Label(
            header, text=title, font=ModernTheme.FONT_HEADING,
            bg=ModernTheme.BG_SURFACE, fg=ModernTheme.FG_PRIMARY
        ).pack(side=tk.LEFT)

        # Status indicator
        self.status_indicator = tk.Label(
            header, text="○", font=ModernTheme.FONT_BODY,
            bg=ModernTheme.BG_SURFACE, fg=ModernTheme.FG_MUTED
        )
        self.status_indicator.pack(side=tk.RIGHT)

        # Description
        tk.Label(
            self, text=description, font=ModernTheme.FONT_SMALL,
            bg=ModernTheme.BG_SURFACE, fg=ModernTheme.FG_SECONDARY,
            wraplength=350, justify='left'
        ).pack(fill=tk.X, padx=15, pady=(0, 10))

        # Progress bar
        self.progress = ttk.Progressbar(self, mode='indeterminate', length=200)

        # Run button
        self.run_btn = tk.Button(
            self, text="Run Step", font=ModernTheme.FONT_BODY,
            bg=ModernTheme.ACCENT, fg='white', relief='flat',
            cursor='hand2', command=self._run
        )
        self.run_btn.pack(pady=(0, 15), ipadx=20, ipady=5)

    def _run(self):
        if self.on_run:
            self.on_run(self.step_num)

    def set_status(self, status: str):
        """Set step status: pending, running, completed, error"""
        self.status = status

        if status == "pending":
            self.status_indicator.config(text="○", fg=ModernTheme.FG_MUTED)
            self.badge.config(bg=ModernTheme.BG_OVERLAY)
            self.run_btn.config(state='normal')
            self.progress.pack_forget()
        elif status == "running":
            self.status_indicator.config(text="◉", fg=ModernTheme.WARNING)
            self.badge.config(bg=ModernTheme.WARNING)
            self.run_btn.config(state='disabled')
            self.progress.pack(pady=(0, 10))
            self.progress.start(10)
        elif status == "completed":
            self.status_indicator.config(text="✓", fg=ModernTheme.SUCCESS)
            self.badge.config(bg=ModernTheme.SUCCESS)
            self.run_btn.config(state='normal')
            self.progress.stop()
            self.progress.pack_forget()
        elif status == "error":
            self.status_indicator.config(text="✗", fg=ModernTheme.ERROR)
            self.badge.config(bg=ModernTheme.ERROR)
            self.run_btn.config(state='normal')
            self.progress.stop()
            self.progress.pack_forget()


# =============================================================================
# MAIN APPLICATION
# =============================================================================

class PHDWorkflowApp:
    """Main application window."""

    # Processing steps definition
    STEPS = [
        {
            "num": 1, "name": "Exploratory Data Analysis",
            "desc": "Statistical analysis and quality assessment of seismic data",
            "module": "eda.seismic_eda_automation"
        },
        {
            "num": 2, "name": "Dead Trace Detection",
            "desc": "Identify and handle dead/bad traces in the data",
            "module": "dead_trace.dead_trace_automation"
        },
        {
            "num": 3, "name": "Well Log Integration",
            "desc": "Process well logs and compute petrophysical properties",
            "module": "well_integration.well_integration_automation"
        },
        {
            "num": 4, "name": "Horizon Interpretation",
            "desc": "Auto-track and interpret seismic horizons",
            "module": "horizon_interpretation.horizon_interpretation_automation"
        },
        {
            "num": 5, "name": "Horizon Attributes",
            "desc": "Extract seismic attributes along horizons",
            "module": "horizon_attributes.horizon_attributes_automation"
        },
        {
            "num": 6, "name": "Seismic Inversion",
            "desc": "Convert seismic to acoustic impedance",
            "module": "inversion.inversion_automation"
        },
        {
            "num": 7, "name": "2D Seismic Processing",
            "desc": "Process and analyze 2D seismic lines",
            "module": "seismic_2d.seismic_2d_automation"
        },
        {
            "num": 8, "name": "2D-3D Integration",
            "desc": "Integrate 2D and 3D interpretation results",
            "module": "integration_2d3d.integration_2d3d_automation"
        },
        {
            "num": 9, "name": "Deep Learning Interpretation",
            "desc": "AI-powered fault detection and facies classification",
            "module": "deep_learning.dl_integration"
        },
    ]

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("PhD Seismic Interpretation Workflow")
        self.root.geometry("1400x900")
        self.root.configure(bg=ModernTheme.BG_DARK)

        # Load configuration
        self.config = get_config()

        # Load processing state
        self.state = ProcessingState()

        # Track step cards
        self.step_cards: Dict[int, StepCard] = {}

        # Build UI
        self._create_menu()
        self._create_main_layout()

        # Load saved paths
        self._load_config_to_ui()

        # Load saved progress
        self._load_saved_progress()

    def _create_menu(self):
        """Create menu bar."""
        menubar = tk.Menu(self.root)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="New Project", command=self._new_project)
        file_menu.add_command(label="Open Project...", command=self._open_project)
        file_menu.add_command(label="Save Project", command=self._save_project)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        tools_menu.add_command(label="Seismic Viewer", command=self._open_viewer)
        tools_menu.add_command(label="AI Assistant", command=self._open_ai_assistant)
        tools_menu.add_separator()
        tools_menu.add_command(label="Run All Steps", command=self._run_all_steps)
        tools_menu.add_command(label="Run Remaining Steps", command=self._run_remaining_steps)
        tools_menu.add_separator()
        tools_menu.add_command(label="Reset All Progress", command=self._reset_all_progress)
        menubar.add_cascade(label="Tools", menu=tools_menu)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Documentation", command=self._show_docs)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menubar)

    def _create_main_layout(self):
        """Create main application layout."""
        # Main container
        main_container = tk.Frame(self.root, bg=ModernTheme.BG_DARK)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel - Configuration (400px)
        left_panel = tk.Frame(main_container, bg=ModernTheme.BG_SURFACE, width=400)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)

        self._create_config_panel(left_panel)

        # Right panel - Workflow steps
        right_panel = tk.Frame(main_container, bg=ModernTheme.BG_DARK)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._create_workflow_panel(right_panel)

    def _create_config_panel(self, parent):
        """Create configuration panel."""
        # Title
        title_frame = tk.Frame(parent, bg=ModernTheme.BG_SURFACE)
        title_frame.pack(fill=tk.X, padx=15, pady=15)

        tk.Label(
            title_frame, text="Project Configuration",
            font=ModernTheme.FONT_TITLE, bg=ModernTheme.BG_SURFACE,
            fg=ModernTheme.ACCENT
        ).pack(anchor='w')

        tk.Label(
            title_frame, text="Configure input data paths",
            font=ModernTheme.FONT_SMALL, bg=ModernTheme.BG_SURFACE,
            fg=ModernTheme.FG_SECONDARY
        ).pack(anchor='w')

        # Scrollable frame for path selectors
        canvas = tk.Canvas(parent, bg=ModernTheme.BG_SURFACE, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=ModernTheme.BG_SURFACE)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Path selectors
        self.path_selectors = {}

        # 3D Seismic
        self.path_selectors['seismic_3d'] = PathSelector(
            scrollable_frame, "3D Seismic Volume (SEGY):",
            is_directory=False,
            file_types=[("SEGY files", "*.segy *.sgy *.SEGY *.SGY"), ("All files", "*.*")],
            on_change=lambda p: self._update_config('seismic_3d_path', p)
        )
        self.path_selectors['seismic_3d'].pack(fill=tk.X, padx=15, pady=10)

        # 2D Seismic Directory
        self.path_selectors['seismic_2d'] = PathSelector(
            scrollable_frame, "2D Seismic Directory:",
            is_directory=True,
            on_change=lambda p: self._update_config('seismic_2d_directory', p)
        )
        self.path_selectors['seismic_2d'].pack(fill=tk.X, padx=15, pady=10)

        # Well Logs Directory
        self.path_selectors['well_logs'] = PathSelector(
            scrollable_frame, "Well Logs Directory (LAS files):",
            is_directory=True,
            on_change=lambda p: self._update_config('well_logs_directory', p)
        )
        self.path_selectors['well_logs'].pack(fill=tk.X, padx=15, pady=10)

        # Well Header File
        self.path_selectors['well_header'] = PathSelector(
            scrollable_frame, "Well Header File (Excel/CSV):",
            is_directory=False,
            file_types=[("Excel files", "*.xlsx *.xls"), ("CSV files", "*.csv"), ("All files", "*.*")],
            on_change=lambda p: self._update_config('well_header_file', p)
        )
        self.path_selectors['well_header'].pack(fill=tk.X, padx=15, pady=10)

        # Output Directory
        self.path_selectors['output'] = PathSelector(
            scrollable_frame, "Output Directory:",
            is_directory=True,
            on_change=lambda p: self._update_config('output_directory', p)
        )
        self.path_selectors['output'].pack(fill=tk.X, padx=15, pady=10)

        # Separator
        ttk.Separator(scrollable_frame, orient='horizontal').pack(fill=tk.X, padx=15, pady=15)

        # Project name
        tk.Label(
            scrollable_frame, text="Project Name:",
            font=ModernTheme.FONT_BODY, bg=ModernTheme.BG_SURFACE,
            fg=ModernTheme.FG_SECONDARY
        ).pack(fill=tk.X, padx=15)

        self.project_name_var = tk.StringVar(value=self.config.project_name)
        self.project_name_entry = tk.Entry(
            scrollable_frame, textvariable=self.project_name_var,
            font=ModernTheme.FONT_BODY, bg=ModernTheme.BG_INPUT,
            fg=ModernTheme.FG_PRIMARY, insertbackground=ModernTheme.FG_PRIMARY,
            relief='flat'
        )
        self.project_name_entry.pack(fill=tk.X, padx=15, pady=(5, 15), ipady=6)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Bottom buttons
        btn_frame = tk.Frame(parent, bg=ModernTheme.BG_SURFACE)
        btn_frame.pack(fill=tk.X, padx=15, pady=15, side=tk.BOTTOM)

        tk.Button(
            btn_frame, text="Validate Configuration",
            font=ModernTheme.FONT_BODY, bg=ModernTheme.INFO,
            fg='white', relief='flat', cursor='hand2',
            command=self._validate_config
        ).pack(fill=tk.X, pady=(0, 5), ipady=8)

        tk.Button(
            btn_frame, text="Save Configuration",
            font=ModernTheme.FONT_BODY, bg=ModernTheme.SUCCESS,
            fg='white', relief='flat', cursor='hand2',
            command=self._save_project
        ).pack(fill=tk.X, ipady=8)

    def _create_workflow_panel(self, parent):
        """Create workflow steps panel with progress display."""
        # Split into top (steps) and bottom (progress) sections
        top_frame = tk.Frame(parent, bg=ModernTheme.BG_DARK)
        top_frame.pack(fill=tk.BOTH, expand=True)

        bottom_frame = tk.Frame(parent, bg=ModernTheme.BG_SURFACE, height=250)
        bottom_frame.pack(fill=tk.X, pady=(10, 0))
        bottom_frame.pack_propagate(False)

        # ===== TOP: Step Cards =====
        # Title
        title_frame = tk.Frame(top_frame, bg=ModernTheme.BG_DARK)
        title_frame.pack(fill=tk.X, pady=(0, 15))

        tk.Label(
            title_frame, text="Processing Workflow",
            font=ModernTheme.FONT_TITLE, bg=ModernTheme.BG_DARK,
            fg=ModernTheme.FG_PRIMARY
        ).pack(side=tk.LEFT)

        tk.Button(
            title_frame, text="Run All Steps",
            font=ModernTheme.FONT_BODY, bg=ModernTheme.ACCENT,
            fg='white', relief='flat', cursor='hand2',
            command=self._run_all_steps
        ).pack(side=tk.RIGHT, ipadx=15, ipady=5)

        # Scrollable frame for step cards
        canvas = tk.Canvas(top_frame, bg=ModernTheme.BG_DARK, highlightthickness=0)
        scrollbar = ttk.Scrollbar(top_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=ModernTheme.BG_DARK)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Create step cards in a grid (3 columns)
        for i, step in enumerate(self.STEPS):
            row = i // 3
            col = i % 3

            card = StepCard(
                scrollable_frame, step["num"], step["name"], step["desc"],
                on_run=self._run_step
            )
            card.grid(row=row, column=col, padx=10, pady=10, sticky='nsew')
            self.step_cards[step["num"]] = card

        # Configure grid weights
        for i in range(3):
            scrollable_frame.columnconfigure(i, weight=1)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # ===== BOTTOM: Progress Panel =====
        self.progress_panel = ProgressPanel(bottom_frame)
        self.progress_panel.pack(fill=tk.BOTH, expand=True)

    # =========================================================================
    # Configuration methods
    # =========================================================================

    def _update_config(self, key: str, value: str):
        """Update configuration value."""
        setattr(self.config, key, value)

    def _load_config_to_ui(self):
        """Load configuration values to UI."""
        self.path_selectors['seismic_3d'].set(self.config.seismic_3d_path)
        self.path_selectors['seismic_2d'].set(self.config.seismic_2d_directory)
        self.path_selectors['well_logs'].set(self.config.well_logs_directory)
        self.path_selectors['well_header'].set(self.config.well_header_file)
        self.path_selectors['output'].set(self.config.output_directory)
        self.project_name_var.set(self.config.project_name)

    def _load_saved_progress(self):
        """Load and display saved progress."""
        for step_num, card in self.step_cards.items():
            if self.state.is_completed(step_num):
                card.set_status("completed")
                timestamp = self.state.get_timestamp(step_num)
                if timestamp:
                    # Show completion time in tooltip or label
                    try:
                        dt = datetime.fromisoformat(timestamp)
                        card.run_btn.config(text=f"Re-run (Done {dt.strftime('%m/%d')})")
                    except:
                        card.run_btn.config(text="Re-run")
            else:
                card.set_status("pending")

        # Show summary
        completed_count = len(self.state.completed_steps)
        if completed_count > 0:
            self.root.title(f"PhD Seismic Workflow - {completed_count}/9 steps completed")

    def _validate_config(self):
        """Validate current configuration."""
        self.config.project_name = self.project_name_var.get()
        validation = self.config.validate()

        msg = ""
        if validation['errors']:
            msg += "ERRORS:\n" + "\n".join(f"  • {e}" for e in validation['errors'])
        if validation['warnings']:
            if msg:
                msg += "\n\n"
            msg += "WARNINGS:\n" + "\n".join(f"  • {w}" for w in validation['warnings'])

        if not validation['errors'] and not validation['warnings']:
            messagebox.showinfo("Validation", "Configuration is valid! Ready to process.")
        elif validation['errors']:
            messagebox.showerror("Validation Failed", msg)
        else:
            messagebox.showwarning("Validation", msg)

    def _save_project(self):
        """Save project configuration."""
        self.config.project_name = self.project_name_var.get()
        filepath = self.config.save()
        messagebox.showinfo("Saved", f"Configuration saved to:\n{filepath}")

    def _new_project(self):
        """Create new project."""
        if messagebox.askyesno("New Project", "Create a new project? Current configuration will be cleared."):
            self.config = ProjectConfig()
            set_config(self.config)
            self._load_config_to_ui()

    def _open_project(self):
        """Open existing project configuration."""
        filepath = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=get_framework_dir()
        )
        if filepath:
            self.config = ProjectConfig.load(filepath)
            set_config(self.config)
            self._load_config_to_ui()
            messagebox.showinfo("Loaded", f"Project loaded from:\n{filepath}")

    # =========================================================================
    # Processing methods
    # =========================================================================

    def _run_step(self, step_num: int):
        """Run a specific processing step with real-time output streaming."""
        # Validate config first
        validation = self.config.validate()
        if validation['errors']:
            messagebox.showerror(
                "Configuration Error",
                "Please fix configuration errors before running:\n" +
                "\n".join(f"  • {e}" for e in validation['errors'])
            )
            return

        # Save config before running
        self.config.save()

        # Get step info
        step_info = next((s for s in self.STEPS if s['num'] == step_num), None)
        if not step_info:
            return

        # Update UI
        card = self.step_cards[step_num]
        card.set_status("running")

        # Update progress panel
        self.progress_panel.set_running(step_num, step_info['name'])

        # Run in background thread
        def run():
            try:
                # Build command based on step-specific requirements
                module = step_info['module']
                cmd = [sys.executable, "-m", module]

                # Step-specific argument handling (matching each script's argparse)
                config_path = str(get_framework_dir() / "project_config.json")
                output_dir = self.config.output_directory

                if step_num == 1:  # EDA: segy_file (positional), -c, -o
                    if self.config.seismic_3d_path:
                        cmd.append(self.config.seismic_3d_path)
                    cmd.extend(["-c", config_path, "-o", output_dir])

                elif step_num == 2:  # Dead Trace: input_segy (positional), -c, --output-dir
                    if self.config.seismic_3d_path:
                        cmd.append(self.config.seismic_3d_path)
                    cmd.extend(["-c", config_path, "--output-dir", output_dir])

                elif step_num == 3:  # Well Integration: las_directory (positional), -c, -o
                    if self.config.well_logs_directory:
                        cmd.append(self.config.well_logs_directory)
                    cmd.extend(["-c", config_path, "-o", output_dir])

                elif step_num == 4:  # Horizon: seismic_file (positional), -c, -o
                    if self.config.seismic_3d_path:
                        cmd.append(self.config.seismic_3d_path)
                    cmd.extend(["-c", config_path, "-o", output_dir])

                elif step_num == 5:  # Attributes: base_dir (positional), -c, -o
                    # Uses outputs from previous steps as base_dir
                    cmd.append(output_dir)
                    cmd.extend(["-c", config_path, "-o", output_dir])

                elif step_num == 6:  # Inversion: seismic_file (positional), -c, -o
                    if self.config.seismic_3d_path:
                        cmd.append(self.config.seismic_3d_path)
                    cmd.extend(["-c", config_path, "-o", output_dir])

                elif step_num == 7:  # 2D: input_directory (positional), -c, -o
                    if self.config.seismic_2d_directory:
                        cmd.append(self.config.seismic_2d_directory)
                    cmd.extend(["-c", config_path, "-o", output_dir])

                elif step_num == 8:  # 2D-3D Integration: -c only (no positional)
                    cmd.extend(["-c", config_path])

                elif step_num == 9:  # Deep Learning: segy_file (positional), --output-dir
                    if self.config.seismic_3d_path:
                        cmd.append(self.config.seismic_3d_path)
                    cmd.extend(["--output-dir", output_dir])

                # Run process with real-time output streaming
                process = subprocess.Popen(
                    cmd,
                    cwd=str(get_framework_dir()),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )

                # Stream output to progress panel
                line_count = 0
                progress_estimate = 0
                output_lines = []

                for line in iter(process.stdout.readline, ''):
                    if not line:
                        break

                    line_count += 1
                    output_lines.append(line)

                    # Determine line type for coloring
                    line_lower = line.lower()
                    if 'error' in line_lower or 'failed' in line_lower:
                        tag = 'error'
                    elif 'warning' in line_lower:
                        tag = 'warning'
                    elif 'complete' in line_lower or 'success' in line_lower or 'done' in line_lower:
                        tag = 'success'
                    else:
                        tag = 'info'

                    # Update progress panel in main thread
                    self.root.after(0, lambda l=line, t=tag: self.progress_panel.append_log(l, t))

                    # Estimate progress based on output patterns
                    # Look for percentage patterns like "50%" or "Processing 5/10"
                    percent_match = re.search(r'(\d+)%', line)
                    if percent_match:
                        progress_estimate = int(percent_match.group(1))
                        self.root.after(0, lambda p=progress_estimate: self.progress_panel.set_progress(p))
                    else:
                        # Estimate based on line count (rough estimate)
                        estimated_total_lines = 100  # Rough estimate
                        progress_estimate = min(95, (line_count / estimated_total_lines) * 100)
                        self.root.after(0, lambda p=progress_estimate: self.progress_panel.set_progress(p))

                process.wait()
                return_code = process.returncode

                # Update UI based on result
                if return_code == 0:
                    # Mark step as completed and save state
                    self.state.mark_completed(step_num)
                    self.root.after(0, lambda: card.set_status("completed"))
                    self.root.after(0, lambda: card.run_btn.config(text="Re-run"))
                    self.root.after(0, lambda: self.progress_panel.set_completed(step_num, step_info['name']))
                    self.root.after(0, self._update_title)
                    self.root.after(0, lambda: self.progress_panel.append_log(
                        f"\n✓ Step {step_num} completed successfully!\n", 'success'
                    ))
                else:
                    self.root.after(0, lambda: card.set_status("error"))
                    self.root.after(0, lambda: self.progress_panel.set_error(step_num, step_info['name']))
                    self.root.after(0, lambda: self.progress_panel.append_log(
                        f"\n✗ Step {step_num} failed with return code {return_code}\n", 'error'
                    ))
                    # Show error message
                    error_output = ''.join(output_lines[-20:])  # Last 20 lines
                    self.root.after(0, lambda: messagebox.showerror(
                        f"Step {step_num} Error",
                        f"Error running step:\n{error_output[:500]}"
                    ))

            except Exception as e:
                self.root.after(0, lambda: card.set_status("error"))
                self.root.after(0, lambda: self.progress_panel.set_error(step_num, step_info['name']))
                self.root.after(0, lambda: self.progress_panel.append_log(f"\n✗ Exception: {e}\n", 'error'))
                self.root.after(0, lambda: messagebox.showerror(
                    f"Step {step_num} Error", str(e)
                ))

        thread = threading.Thread(target=run, daemon=True)
        thread.start()

    def _update_title(self):
        """Update window title with progress."""
        completed_count = len(self.state.completed_steps)
        self.root.title(f"PhD Seismic Workflow - {completed_count}/9 steps completed")

    def _run_all_steps(self):
        """Run all processing steps sequentially."""
        if messagebox.askyesno("Run All Steps",
            "Run all 9 processing steps?\n\n"
            "This will re-run ALL steps including completed ones.\n"
            "Use 'Run Remaining Steps' to skip completed steps."):
            self._run_steps_sequential(list(range(1, 10)))

    def _run_remaining_steps(self):
        """Run only steps that haven't been completed."""
        remaining = [s for s in range(1, 10) if not self.state.is_completed(s)]
        if not remaining:
            messagebox.showinfo("All Done", "All steps have been completed!")
            return
        if messagebox.askyesno("Run Remaining Steps",
            f"Run {len(remaining)} remaining steps?\n\n"
            f"Steps to run: {', '.join(map(str, remaining))}"):
            self._run_steps_sequential(remaining)

    def _run_steps_sequential(self, steps: List[int]):
        """Run a list of steps sequentially."""
        def run_next(step_index=0):
            if step_index >= len(steps):
                messagebox.showinfo("Complete", "All requested steps have been processed.")
                return

            step_num = steps[step_index]
            self._run_step(step_num)

            # Schedule next step check (we'll improve this with proper completion callback)
            # For now, wait for the step to complete before proceeding
            def check_and_continue():
                card = self.step_cards[step_num]
                if card.status == "completed":
                    self.root.after(1000, lambda: run_next(step_index + 1))
                elif card.status == "error":
                    if messagebox.askyesno("Step Failed",
                        f"Step {step_num} failed. Continue with remaining steps?"):
                        self.root.after(500, lambda: run_next(step_index + 1))
                else:
                    # Still running, check again
                    self.root.after(2000, check_and_continue)

            self.root.after(3000, check_and_continue)

        run_next()

    def _reset_all_progress(self):
        """Reset all processing progress."""
        if messagebox.askyesno("Reset Progress",
            "Reset all processing progress?\n\n"
            "This will mark all steps as not completed.\n"
            "Your output files will NOT be deleted."):
            self.state.reset_all()
            self._load_saved_progress()
            messagebox.showinfo("Reset", "All progress has been reset.")

    # =========================================================================
    # Tool launchers
    # =========================================================================

    def _open_viewer(self):
        """Open seismic viewer with configured data paths."""
        try:
            # Build command with config paths for auto-loading
            cmd = [
                sys.executable,
                str(get_framework_dir() / "seismic_viewer.py")
            ]

            # Add 3D seismic path if configured
            if self.config.seismic_3d_path and Path(self.config.seismic_3d_path).exists():
                cmd.extend(["--segy-3d", self.config.seismic_3d_path])

            # Add 2D seismic directory if configured
            if self.config.seismic_2d_directory and Path(self.config.seismic_2d_directory).exists():
                cmd.extend(["--segy-2d-dir", self.config.seismic_2d_directory])

            subprocess.Popen(cmd)
        except Exception as e:
            messagebox.showerror("Error", f"Could not open viewer: {e}")

    def _open_ai_assistant(self):
        """Open AI assistant (v3 - unified agent interface)."""
        try:
            # Try v3 first (new unified interface)
            v3_path = get_framework_dir() / "seismic_ai_assistant_v3.py"
            if v3_path.exists():
                subprocess.Popen([sys.executable, str(v3_path)])
            else:
                # Fall back to v2
                subprocess.Popen([
                    sys.executable,
                    str(get_framework_dir() / "seismic_ai_assistant_v2.py")
                ])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open AI assistant: {e}")

    def _show_docs(self):
        """Show documentation."""
        readme_path = get_framework_dir() / "README.md"
        if readme_path.exists():
            import webbrowser
            webbrowser.open(str(readme_path))
        else:
            messagebox.showinfo("Documentation", "See README.md for documentation.")

    def _show_about(self):
        """Show about dialog."""
        messagebox.showinfo(
            "About",
            "PhD Seismic Interpretation Workflow\n\n"
            "LLM-Assisted Seismic Interpretation Framework\n"
            "Bornu Chad Basin, Nigeria\n\n"
            "Author: Moses Ekene Obasi\n"
            "University of Calabar\n\n"
            "Supervisor: Prof. Dominic Akam Obi"
        )

    def run(self):
        """Start the application."""
        self.root.mainloop()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point."""
    app = PHDWorkflowApp()
    app.run()


if __name__ == "__main__":
    main()
