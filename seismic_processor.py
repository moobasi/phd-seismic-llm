"""
================================================================================
SEISMIC INTERPRETATION WORKFLOW v6.0
PhD Research - Bornu Chad Basin
================================================================================

Author: Moses Ekene Obasi
Institution: University of Calabar, Nigeria
Supervisor: Prof. Dominic Akam Obi

Streamlined 8-step workflow for pre-processed seismic data interpretation.
Both 2D and 3D data are already processed (STK/MIG) - this workflow focuses
on QC, calibration, and interpretation.

Usage:
    python seismic_processor.py          # Interactive menu
    python seismic_processor.py --help   # Show help
    python seismic_processor.py --step 1 # Run specific step

Workflow:
    Step 1: 3D EDA & QC          - Document data quality
    Step 2: Dead Trace Repair    - Fix 22% dead traces in 3D
    Step 3: Well Integration     - Velocity model & ties
    Step 4: Horizon Interpretation
    Step 5: Seismic Attributes
    Step 6: Seismic Inversion
    Step 7: 2D QC & Documentation
    Step 8: 2D-3D Integration
================================================================================
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any

# Base directory
BASE_DIR = Path(__file__).parent.resolve()

# Try to import rich for beautiful console output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.prompt import Prompt, Confirm
    from rich.text import Text
    from rich.layout import Layout
    from rich.live import Live
    from rich import box
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


# =============================================================================
# STEP DEFINITIONS
# =============================================================================

PROCESSING_STEPS = {
    1: {
        "name": "3D EDA & Quality Assessment",
        "description": "Document 3D seismic quality, geometry, frequency content for thesis",
        "folder": "eda",
        "script": "seismic_eda_automation.py",
        "icon": "1",
        "inputs": ["3D SEGY file"],
        "outputs": ["Quality report (JSON)", "Frequency analysis", "Amplitude statistics", "Thesis figures"]
    },
    2: {
        "name": "Dead Trace Repair",
        "description": "Interpolate the 22% dead traces identified in 3D data",
        "folder": "dead_trace",
        "script": "dead_trace_automation.py",
        "icon": "2",
        "inputs": ["3D SEGY file"],
        "outputs": ["Repaired SEGY", "Dead trace map", "QC report"]
    },
    3: {
        "name": "Well Integration",
        "description": "Build velocity model, generate synthetics, tie wells to seismic",
        "folder": "well_integration",
        "script": "well_integration_automation.py",
        "icon": "3",
        "inputs": ["10 LAS files", "Well header (coordinates)"],
        "outputs": ["Velocity model", "Petrophysics", "Synthetic ties", "Well ranking"]
    },
    4: {
        "name": "Horizon Interpretation",
        "description": "Interpret key horizons (Chad, Fika, Gongila, Bima formations)",
        "folder": "horizon_interpretation",
        "script": "horizon_interpretation_automation.py",
        "icon": "4",
        "inputs": ["Repaired 3D SEGY", "Well markers"],
        "outputs": ["Horizon surfaces", "Time structure maps", "Depth maps"]
    },
    5: {
        "name": "Seismic Attributes",
        "description": "Extract attributes for reservoir characterization and DHI",
        "folder": "horizon_attributes",
        "script": "horizon_attributes_automation.py",
        "icon": "5",
        "inputs": ["3D SEGY", "Interpreted horizons"],
        "outputs": ["Amplitude maps", "Coherence", "Spectral decomposition", "DHI analysis"]
    },
    6: {
        "name": "Seismic Inversion",
        "description": "Convert seismic to acoustic impedance and rock properties",
        "folder": "inversion",
        "script": "inversion_automation.py",
        "icon": "6",
        "inputs": ["3D SEGY", "Well logs", "Horizons"],
        "outputs": ["Acoustic impedance volume", "Porosity estimation", "Lithology prediction"]
    },
    7: {
        "name": "2D QC & Documentation",
        "description": "Document 2D line quality, processing state (STK/MIG), well proximity",
        "folder": "seismic_2d",
        "script": "seismic_2d_automation.py",
        "icon": "7",
        "inputs": ["2D SEGY files (already processed)", "Well locations"],
        "outputs": ["Line inventory", "Processing state report", "Basemap", "Well-line ties"]
    },
    8: {
        "name": "2D-3D Integration",
        "description": "Integrate 2D regional framework with 3D detailed interpretation",
        "folder": "integration_2d3d",
        "script": "integration_2d3d_automation.py",
        "icon": "8",
        "inputs": ["3D interpretation", "2D inventory", "Velocity model"],
        "outputs": ["Integration map", "Amplitude comparison", "Regional-local correlation"]
    },
    9: {
        "name": "Deep Learning Interpretation",
        "description": "AI-powered fault detection, facies classification, and LLM interpretation",
        "folder": "deep_learning",
        "script": "dl_integration.py",
        "icon": "9",
        "inputs": ["3D SEGY", "Well markers (optional)"],
        "outputs": ["Fault probability volume", "Facies classification", "LLM geological interpretation", "Integrated report"]
    }
}


# =============================================================================
# CONSOLE OUTPUT (with fallback for no rich)
# =============================================================================

def clear_screen():
    """Clear the console screen"""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header(study_area: str = ""):
    """Print the application header"""
    area_text = study_area if study_area else "PhD Research"

    if RICH_AVAILABLE:
        header_text = f"""
[bold cyan]╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   [bold white]SEISMIC PROCESSING AUTOMATION SUITE v5.0[/bold white]                               ║
║   [dim]{area_text:^68}[/dim] ║
║                                                                              ║
║   [yellow]Author:[/yellow] Moses Ekene Obasi                                               ║
║   [yellow]Institution:[/yellow] University of Calabar, Nigeria                             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝[/bold cyan]
"""
        console.print(header_text)
    else:
        print("=" * 80)
        print("   SEISMIC PROCESSING AUTOMATION SUITE v5.0")
        print(f"   {area_text}")
        print("=" * 80)
        print("   Author: Moses Ekene Obasi")
        print("   Institution: University of Calabar, Nigeria")
        print("=" * 80)
        print()


def print_menu():
    """Print the main menu"""
    if RICH_AVAILABLE:
        table = Table(
            title="[bold]Processing Steps[/bold]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta"
        )
        table.add_column("Step", style="cyan", justify="center", width=6)
        table.add_column("Name", style="white", width=35)
        table.add_column("Description", style="dim", width=40)

        for step_num, step_info in PROCESSING_STEPS.items():
            table.add_row(
                f"[{step_num}]",
                step_info["name"],
                step_info["description"]
            )

        console.print(table)
        console.print()

        # Options
        options = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
        options.add_column("Option", style="green")
        options.add_column("Description", style="white")
        options.add_row("[A]", "Run ALL steps (full pipeline)")
        options.add_row("[C]", "Configure settings")
        options.add_row("[S]", "System status & GPU check")
        options.add_row("[H]", "Help & documentation")
        options.add_row("[Q]", "Quit")

        console.print(Panel(options, title="[bold]Options[/bold]", border_style="green"))
    else:
        print("\n  PROCESSING STEPS:")
        print("  " + "-" * 50)
        for step_num, step_info in PROCESSING_STEPS.items():
            print(f"  [{step_num}] {step_info['name']}")
            print(f"      {step_info['description']}")
        print()
        print("  OPTIONS:")
        print("  [A] Run ALL steps (full pipeline)")
        print("  [C] Configure settings")
        print("  [S] System status & GPU check")
        print("  [H] Help & documentation")
        print("  [Q] Quit")
        print()


def print_step_details(step_num: int):
    """Print detailed information about a step"""
    step = PROCESSING_STEPS.get(step_num)
    if not step:
        return

    if RICH_AVAILABLE:
        console.print(f"\n[bold cyan]Step {step_num}: {step['name']}[/bold cyan]")
        console.print(f"[dim]{step['description']}[/dim]\n")

        console.print("[yellow]Required Inputs:[/yellow]")
        for inp in step["inputs"]:
            console.print(f"  - {inp}")

        console.print("\n[yellow]Outputs:[/yellow]")
        for out in step["outputs"]:
            console.print(f"  - {out}")

        console.print(f"\n[dim]Script: {step['folder']}/{step['script']}[/dim]")
    else:
        print(f"\nStep {step_num}: {step['name']}")
        print(f"{step['description']}\n")
        print("Required Inputs:")
        for inp in step["inputs"]:
            print(f"  - {inp}")
        print("\nOutputs:")
        for out in step["outputs"]:
            print(f"  - {out}")


def print_success(message: str):
    """Print a success message"""
    if RICH_AVAILABLE:
        console.print(f"[bold green]SUCCESS:[/bold green] {message}")
    else:
        print(f"SUCCESS: {message}")


def print_error(message: str):
    """Print an error message"""
    if RICH_AVAILABLE:
        console.print(f"[bold red]ERROR:[/bold red] {message}")
    else:
        print(f"ERROR: {message}")


def print_warning(message: str):
    """Print a warning message"""
    if RICH_AVAILABLE:
        console.print(f"[bold yellow]WARNING:[/bold yellow] {message}")
    else:
        print(f"WARNING: {message}")


def print_info(message: str):
    """Print an info message"""
    if RICH_AVAILABLE:
        console.print(f"[bold blue]INFO:[/bold blue] {message}")
    else:
        print(f"INFO: {message}")


# =============================================================================
# SYSTEM CHECKS
# =============================================================================

def check_dependencies() -> Dict[str, bool]:
    """Check if all required dependencies are installed"""
    dependencies = {
        "numpy": False,
        "scipy": False,
        "matplotlib": False,
        "segyio": False,
        "scikit-learn": False,
        "tqdm": False,
        "rich": RICH_AVAILABLE
    }

    for pkg in dependencies:
        try:
            if pkg == "scikit-learn":
                import sklearn
            elif pkg == "rich":
                dependencies[pkg] = RICH_AVAILABLE
                continue
            else:
                __import__(pkg)
            dependencies[pkg] = True
        except ImportError:
            dependencies[pkg] = False

    return dependencies


def check_gpu() -> Dict[str, Any]:
    """Check GPU availability"""
    gpu_info = {
        "available": False,
        "device_name": "N/A",
        "memory_gb": 0,
        "cupy_installed": False
    }

    try:
        import cupy as cp
        gpu_info["cupy_installed"] = True
        gpu_info["available"] = cp.cuda.is_available()
        if gpu_info["available"]:
            device = cp.cuda.Device()
            try:
                if hasattr(device, 'name'):
                    name = device.name
                    gpu_info["device_name"] = name.decode() if isinstance(name, bytes) else str(name)
                else:
                    gpu_info["device_name"] = f"GPU {device.id}"
            except:
                gpu_info["device_name"] = f"GPU {device.id}"
            try:
                mem = device.mem_info
                gpu_info["memory_gb"] = mem[1] / (1024**3)
            except:
                gpu_info["memory_gb"] = 0
    except:
        pass

    return gpu_info


def show_system_status():
    """Display system status"""
    clear_screen()
    print_header()

    if RICH_AVAILABLE:
        console.print("\n[bold]System Status[/bold]\n")

        # Dependencies
        deps = check_dependencies()
        dep_table = Table(title="Dependencies", box=box.ROUNDED)
        dep_table.add_column("Package", style="cyan")
        dep_table.add_column("Status", justify="center")

        for pkg, installed in deps.items():
            status = "[green]Installed[/green]" if installed else "[red]Missing[/red]"
            dep_table.add_row(pkg, status)

        console.print(dep_table)

        # GPU
        gpu = check_gpu()
        gpu_table = Table(title="GPU Acceleration", box=box.ROUNDED)
        gpu_table.add_column("Property", style="cyan")
        gpu_table.add_column("Value")

        gpu_table.add_row("CuPy Installed", "[green]Yes[/green]" if gpu["cupy_installed"] else "[red]No[/red]")
        gpu_table.add_row("GPU Available", "[green]Yes[/green]" if gpu["available"] else "[yellow]No[/yellow]")
        gpu_table.add_row("Device", gpu["device_name"])
        gpu_table.add_row("Memory", f"{gpu['memory_gb']:.1f} GB" if gpu["memory_gb"] > 0 else "N/A")

        console.print(gpu_table)

        # Processing folders
        folder_table = Table(title="Processing Modules", box=box.ROUNDED)
        folder_table.add_column("Step", style="cyan", justify="center")
        folder_table.add_column("Module", style="white")
        folder_table.add_column("Status", justify="center")

        for step_num, step_info in PROCESSING_STEPS.items():
            folder = BASE_DIR / step_info["folder"]
            script = folder / step_info["script"]

            if script.exists():
                status = "[green]Ready[/green]"
            elif folder.exists():
                status = "[yellow]Folder exists[/yellow]"
            else:
                status = "[red]Missing[/red]"

            folder_table.add_row(str(step_num), step_info["name"], status)

        console.print(folder_table)
    else:
        print("\nSystem Status\n")
        print("-" * 40)

        deps = check_dependencies()
        print("Dependencies:")
        for pkg, installed in deps.items():
            status = "OK" if installed else "MISSING"
            print(f"  {pkg}: {status}")

        gpu = check_gpu()
        print("\nGPU:")
        print(f"  CuPy: {'Installed' if gpu['cupy_installed'] else 'Not installed'}")
        print(f"  GPU: {'Available' if gpu['available'] else 'Not available'}")
        print(f"  Device: {gpu['device_name']}")

    input("\nPress Enter to continue...")


# =============================================================================
# CONFIGURATION
# =============================================================================

class PipelineConfig:
    """Pipeline configuration manager"""

    def __init__(self):
        self.config_file = BASE_DIR / "pipeline_config.json"
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        """Load configuration from file"""
        default_config = {
            "study_area": "Study Area",  # Generic name - user can customize
            "input_segy": "",
            "output_dir": str(BASE_DIR / "outputs"),
            "well_data": "",
            "use_gpu": True,
            "save_figures": True,
            "verbose": True,
            "last_step_completed": 0
        }

        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    loaded = json.load(f)
                    default_config.update(loaded)
            except:
                pass

        return default_config

    def save(self):
        """Save configuration to file"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)

    def configure_interactive(self):
        """Interactive configuration"""
        clear_screen()
        print_header(self.config.get("study_area", ""))

        if RICH_AVAILABLE:
            console.print("\n[bold]Pipeline Configuration[/bold]\n")

            # Current settings
            table = Table(title="Current Settings", box=box.ROUNDED)
            table.add_column("Setting", style="cyan")
            table.add_column("Value", style="white")

            table.add_row("Study Area Name", self.config.get("study_area", "Study Area"))
            table.add_row("Input SEGY", self.config["input_segy"] or "[red]NOT SET - REQUIRED![/red]")
            table.add_row("Output Directory", self.config["output_dir"])
            table.add_row("Well Data", self.config["well_data"] or "[dim]Not set (optional)[/dim]")
            table.add_row("Use GPU", "[green]Yes[/green]" if self.config["use_gpu"] else "[yellow]No[/yellow]")
            table.add_row("Save Figures", "[green]Yes[/green]" if self.config["save_figures"] else "[yellow]No[/yellow]")

            console.print(table)
            console.print()

            # Study area name
            if Confirm.ask("Update study area name?", default=False):
                self.config["study_area"] = Prompt.ask(
                    "Enter study area name (e.g., 'Study Area A', 'Basin X')",
                    default=self.config.get("study_area", "Study Area")
                )

            # Input SEGY - always ask if not set
            if not self.config["input_segy"] or Confirm.ask("Update input SEGY file?", default=not bool(self.config["input_segy"])):
                self.config["input_segy"] = Prompt.ask("Enter SEGY file path (full path)")

            if Confirm.ask("Update output directory?", default=False):
                self.config["output_dir"] = Prompt.ask("Enter output directory", default=self.config["output_dir"])

            if Confirm.ask("Update well data file?", default=False):
                self.config["well_data"] = Prompt.ask("Enter well data file path (or leave empty)")

            self.config["use_gpu"] = Confirm.ask("Enable GPU acceleration?", default=self.config["use_gpu"])
            self.config["save_figures"] = Confirm.ask("Save figures?", default=self.config["save_figures"])

            self.save()
            print_success("Configuration saved!")
        else:
            print("\nCurrent Configuration:")
            print(f"  Study Area: {self.config.get('study_area', 'Study Area')}")
            print(f"  Input SEGY: {self.config['input_segy'] or 'NOT SET - REQUIRED!'}")
            print(f"  Output Dir: {self.config['output_dir']}")
            print(f"  Well Data: {self.config['well_data'] or 'Not set (optional)'}")
            print(f"  Use GPU: {self.config['use_gpu']}")

            print("\nEnter new values (press Enter to keep current):")

            new_area = input(f"  Study Area [{self.config.get('study_area', 'Study Area')}]: ").strip()
            if new_area:
                self.config["study_area"] = new_area

            new_segy = input(f"  Input SEGY [{self.config['input_segy'] or 'REQUIRED'}]: ").strip()
            if new_segy:
                self.config["input_segy"] = new_segy

            new_output = input(f"  Output Dir [{self.config['output_dir']}]: ").strip()
            if new_output:
                self.config["output_dir"] = new_output

            new_well = input(f"  Well Data [{self.config['well_data']}]: ").strip()
            if new_well:
                self.config["well_data"] = new_well

            gpu_input = input(f"  Use GPU (y/n) [{'y' if self.config['use_gpu'] else 'n'}]: ").strip().lower()
            if gpu_input in ['y', 'n']:
                self.config["use_gpu"] = gpu_input == 'y'

            self.save()
            print("\nConfiguration saved!")

        input("\nPress Enter to continue...")

    def check_input_file(self) -> bool:
        """Check if input file is configured"""
        if not self.config["input_segy"]:
            return False
        return Path(self.config["input_segy"]).exists()


# =============================================================================
# STEP EXECUTION
# =============================================================================

def run_step(step_num: int, config: PipelineConfig) -> bool:
    """Run a specific processing step"""
    step = PROCESSING_STEPS.get(step_num)
    if not step:
        print_error(f"Invalid step number: {step_num}")
        return False

    script_path = BASE_DIR / step["folder"] / step["script"]

    if not script_path.exists():
        print_error(f"Script not found: {script_path}")
        return False

    clear_screen()
    print_header()

    if RICH_AVAILABLE:
        console.print(f"\n[bold cyan]Running Step {step_num}: {step['name']}[/bold cyan]")
        console.print(f"[dim]{step['description']}[/dim]\n")
        console.print("-" * 60)
    else:
        print(f"\nRunning Step {step_num}: {step['name']}")
        print(f"{step['description']}\n")
        print("-" * 60)

    # Build command
    cmd = [sys.executable, str(script_path)]

    # Add input file if configured
    if config.config["input_segy"]:
        cmd.append(config.config["input_segy"])

    # Add output directory
    cmd.extend(["--output-dir", config.config["output_dir"]])

    # Run the script
    try:
        start_time = time.time()

        # Run with real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(script_path.parent)
        )

        # Stream output
        for line in iter(process.stdout.readline, ''):
            print(line, end='')

        process.wait()
        elapsed = time.time() - start_time

        if process.returncode == 0:
            print_success(f"Step {step_num} completed in {elapsed:.1f} seconds")
            config.config["last_step_completed"] = step_num
            config.save()
            return True
        else:
            print_error(f"Step {step_num} failed with return code {process.returncode}")
            return False

    except Exception as e:
        print_error(f"Failed to run step: {e}")
        return False


def run_all_steps(config: PipelineConfig, resume: bool = True):
    """Run all processing steps in sequence"""
    clear_screen()
    print_header()

    last_completed = config.config.get("last_step_completed", 0)

    if RICH_AVAILABLE:
        console.print("\n[bold cyan]Running Full Pipeline[/bold cyan]\n")

        if resume and last_completed > 0:
            console.print(f"[yellow]Resuming from step {last_completed + 1} (steps 1-{last_completed} already completed)[/yellow]\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            task = progress.add_task("Pipeline Progress", total=len(PROCESSING_STEPS))

            for step_num in PROCESSING_STEPS:
                # Skip already completed steps if resuming
                if resume and step_num <= last_completed:
                    progress.update(task, description=f"Step {step_num}: {PROCESSING_STEPS[step_num]['name']} [SKIPPED]")
                    progress.advance(task)
                    continue

                progress.update(task, description=f"Step {step_num}: {PROCESSING_STEPS[step_num]['name']}")

                success = run_step(step_num, config)
                if not success:
                    console.print(f"\n[red]Pipeline stopped at step {step_num}[/red]")
                    break

                progress.advance(task)
    else:
        print("\nRunning Full Pipeline\n")

        if resume and last_completed > 0:
            print(f"Resuming from step {last_completed + 1} (steps 1-{last_completed} already completed)\n")

        for step_num in PROCESSING_STEPS:
            # Skip already completed steps if resuming
            if resume and step_num <= last_completed:
                print(f"[Step {step_num}/{len(PROCESSING_STEPS)}] {PROCESSING_STEPS[step_num]['name']} - SKIPPED")
                continue

            print(f"\n[Step {step_num}/{len(PROCESSING_STEPS)}] {PROCESSING_STEPS[step_num]['name']}")

            success = run_step(step_num, config)
            if not success:
                print(f"\nPipeline stopped at step {step_num}")
                break

    input("\nPress Enter to continue...")


# =============================================================================
# HELP & DOCUMENTATION
# =============================================================================

def show_help():
    """Show help and documentation"""
    clear_screen()
    print_header()

    if RICH_AVAILABLE:
        console.print("\n[bold]Help & Documentation[/bold]\n")

        help_text = """
[bold cyan]QUICK START[/bold cyan]
1. Press [C] to configure your input SEGY file
2. Press [1] to run EDA (quality analysis)
3. Run steps 2-8 in sequence, or press [A] for full pipeline

[bold cyan]STEP DESCRIPTIONS[/bold cyan]
[1] EDA         - Analyze data quality and generate statistics
[2] Dead Trace  - Find and fix bad traces
[3] Noise       - Remove unwanted noise
[4] AGC         - Correct amplitude decay
[5] Wells       - Integrate well log data
[6] Horizons    - Auto-track geological surfaces
[7] Attributes  - Extract seismic attributes
[8] Inversion   - Convert to rock properties

[bold cyan]OUTPUT FILES[/bold cyan]
All outputs are saved to the configured output directory:
- JSON files: Machine-readable results
- Figures: PNG visualizations
- Volumes: NPY/SEGY data files

[bold cyan]GPU ACCELERATION[/bold cyan]
GPU is auto-detected. For best performance:
- Install CuPy: pip install cupy-cuda12x
- NVIDIA GPU with 8GB+ memory recommended

[bold cyan]DOCUMENTATION[/bold cyan]
- PHD_AUTOMATION_MASTER_GUIDE.md  - Complete guide
- PHD_ROADMAP_AND_THESIS_GUIDE.md - PhD timeline
- Each step folder has its own README.md

[bold cyan]SUPPORT[/bold cyan]
For issues, check the documentation or contact:
Moses Ekene Obasi - University of Calabar
"""
        console.print(Panel(help_text, title="Help", border_style="cyan"))
    else:
        print("\nQUICK START")
        print("-" * 40)
        print("1. Press [C] to configure your input SEGY file")
        print("2. Press [1] to run EDA (quality analysis)")
        print("3. Run steps 2-8 in sequence, or press [A] for full pipeline")
        print()
        print("For detailed documentation, see:")
        print("- PHD_AUTOMATION_MASTER_GUIDE.md")
        print("- PHD_ROADMAP_AND_THESIS_GUIDE.md")

    input("\nPress Enter to continue...")


# =============================================================================
# MAIN MENU LOOP
# =============================================================================

def main_menu():
    """Main interactive menu loop"""
    config = PipelineConfig()

    # First run check - ask for input file if not configured
    if not config.config["input_segy"]:
        clear_screen()
        print_header(config.config.get("study_area", ""))

        if RICH_AVAILABLE:
            console.print("\n[bold yellow]FIRST TIME SETUP[/bold yellow]\n")
            console.print("Welcome! Before you begin, please configure your input file.\n")
            config.config["study_area"] = Prompt.ask(
                "Enter study area name (generic, e.g., 'Study Area A')",
                default="Study Area"
            )
            config.config["input_segy"] = Prompt.ask(
                "Enter path to your SEGY file"
            )
            config.save()
            print_success("Setup complete! You can change these settings anytime with [C].\n")
        else:
            print("\nFIRST TIME SETUP\n")
            print("Welcome! Before you begin, please configure your input file.\n")
            config.config["study_area"] = input("Enter study area name (generic): ").strip() or "Study Area"
            config.config["input_segy"] = input("Enter path to your SEGY file: ").strip()
            config.save()
            print("\nSetup complete!\n")

        input("Press Enter to continue...")

    while True:
        clear_screen()
        print_header(config.config.get("study_area", ""))
        print_menu()

        # Show warning if input file not found
        if config.config["input_segy"] and not Path(config.config["input_segy"]).exists():
            print_warning(f"Input file not found: {config.config['input_segy']}")
            print_info("Press [C] to update the file path.\n")

        if RICH_AVAILABLE:
            choice = Prompt.ask(
                "Select option",
                choices=["1", "2", "3", "4", "5", "6", "7", "8", "a", "A", "c", "C", "s", "S", "h", "H", "q", "Q"],
                default="1"
            ).upper()
        else:
            choice = input("Enter your choice: ").strip().upper()

        if choice == 'Q':
            if RICH_AVAILABLE:
                console.print("\n[cyan]Thank you for using Seismic Processing Suite![/cyan]")
            else:
                print("\nThank you for using Seismic Processing Suite!")
            break

        elif choice == 'A':
            run_all_steps(config)

        elif choice == 'C':
            config.configure_interactive()

        elif choice == 'S':
            show_system_status()

        elif choice == 'H':
            show_help()

        elif choice.isdigit():
            step_num = int(choice)
            if step_num in PROCESSING_STEPS:
                print_step_details(step_num)

                if RICH_AVAILABLE:
                    if Confirm.ask("\nRun this step?", default=True):
                        run_step(step_num, config)
                        input("\nPress Enter to continue...")
                else:
                    confirm = input("\nRun this step? (y/n): ").strip().lower()
                    if confirm == 'y':
                        run_step(step_num, config)
                        input("\nPress Enter to continue...")
            else:
                print_error("Invalid step number")
                time.sleep(1)


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Main entry point with CLI support"""
    parser = argparse.ArgumentParser(
        description="Seismic Processing Automation Suite v5.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python seismic_processor.py              # Interactive menu
    python seismic_processor.py --step 1     # Run step 1 (EDA)
    python seismic_processor.py --all        # Run full pipeline
    python seismic_processor.py --status     # Show system status
        """
    )

    parser.add_argument("--step", type=int, choices=range(1, 10),
                       help="Run specific step (1-9)")
    parser.add_argument("--all", action="store_true",
                       help="Run all steps in sequence")
    parser.add_argument("--status", action="store_true",
                       help="Show system status")
    parser.add_argument("--input", type=str,
                       help="Input SEGY file")
    parser.add_argument("--output", type=str,
                       help="Output directory")
    parser.add_argument("--no-gpu", action="store_true",
                       help="Disable GPU acceleration")

    args = parser.parse_args()

    # Load config
    config = PipelineConfig()

    # Apply CLI overrides
    if args.input:
        config.config["input_segy"] = args.input
    if args.output:
        config.config["output_dir"] = args.output
    if args.no_gpu:
        config.config["use_gpu"] = False

    # Handle CLI commands
    if args.status:
        show_system_status()
    elif args.step:
        print_header()
        run_step(args.step, config)
    elif args.all:
        run_all_steps(config)
    else:
        # Interactive mode
        main_menu()


if __name__ == "__main__":
    main()
