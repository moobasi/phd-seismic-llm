"""
================================================================================
UNIFIED PROJECT CONFIGURATION SYSTEM
================================================================================

Centralized configuration management for the PhD Seismic Interpretation Framework.
All modules should import and use this configuration system instead of hardcoded paths.

Usage:
    from project_config import ProjectConfig, get_config

    # Load or create project configuration
    config = get_config()

    # Access paths
    segy_3d = config.seismic_3d_path
    well_logs = config.well_logs_directory

    # Save configuration
    config.save()

Author: Moses Ekene Obasi
PhD Research - University of Calabar
================================================================================
"""

import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime
import os


# =============================================================================
# PROJECT CONFIGURATION CLASS
# =============================================================================

@dataclass
class ProjectConfig:
    """
    Unified project configuration for the PhD Seismic Interpretation Framework.

    All paths are stored as strings and validated on access.
    This configuration is saved/loaded from a JSON file.
    """

    # Project metadata
    project_name: str = "Bornu Chad Basin Interpretation"
    project_description: str = "PhD Research - LLM-Assisted Seismic Interpretation"
    created_date: str = ""
    modified_date: str = ""

    # ==========================================================================
    # INPUT DATA PATHS (User must configure these)
    # ==========================================================================

    # Seismic data
    seismic_3d_path: str = ""           # Path to 3D SEGY file or directory
    seismic_2d_directory: str = ""       # Directory containing 2D SEGY files

    # Well data
    well_logs_directory: str = ""        # Directory containing LAS files
    well_header_file: str = ""           # Excel/CSV file with well locations

    # Optional inputs
    velocity_model_path: str = ""        # Velocity model for depth conversion
    formation_tops_file: str = ""        # Formation tops data
    existing_horizons_directory: str = "" # Pre-interpreted horizons

    # ==========================================================================
    # OUTPUT CONFIGURATION
    # ==========================================================================

    output_directory: str = ""           # Main output directory

    # ==========================================================================
    # PROCESSING OPTIONS
    # ==========================================================================

    # GPU settings
    use_gpu: bool = True
    gpu_device_id: int = 0

    # Deep learning settings
    dl_batch_size: int = 1
    dl_model_path: str = ""              # Custom model path (optional)

    # Processing parameters
    sample_rate_ms: float = 4.0

    # LLM settings
    ollama_model: str = "llava"
    ollama_host: str = "http://localhost:11434"

    # ==========================================================================
    # BASIN-SPECIFIC SETTINGS (Bornu Chad Basin defaults)
    # ==========================================================================

    basin_name: str = "Bornu Chad Basin"
    formations: List[str] = field(default_factory=lambda: [
        "Chad Formation",
        "Fika Shale",
        "Gongila Formation",
        "Bima Formation",
        "Basement"
    ])

    facies_classes: List[str] = field(default_factory=lambda: [
        "Channel Sand",
        "Marine Shale",
        "Floodplain",
        "Carbonate",
        "Basement"
    ])

    # ==========================================================================
    # METHODS
    # ==========================================================================

    def __post_init__(self):
        """Initialize dates if not set."""
        if not self.created_date:
            self.created_date = datetime.now().isoformat()
        self.modified_date = datetime.now().isoformat()

        # Set default output directory relative to framework
        if not self.output_directory:
            self.output_directory = str(get_framework_dir() / "outputs")

    def validate(self) -> Dict[str, List[str]]:
        """
        Validate configuration and return any errors/warnings.

        Returns:
            Dict with 'errors' and 'warnings' lists
        """
        errors = []
        warnings = []

        # Check required paths
        if not self.seismic_3d_path:
            warnings.append("3D seismic path not configured")
        elif not Path(self.seismic_3d_path).exists():
            errors.append(f"3D seismic file not found: {self.seismic_3d_path}")

        if not self.well_logs_directory:
            warnings.append("Well logs directory not configured")
        elif not Path(self.well_logs_directory).exists():
            errors.append(f"Well logs directory not found: {self.well_logs_directory}")

        if not self.well_header_file:
            warnings.append("Well header file not configured")
        elif not Path(self.well_header_file).exists():
            errors.append(f"Well header file not found: {self.well_header_file}")

        # Check output directory
        if self.output_directory:
            output_path = Path(self.output_directory)
            if not output_path.exists():
                try:
                    output_path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    errors.append(f"Cannot create output directory: {e}")

        return {"errors": errors, "warnings": warnings}

    def get_output_subdir(self, subdir: str) -> Path:
        """Get a subdirectory under the output directory, creating if needed."""
        path = Path(self.output_directory) / subdir
        path.mkdir(parents=True, exist_ok=True)
        return path

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        self.modified_date = datetime.now().isoformat()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProjectConfig':
        """Create from dictionary."""
        # Handle any missing fields with defaults
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)

    def save(self, filepath: Optional[str] = None) -> str:
        """
        Save configuration to JSON file.

        Args:
            filepath: Optional path. If None, saves to default location.

        Returns:
            Path where config was saved
        """
        if filepath is None:
            filepath = str(get_framework_dir() / "project_config.json")

        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        return filepath

    @classmethod
    def load(cls, filepath: Optional[str] = None) -> 'ProjectConfig':
        """
        Load configuration from JSON file.

        Args:
            filepath: Optional path. If None, loads from default location.

        Returns:
            ProjectConfig instance
        """
        if filepath is None:
            filepath = str(get_framework_dir() / "project_config.json")

        if not Path(filepath).exists():
            # Return default config if file doesn't exist
            return cls()

        with open(filepath, 'r') as f:
            data = json.load(f)

        return cls.from_dict(data)


# =============================================================================
# GLOBAL CONFIGURATION INSTANCE
# =============================================================================

_global_config: Optional[ProjectConfig] = None


def get_framework_dir() -> Path:
    """Get the framework base directory."""
    return Path(__file__).parent.resolve()


def get_config() -> ProjectConfig:
    """
    Get the global project configuration.

    Loads from file on first access, then returns cached instance.
    """
    global _global_config

    if _global_config is None:
        _global_config = ProjectConfig.load()

    return _global_config


def set_config(config: ProjectConfig) -> None:
    """Set the global project configuration."""
    global _global_config
    _global_config = config


def reload_config() -> ProjectConfig:
    """Force reload configuration from file."""
    global _global_config
    _global_config = ProjectConfig.load()
    return _global_config


def create_project_config(
    seismic_3d: str = "",
    seismic_2d_dir: str = "",
    well_logs_dir: str = "",
    well_header: str = "",
    output_dir: str = "",
    project_name: str = "New Project"
) -> ProjectConfig:
    """
    Create a new project configuration with the specified paths.

    This is a convenience function for quickly setting up a new project.
    """
    config = ProjectConfig(
        project_name=project_name,
        seismic_3d_path=seismic_3d,
        seismic_2d_directory=seismic_2d_dir,
        well_logs_directory=well_logs_dir,
        well_header_file=well_header,
        output_directory=output_dir or str(get_framework_dir() / "outputs")
    )

    # Set as global config
    set_config(config)

    return config


# =============================================================================
# PATH RESOLUTION HELPERS
# =============================================================================

def resolve_path(path_str: str, base_dir: Optional[Path] = None) -> Path:
    """
    Resolve a path string to an absolute Path.

    Handles:
    - Absolute paths (returned as-is)
    - Relative paths (resolved against base_dir or framework dir)
    - Environment variables (expanded)
    """
    if not path_str:
        return Path()

    # Expand environment variables
    path_str = os.path.expandvars(path_str)
    path_str = os.path.expanduser(path_str)

    path = Path(path_str)

    if path.is_absolute():
        return path

    # Resolve relative to base_dir or framework dir
    if base_dir is None:
        base_dir = get_framework_dir()

    return (base_dir / path).resolve()


def get_well_files(config: Optional[ProjectConfig] = None) -> List[Path]:
    """Get list of LAS files from the configured well logs directory."""
    if config is None:
        config = get_config()

    if not config.well_logs_directory:
        return []

    well_dir = Path(config.well_logs_directory)
    if not well_dir.exists():
        return []

    return list(well_dir.glob("*.las")) + list(well_dir.glob("*.LAS"))


def get_2d_segy_files(config: Optional[ProjectConfig] = None) -> List[Path]:
    """Get list of 2D SEGY files from the configured directory."""
    if config is None:
        config = get_config()

    if not config.seismic_2d_directory:
        return []

    segy_dir = Path(config.seismic_2d_directory)
    if not segy_dir.exists():
        return []

    files = []
    for ext in ['*.segy', '*.sgy', '*.SEGY', '*.SGY']:
        files.extend(segy_dir.glob(ext))

    return sorted(files)


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    # Test configuration
    print("Testing Project Configuration System")
    print("=" * 50)

    # Create sample config
    config = ProjectConfig(
        project_name="Test Project",
        seismic_3d_path="E:/Seismic/3D/3D Bornu Chad",
        seismic_2d_directory="E:/Seismic/2D",
        well_logs_directory="E:/Logs/ALL_WELL",
        well_header_file="E:/WellHeader/location.xlsx",
        output_directory="E:/PHD_Framework/outputs"
    )

    # Validate
    validation = config.validate()
    print(f"\nValidation Results:")
    print(f"  Errors: {validation['errors']}")
    print(f"  Warnings: {validation['warnings']}")

    # Save
    save_path = config.save()
    print(f"\nSaved config to: {save_path}")

    # Load
    loaded = ProjectConfig.load(save_path)
    print(f"\nLoaded project: {loaded.project_name}")
    print(f"  3D Seismic: {loaded.seismic_3d_path}")
    print(f"  Well Logs: {loaded.well_logs_directory}")
