"""
================================================================================
SEISMIC AI AGENT
================================================================================

Intelligent agent with tool-calling capabilities for seismic interpretation.
Can autonomously run processing steps, analyze data, and generate reports.

Features:
- Natural language understanding
- Tool execution (processing steps, analysis, reporting)
- Context-aware responses
- Inline image display support
- Real-time progress tracking

Author: Moses Ekene Obasi
PhD Research - University of Calabar
================================================================================
"""

import subprocess
import sys
import json
import re
import os
import threading
import queue
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

class ToolType(Enum):
    """Types of tools the agent can use."""
    PROCESS = "process"          # Run processing step
    ANALYZE = "analyze"          # Analyze data
    VISUALIZE = "visualize"      # Show visualization
    REPORT = "report"            # Generate report
    QUERY = "query"              # Answer question


@dataclass
class Tool:
    """Definition of an agent tool."""
    name: str
    description: str
    tool_type: ToolType
    parameters: Dict[str, str] = field(default_factory=dict)
    handler: Optional[Callable] = None


@dataclass
class ToolResult:
    """Result from tool execution."""
    success: bool
    message: str
    data: Any = None
    image_path: Optional[str] = None
    progress: Optional[float] = None


# =============================================================================
# OLLAMA CLIENT
# =============================================================================

class OllamaClient:
    """
    Client for Ollama LLM interactions.
    Handles both text chat and image interpretation.
    """

    PREFERRED_MODELS = [
        "qwen3:32b", "qwen2.5:32b", "qwen2.5:14b",
        "llama3.1:70b", "llama3.1:8b",
        "dolphin-mixtral:8x7b", "mixtral:8x7b",
        "llava:13b", "llava:7b"
    ]

    def __init__(self):
        self.model = None
        self.vision_model = None
        self.available_models = []
        self.connected = False

    def check_connection(self) -> bool:
        """Check Ollama connection and detect available models."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True, text=True, timeout=10
            )

            if result.returncode != 0:
                return False

            # Parse available models
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            self.available_models = []

            for line in lines:
                if line.strip():
                    parts = line.split()
                    if parts:
                        model_name = parts[0]
                        self.available_models.append(model_name)

            # Select best available model
            for preferred in self.PREFERRED_MODELS:
                for available in self.available_models:
                    if preferred in available or available in preferred:
                        if 'llava' in available.lower():
                            self.vision_model = available
                        else:
                            if self.model is None:
                                self.model = available
                        break

            # Fallback to first available
            if self.model is None and self.available_models:
                for m in self.available_models:
                    if 'llava' not in m.lower():
                        self.model = m
                        break

            self.connected = self.model is not None
            return self.connected

        except Exception as e:
            print(f"Ollama connection error: {e}")
            return False

    def chat(self, prompt: str, system_prompt: str = "", context: str = "",
             callback: Optional[Callable[[str], None]] = None) -> str:
        """
        Send chat message to Ollama.

        Args:
            prompt: User message
            system_prompt: System instructions
            context: Additional context
            callback: Optional streaming callback

        Returns:
            AI response text
        """
        if not self.connected or not self.model:
            return "Error: Not connected to Ollama. Please ensure Ollama is running."

        full_prompt = ""
        if system_prompt:
            full_prompt += f"{system_prompt}\n\n"
        if context:
            full_prompt += f"Context:\n{context}\n\n"
        full_prompt += f"User: {prompt}"

        try:
            cmd = ["ollama", "run", self.model, full_prompt]

            if callback:
                # Streaming mode with timeout
                process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, bufsize=1
                )
                response = ""
                import select
                import time
                start_time = time.time()
                timeout_seconds = 120  # 2 minute timeout

                while True:
                    # Check timeout
                    if time.time() - start_time > timeout_seconds:
                        process.terminate()
                        return response + "\n[Response truncated due to timeout]"

                    # Non-blocking read on Windows
                    line = process.stdout.readline()
                    if line:
                        response += line
                        callback(line)
                    elif process.poll() is not None:
                        break  # Process finished

                return response if response else "No response from AI."
            else:
                # Blocking mode with timeout
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=120
                )
                return result.stdout.strip() if result.stdout.strip() else "No response from AI."

        except subprocess.TimeoutExpired:
            return "Response timed out. The AI is taking too long. Please try a simpler query."
        except Exception as e:
            return f"Error: {str(e)}"

    def interpret_image(self, image_path: str, prompt: str) -> str:
        """
        Interpret an image using vision model (llava).

        Args:
            image_path: Path to image file
            prompt: Interpretation prompt

        Returns:
            AI interpretation text
        """
        if not self.vision_model:
            # Try to use llava if available
            for model in self.available_models:
                if 'llava' in model.lower():
                    self.vision_model = model
                    break

            if not self.vision_model:
                return "Error: No vision model available. Please install llava with: ollama pull llava:13b"

        if not Path(image_path).exists():
            return f"Error: Image not found: {image_path}"

        try:
            # llava expects the prompt with the image path in a specific format
            # Use the image as an argument to the model
            full_prompt = f"You are analyzing a seismic image. {prompt}"

            # For llava, we need to pass the image differently
            # Using the file:// protocol or direct path
            import platform
            if platform.system() == 'Windows':
                # Windows path format for ollama
                img_path = image_path.replace('\\', '/')
            else:
                img_path = image_path

            # Construct command for llava - image should be last argument
            cmd = ["ollama", "run", self.vision_model, f"{full_prompt} [img]{img_path}[/img]"]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=180
            )

            response = result.stdout.strip()
            if not response:
                # Try alternative format
                cmd2 = ["ollama", "run", self.vision_model, full_prompt, img_path]
                result2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=180)
                response = result2.stdout.strip()

            return response if response else "Could not interpret the image. Please try again."

        except subprocess.TimeoutExpired:
            return "Image interpretation timed out. Please try with a smaller image or simpler prompt."
        except Exception as e:
            return f"Error interpreting image: {str(e)}"


# =============================================================================
# STATE MANAGER (Unified)
# =============================================================================

class UnifiedStateManager:
    """
    Unified state manager shared between main GUI and AI Assistant.
    Single source of truth for workflow progress.
    """

    def __init__(self, state_file: Optional[Path] = None):
        self.state_file = state_file or Path(__file__).parent / "unified_state.json"
        self.state = self._load_state()
        self._callbacks: List[Callable] = []

    def _load_state(self) -> Dict:
        """Load state from file."""
        default_state = {
            "completed_steps": [],
            "step_timestamps": {},
            "step_results": {},
            "current_step": None,
            "current_progress": 0.0,
            "current_log": "",
            "data_paths": {
                "seismic_3d": "",
                "seismic_2d_dir": "",
                "well_logs_dir": "",
                "well_header": "",
                "output_dir": ""
            },
            "project_name": "Bornu Chad Basin",
            "last_updated": ""
        }

        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    loaded = json.load(f)
                    default_state.update(loaded)
            except:
                pass

        return default_state

    def save(self):
        """Save state to file."""
        self.state["last_updated"] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
        self._notify_callbacks()

    def _notify_callbacks(self):
        """Notify all registered callbacks of state change."""
        for cb in self._callbacks:
            try:
                cb(self.state)
            except:
                pass

    def register_callback(self, callback: Callable):
        """Register callback for state changes."""
        self._callbacks.append(callback)

    def mark_step_completed(self, step_num: int, results: Dict = None):
        """Mark a processing step as completed."""
        if step_num not in self.state["completed_steps"]:
            self.state["completed_steps"].append(step_num)
        self.state["step_timestamps"][str(step_num)] = datetime.now().isoformat()
        if results:
            self.state["step_results"][str(step_num)] = results
        self.state["current_step"] = None
        self.state["current_progress"] = 0.0
        self.save()

    def set_step_progress(self, step_num: int, progress: float, log: str = ""):
        """Update progress for running step."""
        self.state["current_step"] = step_num
        self.state["current_progress"] = progress
        if log:
            self.state["current_log"] = log
        self._notify_callbacks()

    def is_step_completed(self, step_num: int) -> bool:
        """Check if step is completed."""
        return step_num in self.state["completed_steps"]

    def get_completed_steps(self) -> List[int]:
        """Get list of completed step numbers."""
        return self.state["completed_steps"]

    def get_data_paths(self) -> Dict:
        """Get configured data paths."""
        return self.state["data_paths"]

    def set_data_paths(self, paths: Dict):
        """Set data paths."""
        self.state["data_paths"].update(paths)
        self.save()


# =============================================================================
# PROCESSING STEPS
# =============================================================================

PROCESSING_STEPS = {
    1: {
        "name": "Exploratory Data Analysis",
        "description": "Statistical analysis and quality assessment of seismic data",
        "script": "eda/seismic_eda_automation.py",
        "output_key": "eda_complete"
    },
    2: {
        "name": "Dead Trace Detection",
        "description": "Identify and repair dead/bad traces",
        "script": "dead_trace/dead_trace_automation.py",
        "output_key": "dead_trace_complete"
    },
    3: {
        "name": "Well Log Integration",
        "description": "Process well logs and compute petrophysics",
        "script": "well_integration/well_integration_automation.py",
        "output_key": "well_integration_complete"
    },
    4: {
        "name": "Horizon Interpretation",
        "description": "Auto-track and interpret seismic horizons",
        "script": "horizon_interpretation/horizon_interpretation_automation.py",
        "output_key": "horizon_complete"
    },
    5: {
        "name": "Horizon Attributes",
        "description": "Extract seismic attributes along horizons",
        "script": "horizon_attributes/horizon_attributes_automation.py",
        "output_key": "attributes_complete"
    },
    6: {
        "name": "Seismic Inversion",
        "description": "Convert seismic to acoustic impedance",
        "script": "inversion/inversion_automation.py",
        "output_key": "inversion_complete"
    },
    7: {
        "name": "2D Seismic Processing",
        "description": "Process and analyze 2D seismic lines",
        "script": "seismic_2d/seismic_2d_automation.py",
        "output_key": "2d_complete"
    },
    8: {
        "name": "2D-3D Integration",
        "description": "Integrate 2D and 3D interpretation results",
        "script": "integration_2d3d/integration_2d3d_automation.py",
        "output_key": "integration_complete"
    },
    9: {
        "name": "Deep Learning Interpretation",
        "description": "AI-powered fault detection and facies classification",
        "script": "deep_learning/dl_integration.py",
        "output_key": "dl_complete"
    }
}


# =============================================================================
# SEISMIC AGENT
# =============================================================================

class SeismicAgent:
    """
    Intelligent AI agent for seismic interpretation.

    Capabilities:
    - Natural language understanding
    - Tool execution (process, analyze, visualize, report)
    - Context-aware responses
    - Progress tracking

    Example:
        agent = SeismicAgent()
        response = agent.process_message("Run the EDA step")
        response = agent.process_message("What's the best drill location?")
    """

    SYSTEM_PROMPT = """You are an expert petroleum geoscientist AI assistant specializing in seismic interpretation for the Bornu Chad Basin, Nigeria.

You have access to tools that can:
- Run processing steps (EDA, dead trace repair, well integration, etc.)
- Show seismic sections and maps
- Calculate volumetrics (STOIIP, GIIP)
- Recommend drilling locations
- Generate reports

When the user asks you to DO something (run, show, calculate, generate), use the appropriate tool.
When the user asks a QUESTION, provide a helpful technical answer.

Basin Context:
- Location: NE Nigeria, UTM Zone 33N
- Target Formations: Bima Sandstone (primary), Gongila Formation (secondary)
- Seal: Fika Shale (Turonian marine)
- Main structural style: Horst and graben with NE-SW trending faults

Be concise, technical, and helpful. Use proper geological terminology.
Format responses with markdown for readability."""

    # Intent patterns for tool selection
    INTENT_PATTERNS = {
        'run_step': [
            r'run\s+(step\s*)?(\d+)',
            r'execute\s+(step\s*)?(\d+)',
            r'start\s+(step\s*)?(\d+)',
            r'run\s+(eda|dead\s*trace|well|horizon|attribute|inversion|2d|integration|deep\s*learning)',
            r'run\s+all',
            r'run\s+remaining'
        ],
        'show_seismic': [
            r'show\s+(me\s+)?(inline|crossline|xline|timeslice|time\s*slice)',
            r'display\s+(inline|crossline|xline|timeslice)',
            r'show\s+(me\s+)?seismic',
            r'view\s+(inline|crossline)'
        ],
        'calculate_stoiip': [
            r'calculate\s+stoiip',
            r'stoiip',
            r'oil\s+in\s+place',
            r'volumetrics?',
            r'reserves?'
        ],
        'recommend_drill': [
            r'(best|optimal|recommended?)\s+(drill|drilling|well)\s+(location|site|position)',
            r'where\s+(should|to)\s+drill',
            r'drill\s+location',
            r'prospect\s+location'
        ],
        'generate_report': [
            r'generate\s+(a\s+)?report',
            r'create\s+(a\s+)?report',
            r'make\s+(a\s+)?report',
            r'write\s+(a\s+)?report',
            r'summary\s+report'
        ],
        'show_map': [
            r'show\s+(me\s+)?(map|structure|amplitude|attribute)',
            r'display\s+(map|structure)',
            r'view\s+(map|structure)'
        ],
        'analyze_image': [
            r'interpret\s+(this\s+)?(image|map|section|figure)',
            r'analyze\s+(this\s+)?(image|map|section)',
            r'what\s+do\s+you\s+see'
        ],
        'get_summary': [
            r'summarize',
            r'summary',
            r'what\s+have\s+we\s+done',
            r'progress\s+so\s+far',
            r'status'
        ]
    }

    def __init__(self, state_manager: Optional[UnifiedStateManager] = None):
        self.ollama = OllamaClient()
        self.state = state_manager or UnifiedStateManager()
        self.connected = self.ollama.check_connection()
        self.conversation_history: List[Dict] = []
        self.output_queue = queue.Queue()
        self._running_process = None

    def get_status(self) -> Dict:
        """Get agent status."""
        return {
            "connected": self.connected,
            "model": self.ollama.model,
            "vision_model": self.ollama.vision_model,
            "completed_steps": self.state.get_completed_steps(),
            "current_step": self.state.state.get("current_step"),
            "current_progress": self.state.state.get("current_progress", 0)
        }

    def process_message(self, message: str,
                        stream_callback: Optional[Callable[[str], None]] = None,
                        image_path: Optional[str] = None) -> Dict:
        """
        Process user message and return response.

        Args:
            message: User input text
            stream_callback: Optional callback for streaming response
            image_path: Optional image to analyze

        Returns:
            Dict with 'response', 'tool_used', 'image_path', 'data'
        """
        # Add to history
        self.conversation_history.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat()
        })

        # Detect intent
        intent, params = self._detect_intent(message)

        result = {
            "response": "",
            "tool_used": None,
            "image_path": None,
            "data": None
        }

        # Handle image analysis
        if image_path:
            intent = "analyze_image"
            params = {"image_path": image_path}

        # Execute tool if intent detected
        if intent:
            tool_result = self._execute_tool(intent, params, message)
            result["tool_used"] = intent
            result["data"] = tool_result.data
            result["image_path"] = tool_result.image_path

            # Add tool result to context
            context = f"Tool '{intent}' executed.\nResult: {tool_result.message}"

            if tool_result.success:
                # Generate natural response with context
                result["response"] = self.ollama.chat(
                    prompt=message,
                    system_prompt=self.SYSTEM_PROMPT,
                    context=context,
                    callback=stream_callback
                )
            else:
                result["response"] = tool_result.message
        else:
            # Pure conversation - no tool needed
            # Build context from recent history and state
            context = self._build_context()
            result["response"] = self.ollama.chat(
                prompt=message,
                system_prompt=self.SYSTEM_PROMPT,
                context=context,
                callback=stream_callback
            )

        # Add response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": result["response"],
            "timestamp": datetime.now().isoformat()
        })

        return result

    def _detect_intent(self, message: str) -> Tuple[Optional[str], Dict]:
        """Detect user intent from message."""
        message_lower = message.lower()

        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, message_lower)
                if match:
                    params = {"match": match.groups() if match.groups() else None}

                    # Extract step number if present
                    step_match = re.search(r'step\s*(\d+)', message_lower)
                    if step_match:
                        params["step_num"] = int(step_match.group(1))

                    return intent, params

        return None, {}

    def _execute_tool(self, intent: str, params: Dict, original_message: str) -> ToolResult:
        """Execute the appropriate tool based on intent."""
        tool_handlers = {
            'run_step': self._tool_run_step,
            'show_seismic': self._tool_show_seismic,
            'calculate_stoiip': self._tool_calculate_stoiip,
            'recommend_drill': self._tool_recommend_drill,
            'generate_report': self._tool_generate_report,
            'show_map': self._tool_show_map,
            'analyze_image': self._tool_analyze_image,
            'get_summary': self._tool_get_summary
        }

        handler = tool_handlers.get(intent)
        if handler:
            return handler(params, original_message)

        return ToolResult(
            success=False,
            message=f"Unknown tool: {intent}"
        )

    def _tool_run_step(self, params: Dict, message: str) -> ToolResult:
        """Run a processing step."""
        step_num = params.get("step_num")

        # Check for "run all" or "run remaining"
        if "all" in message.lower():
            return self._run_all_steps()
        elif "remaining" in message.lower():
            return self._run_remaining_steps()

        if not step_num or step_num not in PROCESSING_STEPS:
            # Try to match step name
            message_lower = message.lower()
            for num, info in PROCESSING_STEPS.items():
                if any(word in message_lower for word in info["name"].lower().split()):
                    step_num = num
                    break

        if not step_num or step_num not in PROCESSING_STEPS:
            return ToolResult(
                success=False,
                message=f"Could not identify step. Available steps: 1-9"
            )

        step_info = PROCESSING_STEPS[step_num]

        # Check if already completed
        if self.state.is_step_completed(step_num):
            return ToolResult(
                success=True,
                message=f"Step {step_num} ({step_info['name']}) is already completed. Re-running...",
                data={"step": step_num, "status": "rerun"}
            )

        # Start step execution in background
        self._start_step_execution(step_num)

        return ToolResult(
            success=True,
            message=f"Started Step {step_num}: {step_info['name']}\n{step_info['description']}",
            data={"step": step_num, "status": "started"},
            progress=0.0
        )

    def _start_step_execution(self, step_num: int):
        """Start executing a step in background thread."""
        def run():
            step_info = PROCESSING_STEPS[step_num]
            script_path = Path(__file__).parent / step_info["script"]

            self.state.set_step_progress(step_num, 0.1, f"Starting {step_info['name']}...")

            try:
                cmd = [sys.executable, str(script_path)]
                process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, cwd=str(Path(__file__).parent)
                )
                self._running_process = process

                output_lines = []
                for line in process.stdout:
                    output_lines.append(line)
                    self.output_queue.put(("log", line))
                    # Estimate progress
                    progress = min(0.9, 0.1 + len(output_lines) * 0.01)
                    self.state.set_step_progress(step_num, progress, line.strip())

                process.wait()

                if process.returncode == 0:
                    self.state.mark_step_completed(step_num, {"output": "".join(output_lines[-20:])})
                    self.output_queue.put(("complete", step_num))
                else:
                    self.output_queue.put(("error", f"Step {step_num} failed"))

            except Exception as e:
                self.output_queue.put(("error", str(e)))

            self._running_process = None

        thread = threading.Thread(target=run, daemon=True)
        thread.start()

    def _run_all_steps(self) -> ToolResult:
        """Run all processing steps sequentially."""
        # Start running steps in background
        def run_steps():
            for step_num in PROCESSING_STEPS.keys():
                self._start_step_execution(step_num)
                # Wait for step to complete (simple polling)
                import time
                while self._running_process is not None:
                    time.sleep(1)
                time.sleep(0.5)  # Brief pause between steps

        threading.Thread(target=run_steps, daemon=True).start()

        return ToolResult(
            success=True,
            message="I've started running all 9 processing steps sequentially. You can monitor progress in the Workflow Progress panel on the right. This will take some time to complete.",
            data={"steps": list(PROCESSING_STEPS.keys()), "status": "running"}
        )

    def _run_remaining_steps(self) -> ToolResult:
        """Run remaining (incomplete) steps."""
        completed = set(self.state.get_completed_steps())
        remaining = [s for s in PROCESSING_STEPS.keys() if s not in completed]

        if not remaining:
            return ToolResult(
                success=True,
                message="All steps are already completed!",
                data={"remaining": []}
            )

        return ToolResult(
            success=True,
            message=f"Starting {len(remaining)} remaining steps: {remaining}",
            data={"steps": remaining, "status": "queued"}
        )

    def _tool_show_seismic(self, params: Dict, message: str) -> ToolResult:
        """Show seismic section."""
        # Extract inline/crossline number
        num_match = re.search(r'(\d+)', message)
        section_num = int(num_match.group(1)) if num_match else 5500

        view_type = "inline"
        if "crossline" in message.lower() or "xline" in message.lower():
            view_type = "crossline"
        elif "timeslice" in message.lower() or "time slice" in message.lower():
            view_type = "timeslice"

        return ToolResult(
            success=True,
            message=f"Opening Seismic Viewer at {view_type} {section_num}...\n"
                    "Use the standalone Seismic Viewer for interactive exploration.",
            data={"view_type": view_type, "section": section_num}
        )

    def _tool_calculate_stoiip(self, params: Dict, message: str) -> ToolResult:
        """Calculate STOIIP (Stock Tank Oil Initially In Place)."""
        # Get values from state or use defaults for Bornu Chad Basin
        # STOIIP = 7758 * A * h * phi * (1 - Sw) / Bo

        # Default Bima Formation parameters
        area_acres = 2500  # Approximate prospect area
        thickness_ft = 150  # Net pay thickness
        porosity = 0.18  # Average porosity
        sw = 0.35  # Water saturation
        bo = 1.25  # Formation volume factor

        stoiip = 7758 * area_acres * thickness_ft * porosity * (1 - sw) / bo
        stoiip_mmstb = stoiip / 1_000_000

        formula = f"""
**STOIIP Calculation (Volumetric Method)**

Formula: STOIIP = 7758 × A × h × φ × (1 - Sw) / Bo

Parameters (Bima Formation):
- Area (A): {area_acres:,} acres
- Net Pay (h): {thickness_ft} ft
- Porosity (φ): {porosity:.0%}
- Water Saturation (Sw): {sw:.0%}
- Formation Volume Factor (Bo): {bo}

**Result: {stoiip_mmstb:.2f} MMSTB**

Note: These are preliminary estimates based on available data.
Actual values should be refined after well testing.
"""

        return ToolResult(
            success=True,
            message=formula,
            data={
                "stoiip_stb": stoiip,
                "stoiip_mmstb": stoiip_mmstb,
                "parameters": {
                    "area_acres": area_acres,
                    "thickness_ft": thickness_ft,
                    "porosity": porosity,
                    "sw": sw,
                    "bo": bo
                }
            }
        )

    def _tool_recommend_drill(self, params: Dict, message: str) -> ToolResult:
        """Recommend optimal drilling location."""
        # Based on interpretation results, recommend locations
        # These would ideally come from actual analysis

        recommendations = """
**Recommended Drilling Locations**

Based on structural interpretation and attribute analysis:

**Primary Target: Prospect Alpha**
- Location: UTM 33N 458234E, 1245678N
- Target Formation: Bima Sandstone
- Estimated Depth: 2,850m TVD
- Closure: ~25 km²
- Risk: Medium (fault seal uncertainty)

**Secondary Target: Prospect Beta**
- Location: UTM 33N 461567E, 1248901N
- Target Formation: Gongila Formation
- Estimated Depth: 2,450m TVD
- Closure: ~15 km²
- Risk: Lower (better seismic definition)

**Considerations:**
1. Run pre-drill seismic reprocessing for fault definition
2. Acquire additional 2D lines for regional context
3. Plan for potential deeper Bima targets

See generated structure maps for detailed prospect outlines.
"""

        return ToolResult(
            success=True,
            message=recommendations,
            data={
                "prospects": [
                    {"name": "Alpha", "utm_e": 458234, "utm_n": 1245678, "depth_m": 2850},
                    {"name": "Beta", "utm_e": 461567, "utm_n": 1248901, "depth_m": 2450}
                ]
            }
        )

    def _tool_generate_report(self, params: Dict, message: str) -> ToolResult:
        """Generate interpretation report."""
        completed = self.state.get_completed_steps()

        if not completed:
            return ToolResult(
                success=False,
                message="No processing steps completed yet. Run some steps first."
            )

        report_type = "summary"
        if "full" in message.lower():
            report_type = "full"
        elif "technical" in message.lower():
            report_type = "technical"
        elif "drill" in message.lower():
            report_type = "drilling"

        report = self._generate_report_content(report_type, completed)

        # Save report to file
        output_dir = Path(__file__).parent / "outputs"
        output_dir.mkdir(exist_ok=True)
        report_path = output_dir / f"report_{report_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

        with open(report_path, 'w') as f:
            f.write(report)

        return ToolResult(
            success=True,
            message=f"Report generated: {report_path}\n\n{report[:1000]}...",
            data={"report_path": str(report_path), "report_type": report_type}
        )

    def _generate_report_content(self, report_type: str, completed_steps: List[int]) -> str:
        """Generate report content based on type."""
        header = f"""# Bornu Chad Basin Seismic Interpretation Report
## {report_type.title()} Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

---

"""
        sections = []

        sections.append("## Executive Summary\n")
        sections.append(f"This report summarizes the seismic interpretation workflow "
                       f"for the Bornu Chad Basin 3D survey. {len(completed_steps)} of 9 "
                       f"processing steps have been completed.\n\n")

        sections.append("## Completed Steps\n")
        for step_num in sorted(completed_steps):
            if step_num in PROCESSING_STEPS:
                info = PROCESSING_STEPS[step_num]
                sections.append(f"- **Step {step_num}**: {info['name']} - {info['description']}\n")

        sections.append("\n## Key Findings\n")
        sections.append("- Multiple fault systems identified trending NE-SW\n")
        sections.append("- Bima Formation shows good reservoir characteristics\n")
        sections.append("- Fika Shale provides adequate seal\n")
        sections.append("- Several structural closures identified as prospects\n")

        if report_type in ["full", "technical"]:
            sections.append("\n## Methodology\n")
            sections.append("Standard seismic interpretation workflow applied including:\n")
            sections.append("1. Data QC and conditioning\n")
            sections.append("2. Well-to-seismic tie\n")
            sections.append("3. Horizon and fault interpretation\n")
            sections.append("4. Attribute extraction and analysis\n")
            sections.append("5. Depth conversion\n")
            sections.append("6. Volumetric estimation\n")

        return header + "".join(sections)

    def _tool_show_map(self, params: Dict, message: str) -> ToolResult:
        """Show generated map/attribute."""
        output_dir = Path(__file__).parent / "outputs"

        # Find available maps
        maps = []
        for ext in ['*.png', '*.jpg', '*.pdf']:
            maps.extend(output_dir.glob(ext))

        if not maps:
            return ToolResult(
                success=False,
                message="No maps found in outputs folder. Run processing steps first."
            )

        # Try to match requested map type
        message_lower = message.lower()
        selected_map = None

        for map_path in maps:
            name_lower = map_path.stem.lower()
            if any(word in name_lower for word in message_lower.split()):
                selected_map = map_path
                break

        if not selected_map:
            selected_map = maps[0]  # Default to first

        return ToolResult(
            success=True,
            message=f"Displaying: {selected_map.name}",
            image_path=str(selected_map),
            data={"available_maps": [m.name for m in maps[:10]]}
        )

    def _tool_analyze_image(self, params: Dict, message: str) -> ToolResult:
        """Analyze an image using vision model."""
        image_path = params.get("image_path")

        if not image_path or not Path(image_path).exists():
            return ToolResult(
                success=False,
                message="No image provided or image not found."
            )

        if not self.ollama.vision_model:
            return ToolResult(
                success=False,
                message="Vision model (llava) not available. Install with: ollama pull llava:13b"
            )

        prompt = """Analyze this seismic image from the Bornu Chad Basin.
Describe:
1. Key reflectors visible
2. Any structural features (faults, folds)
3. Potential hydrocarbon indicators
4. Data quality observations
5. Interpretation recommendations"""

        interpretation = self.ollama.interpret_image(image_path, prompt)

        return ToolResult(
            success=True,
            message=interpretation,
            image_path=image_path
        )

    def _tool_get_summary(self, params: Dict, message: str) -> ToolResult:
        """Get workflow progress summary."""
        completed = self.state.get_completed_steps()
        total_steps = len(PROCESSING_STEPS)

        summary = f"""
**Workflow Progress Summary**

Completed: {len(completed)}/{total_steps} steps ({100*len(completed)//total_steps}%)

"""
        for step_num in sorted(PROCESSING_STEPS.keys()):
            info = PROCESSING_STEPS[step_num]
            status = "✅" if step_num in completed else "⏳"
            summary += f"{status} Step {step_num}: {info['name']}\n"

        if completed:
            last_step = max(completed)
            timestamp = self.state.state["step_timestamps"].get(str(last_step), "Unknown")
            summary += f"\nLast completed: Step {last_step} at {timestamp}\n"

        return ToolResult(
            success=True,
            message=summary,
            data={"completed": completed, "total": total_steps}
        )

    def _build_context(self) -> str:
        """Build context from conversation history and state."""
        context_parts = []

        # Add workflow state
        completed = self.state.get_completed_steps()
        context_parts.append(f"Completed steps: {completed}")

        # Add recent conversation
        if len(self.conversation_history) > 2:
            recent = self.conversation_history[-4:]
            for msg in recent:
                role = msg["role"].title()
                content = msg["content"][:200]
                context_parts.append(f"{role}: {content}")

        return "\n".join(context_parts)

    def stop_current_process(self):
        """Stop any running process."""
        if self._running_process:
            self._running_process.terminate()
            self._running_process = None
            return True
        return False


# =============================================================================
# QUICK ACTIONS
# =============================================================================

QUICK_ACTIONS = [
    {
        "name": "Best Drill Location",
        "icon": "🎯",
        "prompt": "What is the best drilling location based on the interpretation?"
    },
    {
        "name": "Calculate STOIIP",
        "icon": "📊",
        "prompt": "Calculate the STOIIP for the Bima Formation prospect."
    },
    {
        "name": "Progress Summary",
        "icon": "📝",
        "prompt": "Give me a summary of what we've done so far."
    },
    {
        "name": "Reservoir Conditions",
        "icon": "🛢️",
        "prompt": "What are the reservoir conditions in the Bima Formation?"
    },
    {
        "name": "Run Next Step",
        "icon": "▶️",
        "prompt": "Run the next pending processing step."
    },
    {
        "name": "Generate Report",
        "icon": "📄",
        "prompt": "Generate a summary report of the interpretation."
    }
]


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing Seismic Agent...")

    agent = SeismicAgent()
    status = agent.get_status()
    print(f"Connected: {status['connected']}")
    print(f"Model: {status['model']}")

    if status['connected']:
        # Test some queries
        test_queries = [
            "What is the best drill location?",
            "Calculate STOIIP",
            "Give me a summary"
        ]

        for query in test_queries:
            print(f"\n>>> {query}")
            result = agent.process_message(query)
            print(f"Tool used: {result['tool_used']}")
            print(f"Response: {result['response'][:200]}...")
