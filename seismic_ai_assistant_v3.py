"""
================================================================================
PHD SEISMIC AI ASSISTANT v3.0
================================================================================

Interactive AI-powered seismic interpretation assistant with agent capabilities.

Features:
- Natural language chat interface
- Agent with tool-calling (run steps, analyze, report)
- Visual analysis gallery
- Unified report generation
- Real-time workflow progress tracking

Author: Moses Ekene Obasi
PhD Research - University of Calabar
================================================================================
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
from typing import Optional, Dict, List, Callable
import threading
import queue
from datetime import datetime
import json
import os

# Import agent
try:
    from seismic_agent import (
        SeismicAgent, UnifiedStateManager, OllamaClient,
        PROCESSING_STEPS, QUICK_ACTIONS, ToolResult
    )
    AGENT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Agent not available: {e}")
    AGENT_AVAILABLE = False

# Import PIL for image display
try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# =============================================================================
# THEME
# =============================================================================

class Theme:
    """Modern dark theme."""
    BG_DARK = "#1a1a2e"
    BG_SURFACE = "#16213e"
    BG_OVERLAY = "#0f3460"
    BG_INPUT = "#1f2940"
    BG_CHAT_USER = "#2d4a6f"
    BG_CHAT_AI = "#1e3a5f"

    FG_PRIMARY = "#e6e6e6"
    FG_SECONDARY = "#a0a0a0"
    FG_MUTED = "#6b7280"

    ACCENT = "#e94560"
    ACCENT_HOVER = "#ff6b6b"
    SUCCESS = "#4ecca3"
    WARNING = "#ffc107"
    ERROR = "#ff4757"
    INFO = "#3498db"

    FONT_TITLE = ("Segoe UI", 16, "bold")
    FONT_HEADING = ("Segoe UI", 12, "bold")
    FONT_BODY = ("Segoe UI", 10)
    FONT_SMALL = ("Segoe UI", 9)
    FONT_CHAT = ("Segoe UI", 11)
    FONT_MONO = ("Consolas", 10)


# =============================================================================
# CHAT DISPLAY WIDGET
# =============================================================================

class ChatDisplay(tk.Frame):
    """
    Rich chat display with message bubbles.
    Supports text, images, and progress indicators.
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=Theme.BG_DARK, **kwargs)

        # Scrollable canvas
        self.canvas = tk.Canvas(self, bg=Theme.BG_DARK, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Container frame for messages
        self.messages_frame = tk.Frame(self.canvas, bg=Theme.BG_DARK)
        self.canvas_window = self.canvas.create_window(
            (0, 0), window=self.messages_frame, anchor="nw"
        )

        # Bind resize
        self.messages_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # Enable scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        self.messages = []
        self.image_refs = []  # Keep references to prevent garbage collection

    def _on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self.canvas_window, width=event.width - 20)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def add_message(self, text: str, is_user: bool = False,
                    image_path: Optional[str] = None):
        """Add a message bubble to the chat."""
        bubble_frame = tk.Frame(self.messages_frame, bg=Theme.BG_DARK)
        bubble_frame.pack(fill=tk.X, padx=10, pady=5)

        # Alignment
        if is_user:
            anchor = "e"
            bg_color = Theme.BG_CHAT_USER
            label_text = "You"
        else:
            anchor = "w"
            bg_color = Theme.BG_CHAT_AI
            label_text = "AI Assistant"

        # Inner frame for bubble
        inner_frame = tk.Frame(bubble_frame, bg=bg_color, padx=12, pady=8)
        inner_frame.pack(anchor=anchor, padx=5)

        # Role label
        tk.Label(
            inner_frame, text=label_text,
            font=Theme.FONT_SMALL, bg=bg_color,
            fg=Theme.ACCENT if is_user else Theme.SUCCESS
        ).pack(anchor="w")

        # Message text
        text_label = tk.Label(
            inner_frame, text=text,
            font=Theme.FONT_CHAT, bg=bg_color, fg=Theme.FG_PRIMARY,
            wraplength=500, justify="left", anchor="w"
        )
        text_label.pack(anchor="w", pady=(5, 0))

        # Image if provided
        if image_path and PIL_AVAILABLE and Path(image_path).exists():
            try:
                img = Image.open(image_path)
                # Resize for display
                img.thumbnail((400, 300))
                photo = ImageTk.PhotoImage(img)
                self.image_refs.append(photo)  # Keep reference

                img_label = tk.Label(inner_frame, image=photo, bg=bg_color)
                img_label.pack(anchor="w", pady=(10, 0))
            except Exception as e:
                print(f"Error loading image: {e}")

        self.messages.append(bubble_frame)

        # Scroll to bottom
        self.canvas.update_idletasks()
        self.canvas.yview_moveto(1.0)

    def add_typing_indicator(self):
        """Add typing indicator."""
        self.typing_frame = tk.Frame(self.messages_frame, bg=Theme.BG_DARK)
        self.typing_frame.pack(fill=tk.X, padx=10, pady=5)

        inner = tk.Frame(self.typing_frame, bg=Theme.BG_CHAT_AI, padx=12, pady=8)
        inner.pack(anchor="w", padx=5)

        tk.Label(
            inner, text="AI is thinking...",
            font=Theme.FONT_SMALL, bg=Theme.BG_CHAT_AI, fg=Theme.FG_MUTED
        ).pack()

        self.canvas.update_idletasks()
        self.canvas.yview_moveto(1.0)

    def remove_typing_indicator(self):
        """Remove typing indicator."""
        if hasattr(self, 'typing_frame') and self.typing_frame:
            self.typing_frame.destroy()
            self.typing_frame = None

    def clear(self):
        """Clear all messages."""
        for msg in self.messages:
            msg.destroy()
        self.messages = []
        self.image_refs = []


# =============================================================================
# PROGRESS PANEL
# =============================================================================

class ProgressPanel(tk.Frame):
    """
    Workflow progress sidebar showing step statuses.
    """

    def __init__(self, parent, state_manager: Optional['UnifiedStateManager'] = None, **kwargs):
        super().__init__(parent, bg=Theme.BG_SURFACE, **kwargs)

        self.state = state_manager
        self.step_labels: Dict[int, tk.Label] = {}
        self.step_status: Dict[int, tk.Label] = {}

        self._create_ui()

        # Register for state updates
        if self.state:
            self.state.register_callback(self._on_state_change)

    def _create_ui(self):
        # Title
        tk.Label(
            self, text="Workflow Progress",
            font=Theme.FONT_HEADING, bg=Theme.BG_SURFACE, fg=Theme.ACCENT
        ).pack(pady=(10, 15), padx=10)

        # Steps container
        steps_frame = tk.Frame(self, bg=Theme.BG_SURFACE)
        steps_frame.pack(fill=tk.BOTH, expand=True, padx=10)

        for step_num, info in PROCESSING_STEPS.items():
            row = tk.Frame(steps_frame, bg=Theme.BG_SURFACE)
            row.pack(fill=tk.X, pady=3)

            # Status icon
            status_label = tk.Label(
                row, text="⏳", font=("Segoe UI", 10),
                bg=Theme.BG_SURFACE, fg=Theme.FG_MUTED, width=2
            )
            status_label.pack(side=tk.LEFT)
            self.step_status[step_num] = status_label

            # Step name (abbreviated)
            name = info["name"]
            if len(name) > 18:
                name = name[:16] + "..."
            step_label = tk.Label(
                row, text=f"{step_num}. {name}",
                font=Theme.FONT_SMALL, bg=Theme.BG_SURFACE, fg=Theme.FG_SECONDARY,
                anchor="w"
            )
            step_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.step_labels[step_num] = step_label

        # Progress bar for current step
        self.progress_frame = tk.Frame(self, bg=Theme.BG_SURFACE)
        self.progress_frame.pack(fill=tk.X, padx=10, pady=15)

        self.current_step_label = tk.Label(
            self.progress_frame, text="",
            font=Theme.FONT_SMALL, bg=Theme.BG_SURFACE, fg=Theme.WARNING
        )
        self.current_step_label.pack(fill=tk.X)

        self.progress_bar = ttk.Progressbar(
            self.progress_frame, mode='determinate', length=150
        )

        self._update_display()

    def _on_state_change(self, state: Dict):
        """Handle state change callback."""
        self.after(0, self._update_display)

    def _update_display(self):
        """Update the display from current state."""
        if not self.state:
            return

        completed = self.state.get_completed_steps()
        current = self.state.state.get("current_step")
        progress = self.state.state.get("current_progress", 0)

        for step_num in PROCESSING_STEPS.keys():
            if step_num in completed:
                self.step_status[step_num].config(text="✅", fg=Theme.SUCCESS)
                self.step_labels[step_num].config(fg=Theme.SUCCESS)
            elif step_num == current:
                self.step_status[step_num].config(text="🔄", fg=Theme.WARNING)
                self.step_labels[step_num].config(fg=Theme.WARNING)
            else:
                self.step_status[step_num].config(text="⏳", fg=Theme.FG_MUTED)
                self.step_labels[step_num].config(fg=Theme.FG_SECONDARY)

        # Update progress bar
        if current:
            self.current_step_label.config(
                text=f"Running Step {current}..."
            )
            self.progress_bar.pack(fill=tk.X, pady=(5, 0))
            self.progress_bar['value'] = progress * 100
        else:
            self.current_step_label.config(text="")
            self.progress_bar.pack_forget()


# =============================================================================
# QUICK ACTIONS PANEL
# =============================================================================

class QuickActionsPanel(tk.Frame):
    """
    Quick action buttons for common tasks.
    """

    def __init__(self, parent, on_action: Callable[[str], None], **kwargs):
        super().__init__(parent, bg=Theme.BG_SURFACE, **kwargs)

        self.on_action = on_action

        tk.Label(
            self, text="Quick Actions",
            font=Theme.FONT_HEADING, bg=Theme.BG_SURFACE, fg=Theme.ACCENT
        ).pack(pady=(10, 10), padx=10)

        for action in QUICK_ACTIONS:
            btn = tk.Button(
                self,
                text=f"{action['icon']} {action['name']}",
                font=Theme.FONT_SMALL,
                bg=Theme.BG_OVERLAY, fg=Theme.FG_PRIMARY,
                relief="flat", cursor="hand2",
                command=lambda a=action: self.on_action(a['prompt'])
            )
            btn.pack(fill=tk.X, padx=10, pady=3)


# =============================================================================
# VISUAL ANALYSIS TAB
# =============================================================================

class VisualAnalysisTab(tk.Frame):
    """
    Visual analysis tab for viewing generated outputs.
    """

    def __init__(self, parent, agent: Optional['SeismicAgent'] = None, **kwargs):
        super().__init__(parent, bg=Theme.BG_DARK, **kwargs)

        self.agent = agent
        self.current_image_path = None
        self.image_ref = None

        self._create_ui()

    def _create_ui(self):
        # Controls bar
        controls = tk.Frame(self, bg=Theme.BG_SURFACE)
        controls.pack(fill=tk.X, padx=10, pady=10)

        tk.Button(
            controls, text="📁 Load Image",
            font=Theme.FONT_BODY, bg=Theme.BG_OVERLAY, fg=Theme.FG_PRIMARY,
            relief="flat", cursor="hand2", command=self._load_image
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            controls, text="🔄 Refresh Outputs",
            font=Theme.FONT_BODY, bg=Theme.BG_OVERLAY, fg=Theme.FG_PRIMARY,
            relief="flat", cursor="hand2", command=self._refresh_outputs
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            controls, text="🤖 Interpret Image",
            font=Theme.FONT_BODY, bg=Theme.ACCENT, fg="white",
            relief="flat", cursor="hand2", command=self._interpret_current
        ).pack(side=tk.RIGHT, padx=5)

        # Main content
        content = tk.Frame(self, bg=Theme.BG_DARK)
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Image display (left, larger)
        image_frame = tk.LabelFrame(
            content, text="Image Preview",
            font=Theme.FONT_BODY, bg=Theme.BG_SURFACE, fg=Theme.FG_PRIMARY
        )
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.image_label = tk.Label(
            image_frame, text="No image loaded\n\nClick 'Load Image' or select from outputs",
            font=Theme.FONT_BODY, bg=Theme.BG_SURFACE, fg=Theme.FG_MUTED
        )
        self.image_label.pack(expand=True, padx=10, pady=10)

        # Outputs list (right)
        outputs_frame = tk.LabelFrame(
            content, text="Generated Outputs",
            font=Theme.FONT_BODY, bg=Theme.BG_SURFACE, fg=Theme.FG_PRIMARY,
            width=250
        )
        outputs_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        outputs_frame.pack_propagate(False)

        self.outputs_listbox = tk.Listbox(
            outputs_frame, bg=Theme.BG_INPUT, fg=Theme.FG_PRIMARY,
            font=Theme.FONT_SMALL, selectbackground=Theme.ACCENT,
            relief="flat", highlightthickness=0
        )
        self.outputs_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.outputs_listbox.bind('<<ListboxSelect>>', self._on_output_selected)

        self._refresh_outputs()

    def _refresh_outputs(self):
        """Refresh list of output files."""
        self.outputs_listbox.delete(0, tk.END)

        output_dirs = [
            Path(__file__).parent / "outputs",
            Path(__file__).parent / "interpretation" / "real_outputs",
            Path(__file__).parent / "horizon_outputs"
        ]

        self.output_files = []
        for output_dir in output_dirs:
            if output_dir.exists():
                for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
                    self.output_files.extend(output_dir.glob(ext))

        for f in sorted(self.output_files, key=lambda x: x.stat().st_mtime, reverse=True)[:30]:
            self.outputs_listbox.insert(tk.END, f.name)

    def _on_output_selected(self, event):
        """Handle output file selection."""
        selection = self.outputs_listbox.curselection()
        if selection:
            idx = selection[0]
            if idx < len(self.output_files):
                self._display_image(str(self.output_files[idx]))

    def _load_image(self):
        """Load image from file dialog."""
        filepath = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        if filepath:
            self._display_image(filepath)

    def _display_image(self, path: str):
        """Display image in preview area."""
        self.current_image_path = path

        if not PIL_AVAILABLE:
            self.image_label.config(text=f"PIL not available\n\n{Path(path).name}")
            return

        try:
            img = Image.open(path)

            # Get frame size
            frame_width = self.image_label.winfo_width() or 600
            frame_height = self.image_label.winfo_height() or 400

            # Resize maintaining aspect ratio
            img.thumbnail((frame_width - 20, frame_height - 20))

            self.image_ref = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.image_ref, text="")

        except Exception as e:
            self.image_label.config(text=f"Error loading image:\n{e}")

    def _interpret_current(self):
        """Send current image to AI for interpretation."""
        if not self.current_image_path:
            messagebox.showinfo("No Image", "Please load an image first.")
            return

        if not self.agent:
            messagebox.showerror("Error", "AI Agent not available.")
            return

        # This will trigger interpretation through the agent
        # Parent should handle switching to chat tab
        self.event_generate("<<InterpretImage>>", data=self.current_image_path)


# =============================================================================
# REPORTS TAB
# =============================================================================

class ReportsTab(tk.Frame):
    """
    Reports generation and viewing tab.
    """

    def __init__(self, parent, agent: Optional['SeismicAgent'] = None, **kwargs):
        super().__init__(parent, bg=Theme.BG_DARK, **kwargs)

        self.agent = agent
        self._create_ui()

    def _create_ui(self):
        # Controls
        controls = tk.Frame(self, bg=Theme.BG_SURFACE)
        controls.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(
            controls, text="Report Type:",
            font=Theme.FONT_BODY, bg=Theme.BG_SURFACE, fg=Theme.FG_PRIMARY
        ).pack(side=tk.LEFT, padx=5)

        self.report_type_var = tk.StringVar(value="summary")
        report_types = ["summary", "technical", "drilling", "volumetrics", "full"]
        self.report_combo = ttk.Combobox(
            controls, textvariable=self.report_type_var,
            values=report_types, state='readonly', width=15
        )
        self.report_combo.pack(side=tk.LEFT, padx=5)

        tk.Button(
            controls, text="📝 Generate Report",
            font=Theme.FONT_BODY, bg=Theme.ACCENT, fg="white",
            relief="flat", cursor="hand2", command=self._generate_report
        ).pack(side=tk.LEFT, padx=15)

        tk.Button(
            controls, text="💾 Export PDF",
            font=Theme.FONT_BODY, bg=Theme.BG_OVERLAY, fg=Theme.FG_PRIMARY,
            relief="flat", cursor="hand2", command=self._export_pdf
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            controls, text="📋 Copy to Clipboard",
            font=Theme.FONT_BODY, bg=Theme.BG_OVERLAY, fg=Theme.FG_PRIMARY,
            relief="flat", cursor="hand2", command=self._copy_to_clipboard
        ).pack(side=tk.LEFT, padx=5)

        # Report display
        report_frame = tk.LabelFrame(
            self, text="Report",
            font=Theme.FONT_BODY, bg=Theme.BG_SURFACE, fg=Theme.FG_PRIMARY
        )
        report_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.report_text = scrolledtext.ScrolledText(
            report_frame, bg=Theme.BG_INPUT, fg=Theme.FG_PRIMARY,
            font=Theme.FONT_MONO, wrap=tk.WORD,
            insertbackground=Theme.FG_PRIMARY
        )
        self.report_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Insert placeholder
        self.report_text.insert(tk.END, """
═══════════════════════════════════════════════════════════════
                    SEISMIC INTERPRETATION REPORT
                       Bornu Chad Basin, Nigeria
═══════════════════════════════════════════════════════════════

Click "Generate Report" to create a report based on completed
processing steps and interpretation results.

Report Types:
• Summary    - Quick overview of completed work
• Technical  - Detailed methodology and results
• Drilling   - Recommended well locations
• Volumetrics - STOIIP/GIIP calculations
• Full       - Comprehensive document

═══════════════════════════════════════════════════════════════
""")

    def _generate_report(self):
        """Generate report using agent."""
        if not self.agent:
            messagebox.showerror("Error", "AI Agent not available.")
            return

        report_type = self.report_type_var.get()

        self.report_text.delete(1.0, tk.END)
        self.report_text.insert(tk.END, f"Generating {report_type} report...\n")
        self.update()

        # Run in thread
        def generate():
            result = self.agent.process_message(f"Generate a {report_type} report")
            self.after(0, lambda: self._display_report(result['response']))

        threading.Thread(target=generate, daemon=True).start()

    def _display_report(self, content: str):
        """Display generated report."""
        self.report_text.delete(1.0, tk.END)
        self.report_text.insert(tk.END, content)

    def _export_pdf(self):
        """Export report to PDF (placeholder)."""
        messagebox.showinfo(
            "Export PDF",
            "PDF export requires additional dependencies.\n\n"
            "For now, you can copy the report and paste into Word/Google Docs."
        )

    def _copy_to_clipboard(self):
        """Copy report to clipboard."""
        content = self.report_text.get(1.0, tk.END)
        self.clipboard_clear()
        self.clipboard_append(content)
        messagebox.showinfo("Copied", "Report copied to clipboard!")


# =============================================================================
# MAIN APPLICATION
# =============================================================================

class SeismicAIAssistant:
    """
    Main AI Assistant application.
    """

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("PhD Seismic AI Assistant")
        self.root.geometry("1300x850")
        self.root.configure(bg=Theme.BG_DARK)

        # Initialize state and agent
        self.state = UnifiedStateManager() if AGENT_AVAILABLE else None
        self.agent = SeismicAgent(self.state) if AGENT_AVAILABLE else None

        # Message queue for thread-safe UI updates
        self.message_queue = queue.Queue()

        self._setup_styles()
        self._create_ui()
        self._start_queue_processor()

        # Check connection
        self._check_connection()

    def _setup_styles(self):
        """Configure ttk styles."""
        style = ttk.Style()
        style.theme_use('clam')

        style.configure('TFrame', background=Theme.BG_DARK)
        style.configure('TLabel', background=Theme.BG_DARK, foreground=Theme.FG_PRIMARY)
        style.configure('TNotebook', background=Theme.BG_DARK)
        style.configure('TNotebook.Tab', background=Theme.BG_SURFACE,
                       foreground=Theme.FG_PRIMARY, padding=[15, 8])
        style.map('TNotebook.Tab',
                 background=[('selected', Theme.BG_OVERLAY)],
                 foreground=[('selected', Theme.ACCENT)])

    def _create_ui(self):
        """Create main UI."""
        # Header
        header = tk.Frame(self.root, bg=Theme.BG_SURFACE, height=60)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        tk.Label(
            header, text="🔬 PhD Seismic AI Assistant",
            font=Theme.FONT_TITLE, bg=Theme.BG_SURFACE, fg=Theme.FG_PRIMARY
        ).pack(side=tk.LEFT, padx=20, pady=15)

        # Connection status
        self.status_label = tk.Label(
            header, text="● Connecting...",
            font=Theme.FONT_BODY, bg=Theme.BG_SURFACE, fg=Theme.WARNING
        )
        self.status_label.pack(side=tk.RIGHT, padx=20)

        # Model info
        self.model_label = tk.Label(
            header, text="",
            font=Theme.FONT_SMALL, bg=Theme.BG_SURFACE, fg=Theme.FG_SECONDARY
        )
        self.model_label.pack(side=tk.RIGHT, padx=10)

        # Main notebook
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Tab 1: Chat & Agent
        self._create_chat_tab()

        # Tab 2: Visual Analysis
        visual_tab = VisualAnalysisTab(self.notebook, agent=self.agent)
        self.notebook.add(visual_tab, text="📊 Visual Analysis")
        visual_tab.bind("<<InterpretImage>>", self._on_interpret_image)

        # Tab 3: Reports
        reports_tab = ReportsTab(self.notebook, agent=self.agent)
        self.notebook.add(reports_tab, text="📋 Reports")

    def _create_chat_tab(self):
        """Create the chat & agent tab."""
        chat_tab = tk.Frame(self.notebook, bg=Theme.BG_DARK)
        self.notebook.add(chat_tab, text="💬 Chat & Agent")

        # Main layout: left (chat) + right (sidebar)
        main_frame = tk.Frame(chat_tab, bg=Theme.BG_DARK)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Chat area (left)
        chat_frame = tk.Frame(main_frame, bg=Theme.BG_DARK)
        chat_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # Chat display
        self.chat_display = ChatDisplay(chat_frame)
        self.chat_display.pack(fill=tk.BOTH, expand=True)

        # Add welcome message
        self.chat_display.add_message(
            "👋 Hello! I'm your AI assistant for seismic interpretation.\n\n"
            "I can help you:\n"
            "• Run processing steps\n"
            "• Analyze seismic data\n"
            "• Calculate volumetrics (STOIIP)\n"
            "• Recommend drilling locations\n"
            "• Generate reports\n\n"
            "Just ask me anything or use the Quick Actions on the right!",
            is_user=False
        )

        # Input area
        input_frame = tk.Frame(chat_frame, bg=Theme.BG_SURFACE, height=80)
        input_frame.pack(fill=tk.X, pady=(10, 0))
        input_frame.pack_propagate(False)

        # Input controls row
        input_row = tk.Frame(input_frame, bg=Theme.BG_SURFACE)
        input_row.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.input_text = tk.Text(
            input_row, height=2, bg=Theme.BG_INPUT, fg=Theme.FG_PRIMARY,
            font=Theme.FONT_CHAT, wrap=tk.WORD,
            insertbackground=Theme.FG_PRIMARY, relief="flat"
        )
        self.input_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        self.input_text.bind("<Return>", self._on_enter_pressed)
        self.input_text.bind("<Shift-Return>", lambda e: None)  # Allow shift+enter for newline

        # Buttons
        btn_frame = tk.Frame(input_row, bg=Theme.BG_SURFACE)
        btn_frame.pack(side=tk.RIGHT)

        tk.Button(
            btn_frame, text="📎",
            font=("Segoe UI", 14), bg=Theme.BG_OVERLAY, fg=Theme.FG_PRIMARY,
            relief="flat", cursor="hand2", width=3,
            command=self._attach_image
        ).pack(pady=2)

        self.send_btn = tk.Button(
            btn_frame, text="Send",
            font=Theme.FONT_BODY, bg=Theme.ACCENT, fg="white",
            relief="flat", cursor="hand2", width=8,
            command=self._send_message
        )
        self.send_btn.pack(pady=2)

        # Right sidebar
        sidebar = tk.Frame(main_frame, bg=Theme.BG_SURFACE, width=220)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        sidebar.pack_propagate(False)

        # Quick actions
        quick_actions = QuickActionsPanel(sidebar, on_action=self._on_quick_action)
        quick_actions.pack(fill=tk.X)

        # Separator
        ttk.Separator(sidebar, orient="horizontal").pack(fill=tk.X, pady=10)

        # Progress panel
        self.progress_panel = ProgressPanel(sidebar, state_manager=self.state)
        self.progress_panel.pack(fill=tk.BOTH, expand=True)

    def _check_connection(self):
        """Check Ollama connection."""
        def check():
            if self.agent:
                status = self.agent.get_status()
                self.message_queue.put(("connection", status))
            else:
                self.message_queue.put(("connection", {"connected": False}))

        threading.Thread(target=check, daemon=True).start()

    def _start_queue_processor(self):
        """Process message queue for thread-safe UI updates."""
        try:
            while True:
                msg_type, data = self.message_queue.get_nowait()

                if msg_type == "connection":
                    if data.get("connected"):
                        self.status_label.config(text="● Connected", fg=Theme.SUCCESS)
                        self.model_label.config(text=f"Model: {data.get('model', 'Unknown')}")
                    else:
                        self.status_label.config(text="● Not Connected", fg=Theme.ERROR)
                        self.model_label.config(text="Start Ollama to enable AI")

                elif msg_type == "response":
                    self.chat_display.remove_typing_indicator()
                    self.chat_display.add_message(
                        data.get("response", ""),
                        is_user=False,
                        image_path=data.get("image_path")
                    )
                    self.send_btn.config(state="normal")

                elif msg_type == "error":
                    self.chat_display.remove_typing_indicator()
                    self.chat_display.add_message(f"Error: {data}", is_user=False)
                    self.send_btn.config(state="normal")

        except queue.Empty:
            pass

        self.root.after(100, self._start_queue_processor)

    def _on_enter_pressed(self, event):
        """Handle enter key in input."""
        if not event.state & 0x1:  # Not shift+enter
            self._send_message()
            return "break"

    def _send_message(self):
        """Send user message to agent."""
        message = self.input_text.get(1.0, tk.END).strip()
        if not message:
            return

        self.input_text.delete(1.0, tk.END)

        # Display user message
        self.chat_display.add_message(message, is_user=True)
        self.chat_display.add_typing_indicator()
        self.send_btn.config(state="disabled")

        # Process in thread
        def process():
            try:
                if self.agent:
                    result = self.agent.process_message(message)
                    self.message_queue.put(("response", result))
                else:
                    self.message_queue.put(("error", "Agent not available"))
            except Exception as e:
                self.message_queue.put(("error", str(e)))

        threading.Thread(target=process, daemon=True).start()

    def _on_quick_action(self, prompt: str):
        """Handle quick action button click."""
        self.input_text.delete(1.0, tk.END)
        self.input_text.insert(tk.END, prompt)
        self._send_message()

    def _attach_image(self):
        """Attach image for analysis."""
        filepath = filedialog.askopenfilename(
            title="Select Image to Analyze",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp"),
                ("All files", "*.*")
            ]
        )
        if filepath:
            self.chat_display.add_message(
                f"Analyzing image: {Path(filepath).name}",
                is_user=True,
                image_path=filepath
            )
            self.chat_display.add_typing_indicator()
            self.send_btn.config(state="disabled")

            def process():
                try:
                    if self.agent:
                        result = self.agent.process_message(
                            "Interpret this seismic image",
                            image_path=filepath
                        )
                        self.message_queue.put(("response", result))
                    else:
                        self.message_queue.put(("error", "Agent not available"))
                except Exception as e:
                    self.message_queue.put(("error", str(e)))

            threading.Thread(target=process, daemon=True).start()

    def _on_interpret_image(self, event):
        """Handle image interpretation request from Visual Analysis tab."""
        # Switch to chat tab
        self.notebook.select(0)

        # Get image path (would need custom event handling)
        # For now, show message
        self.chat_display.add_message(
            "Image selected for interpretation. Click 'Send' to analyze.",
            is_user=False
        )

    def run(self):
        """Start the application."""
        self.root.mainloop()


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    app = SeismicAIAssistant()
    app.run()


if __name__ == "__main__":
    main()
