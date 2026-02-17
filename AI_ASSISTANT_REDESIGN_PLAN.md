# AI Assistant Redesign Plan

## Current Problems

### Redundancies Identified:
1. **Data Configuration Tab** - Duplicates the main workflow's Project Configuration
2. **Processing Tab** - Duplicates main workflow's step execution
3. **Seismic Viewer Tab** - Already exists as separate tool
4. **Well Tie Validation Tab** - Duplicates seismic_viewer.py functionality
5. **Multiple Interpretation Sources** - Different tabs produce conflicting interpretations

### Missing Agent Capabilities:
- Cannot autonomously run processing steps
- No tool/function calling framework
- No decision-making logic
- Cannot parse outputs and act on them

---

## Redesign Goals

1. **Single Interactive Chat Interface** - Fun, technical, conversational
2. **True Agent Capabilities** - Can run processes, analyze data, generate reports
3. **Unified with Main Workflow** - Shares state, no duplicate configuration
4. **Visual Integration** - Can show images/plots inline in chat
5. **Remove Redundant Tabs** - Only keep what's unique and necessary

---

## New Architecture

### Tab Structure (Reduced from 8 to 3)

```
┌─────────────────────────────────────────────────────────────┐
│  PhD Seismic AI Assistant                              [─][□][×]│
├─────────────────────────────────────────────────────────────┤
│ [💬 Chat & Agent] [📊 Visual Analysis] [📋 Reports]        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                    Tab Content Area                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Tab 1: Chat & Agent (Primary Interface)

### Layout:
```
┌──────────────────────────────────────┬─────────────────────┐
│                                      │   AGENT STATUS      │
│         CHAT CONVERSATION            │                     │
│                                      │   Model: qwen3:32b  │
│   [AI responses with rich formatting]│   Status: ● Ready   │
│   [Inline images when relevant]      │                     │
│   [Progress bars for running tasks]  │   ─────────────────│
│                                      │   QUICK ACTIONS     │
│                                      │                     │
│                                      │   [🎯 Best Drill]   │
│                                      │   [📊 STOIIP]       │
│                                      │   [📝 Summary]      │
│                                      │   [🗺️ Show Map]     │
│                                      │   [📈 Run Step...]  │
├──────────────────────────────────────┤                     │
│ Type your question or command...     │   ─────────────────│
│ [Send]  [🎤]  [📎 Attach Image]      │   WORKFLOW PROGRESS │
│                                      │                     │
│                                      │   Step 1: ✅ Done   │
│                                      │   Step 2: ✅ Done   │
│                                      │   Step 3: 🔄 75%    │
│                                      │   Step 4: ⏳ Pending│
│                                      │   ...               │
└──────────────────────────────────────┴─────────────────────┘
```

### Features:

1. **Conversational AI Chat**
   - Natural language queries
   - Rich text responses with formatting
   - Inline image display (seismic sections, maps)
   - Code blocks for technical details
   - Typing indicator animation

2. **Agent Capabilities** (Tool Calling)
   - `@run step 3` - Execute processing step
   - `@show inline 5500` - Display seismic section
   - `@analyze image` - Interpret attached image
   - `@report summary` - Generate summary report
   - `@calculate stoiip` - Run volumetrics
   - `@recommend drill` - Get drilling recommendations

3. **Workflow Progress Panel**
   - Real-time status from shared state
   - Progress bars for running steps
   - Click step to see details
   - Synced with main workflow GUI

4. **Quick Action Buttons**
   - Pre-built prompts for common tasks
   - One-click technical queries

### Agent Commands (Natural Language):

| User Says | Agent Action |
|-----------|--------------|
| "Run the EDA step" | Executes Step 1, streams output |
| "Show me inline 5500" | Displays seismic section inline |
| "What's the best drill location?" | Analyzes data, recommends coords |
| "Calculate STOIIP for Bima" | Runs volumetrics, shows formula |
| "Summarize what we've done" | Compiles processing summary |
| "Interpret this map" | Analyzes attached/displayed image |
| "Run all remaining steps" | Queues and executes pending steps |
| "What are the reservoir conditions?" | Describes pay zones, Sw, porosity |
| "Generate a full report" | Creates comprehensive PDF/markdown |

---

## Tab 2: Visual Analysis (Streamlined)

### Purpose: Quick visual review without redundant viewers

### Layout:
```
┌─────────────────────────────────────────────────────────────┐
│  [📁 Load Image]  [🔄 Refresh Outputs]  [View: Dropdown ▼]  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                    IMAGE/MAP DISPLAY                        │
│                    (Large preview area)                     │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  Available Outputs:                                         │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐ │
│  │ Structure   │ RMS         │ Sweetness   │ Synthetic   │ │
│  │ Map         │ Amplitude   │ Map         │ Seismogram  │ │
│  └─────────────┴─────────────┴─────────────┴─────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  [🤖 Interpret This Image]  ──→ Opens in Chat Tab          │
└─────────────────────────────────────────────────────────────┘
```

### Features:
- Gallery of generated outputs
- Quick preview with zoom
- One-click AI interpretation (sends to chat)
- No embedded seismic viewer (use standalone tool)

---

## Tab 3: Reports (Unified Output)

### Purpose: Generate and view professional reports

### Layout:
```
┌─────────────────────────────────────────────────────────────┐
│  Report Type: [Summary ▼]  [Generate Report]  [Export PDF]  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ═══════════════════════════════════════════════════════   │
│  BORNU CHAD BASIN INTERPRETATION REPORT                     │
│  ═══════════════════════════════════════════════════════   │
│                                                             │
│  ## Executive Summary                                       │
│  The analysis of 3D seismic data covering...               │
│                                                             │
│  ## Key Findings                                            │
│  - 3 major fault systems identified                         │
│  - Bima Formation shows good reservoir quality              │
│  - Recommended drill location: UTM 33N 456789E, 1234567N   │
│                                                             │
│  ## Volumetrics                                             │
│  STOIIP: 45.2 MMSTB                                         │
│  ...                                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Report Types:
- **Summary** - Quick overview of completed work
- **Technical** - Detailed methodology and results
- **Drilling Proposal** - Location recommendations
- **Volumetrics** - STOIIP/GIIP calculations
- **Full Report** - Comprehensive document

---

## Removed Tabs (Handled Elsewhere)

| Tab | Reason | New Location |
|-----|--------|--------------|
| Data Configuration | Duplicate | Main Workflow GUI only |
| Processing | Duplicate | Main GUI + Chat commands |
| Seismic Viewer | Duplicate | Standalone tool (already improved) |
| Well Tie Validation | Duplicate | Standalone seismic viewer |
| Well Logs & Picks | Partially keep | Merge into Visual Analysis |

---

## Main Workflow GUI Updates

### Add Progress Display:
```
┌─────────────────────────────────────────────────────────────┐
│  PhD Seismic Interpretation Workflow                        │
├──────────────────────────┬──────────────────────────────────┤
│                          │                                  │
│  Project Configuration   │  Processing Steps                │
│                          │                                  │
│  [Path selectors...]     │  ┌────────┐ ┌────────┐ ┌────────┐│
│                          │  │ Step 1 │ │ Step 2 │ │ Step 3 ││
│                          │  │  ✅    │ │  ✅    │ │  🔄    ││
│                          │  └────────┘ └────────┘ └────────┘│
│                          │                                  │
│                          │  Currently Running:              │
│                          │  ┌──────────────────────────────┐│
│                          │  │ Step 3: Well Integration     ││
│                          │  │ ████████████░░░░░░░░ 65%     ││
│                          │  │ Processing BULTE-1 logs...   ││
│                          │  └──────────────────────────────┘│
│                          │                                  │
└──────────────────────────┴──────────────────────────────────┘
```

---

## Implementation Steps

### Step 1: Create Unified State Manager
- Single source of truth for workflow state
- Shared between main GUI and AI Assistant
- Real-time sync using file watching

### Step 2: Build Agent Framework
```python
class SeismicAgent:
    """
    AI Agent with tool-calling capabilities.
    Can run processing steps, analyze data, generate reports.
    """

    def __init__(self, ollama_client, state_manager):
        self.ollama = ollama_client
        self.state = state_manager
        self.tools = self._register_tools()

    def _register_tools(self):
        return {
            'run_step': self.tool_run_step,
            'show_seismic': self.tool_show_seismic,
            'calculate_stoiip': self.tool_calculate_stoiip,
            'recommend_drill': self.tool_recommend_drill,
            'generate_report': self.tool_generate_report,
            'analyze_image': self.tool_analyze_image,
        }

    async def process_message(self, user_message):
        # Parse intent
        intent = self._parse_intent(user_message)

        # Execute tools if needed
        if intent.requires_tool:
            result = await self.tools[intent.tool_name](**intent.params)
            context = f"Tool result: {result}"
        else:
            context = ""

        # Generate response
        response = self.ollama.chat(
            prompt=user_message,
            context=context
        )

        return response
```

### Step 3: Redesign Chat Interface
- Rich text display with markdown rendering
- Inline image support
- Progress indicators for running tasks
- Animated typing indicator

### Step 4: Update Main Workflow GUI
- Add progress bar panel
- Show current step status
- Real-time log streaming

### Step 5: Remove Redundant Code
- Delete duplicate tabs
- Consolidate state management
- Unify configuration paths

---

## Files to Modify

| File | Changes |
|------|---------|
| `seismic_ai_assistant_v2.py` | Complete rewrite (3 tabs, agent framework) |
| `phd_workflow_gui.py` | Add progress panel, sync state |
| `project_config.py` | Add unified state management |
| `NEW: seismic_agent.py` | Agent framework with tools |

---

## Visual Design Guidelines

### Color Theme (Keep existing dark theme):
- Background: `#1a1a2e`
- Surface: `#16213e`
- Accent: `#e94560`
- Success: `#4ecca3`
- Text: `#e6e6e6`

### Chat Bubbles:
- User messages: Right-aligned, accent color border
- AI responses: Left-aligned, surface background
- System messages: Centered, muted color

### Animations:
- Typing indicator (3 bouncing dots)
- Progress bars (smooth transitions)
- Fade-in for new messages

---

## Summary

This redesign transforms the AI Assistant from a **tabbed viewer** into a **true AI agent** that:
1. Understands natural language commands
2. Can execute processing steps autonomously
3. Analyzes data and provides recommendations
4. Generates professional reports
5. Shows visual results inline in chat
6. Syncs with main workflow in real-time

The result is a **fun, interactive, unified experience** without redundancy.
