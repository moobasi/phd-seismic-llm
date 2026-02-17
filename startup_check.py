"""
Startup Check Script for PhD Seismic Framework
Runs all checks and then launches the main GUI.
"""

import sys
import subprocess
import os
import json

def main():
    print()
    print('=' * 70)
    print('   PhD SEISMIC INTERPRETATION FRAMEWORK')
    print('   Bornu Chad Basin - LLM-Assisted Interpretation')
    print('=' * 70)
    print('   Author: Moses Ekene Obasi')
    print('   Supervisor: Prof. Dominic Akam Obi')
    print('   Institution: University of Calabar, Nigeria')
    print('=' * 70)
    print()

    # Check dependencies
    print('[STEP 1/4] Checking core dependencies...')
    missing = []
    for mod in ['tkinter', 'numpy', 'scipy', 'matplotlib', 'pandas', 'segyio', 'lasio']:
        try:
            __import__(mod)
            print(f'   [OK] {mod}')
        except ImportError:
            print(f'   [X] {mod} - MISSING')
            missing.append(mod)

    if missing:
        print()
        print('[ACTION] Installing missing packages...')
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet'] + missing, check=False)
        print('[DONE] Packages installed.')
    print()

    # Check PyTorch
    print('[STEP 2/4] Checking deep learning dependencies...')
    try:
        import torch
        cuda_status = 'Available (GPU)' if torch.cuda.is_available() else 'CPU mode'
        print(f'   [OK] PyTorch - CUDA: {cuda_status}')
    except ImportError:
        print('   [!] PyTorch - NOT INSTALLED (optional)')

    # Check model weights
    model_path = os.path.join(os.path.dirname(__file__),
                               'deep_learning', 'models', 'faultseg3d', 'faultseg3d_pytorch.pth')
    if os.path.exists(model_path):
        print('   [OK] FaultSeg3D model weights found')
    else:
        print('   [!] FaultSeg3D weights not found')
    print()

    # Check Ollama
    print('[STEP 3/4] Checking Ollama LLM service...')
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print('   [OK] Ollama is running')
            if 'llava' in result.stdout.lower():
                print('   [OK] llava model available')
            else:
                print('   [!] llava not found (ollama pull llava:13b)')
        else:
            print('   [!] Ollama not responding')
    except FileNotFoundError:
        print('   [!] Ollama not installed (optional)')
    except Exception:
        print('   [!] Ollama check failed (optional)')
    print()

    # Check saved progress
    print('[STEP 4/4] Checking saved progress...')
    base_dir = os.path.dirname(__file__)

    config_path = os.path.join(base_dir, 'project_config.json')
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            print(f"   [OK] Project: {cfg.get('project_name', 'Unknown')}")
        except:
            print('   [i] Project config exists but could not read')
    else:
        print('   [i] No saved project - will create new')

    state_path = os.path.join(base_dir, 'processing_state.json')
    if os.path.exists(state_path):
        try:
            with open(state_path) as f:
                state = json.load(f)
            steps = len(state.get('completed_steps', []))
            print(f'   [OK] Progress: {steps}/9 steps completed')
        except:
            print('   [i] Progress file exists but could not read')
    else:
        print('   [i] No saved progress - starting fresh')

    print()
    print('=' * 70)
    print('   LAUNCHING GUI...')
    print('=' * 70)
    print()

    # Launch GUI
    from phd_workflow_gui import main as gui_main
    gui_main()


if __name__ == "__main__":
    main()
