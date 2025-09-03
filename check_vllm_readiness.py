#!/usr/bin/env python3
"""
Pre-installation validation script for vLLM on RTX 5000
Checks system readiness and identifies potential issues before installation
"""

import sys
import subprocess
import shutil
from pathlib import Path

def run_command(cmd, capture_output=True):
    """Run command and return result"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=capture_output, text=True)
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return False, "", str(e)

def check_python():
    """Check Python installation"""
    print("🐍 Checking Python installation...")
    
    version = sys.version_info
    print(f"   Current Python: {version.major}.{version.minor}.{version.micro}")
    
    if version >= (3, 13):
        print("   ⚠️  Python 3.13 detected - may have compatibility issues with vLLM")
        print("   💡 Recommendation: Use Python 3.10 or 3.11 for better compatibility")
    elif version >= (3, 10):
        print("   ✅ Python version is compatible with vLLM")
    else:
        print("   ❌ Python version too old for vLLM (requires 3.8+)")
        return False
    
    # Check for alternative Python versions
    alt_pythons = []
    for py_version in ["python3.11", "python3.10", "python3.9"]:
        if shutil.which(py_version):
            success, stdout, _ = run_command(f"{py_version} --version")
            if success:
                alt_pythons.append((py_version, stdout))
    
    if alt_pythons:
        print("   📋 Alternative Python versions available:")
        for py_cmd, py_ver in alt_pythons:
            print(f"      {py_cmd}: {py_ver}")
    
    return True

def check_gpu():
    """Check GPU and CUDA setup"""
    print("\n🎮 Checking GPU and CUDA...")
    
    # Check nvidia-smi
    success, stdout, stderr = run_command("nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader,nounits")
    if not success:
        print("   ❌ nvidia-smi not working - GPU drivers may not be installed")
        return False
    
    lines = stdout.strip().split('\n')
    for i, line in enumerate(lines):
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 4:
            name, memory, driver, compute_cap = parts[0], parts[1], parts[2], parts[3]
            print(f"   GPU {i}: {name}")
            print(f"      Memory: {memory} MB")
            print(f"      Driver: {driver}")
            print(f"      Compute Capability: {compute_cap}")
            
            # Check if it's RTX 5000
            if "RTX 5000" in name or "Quadro RTX 5000" in name:
                print("   ✅ Quadro RTX 5000 detected - perfect for vLLM!")
                memory_gb = int(memory) / 1024
                if memory_gb >= 15:
                    print(f"   ✅ {memory_gb:.1f} GB VRAM - sufficient for medium-large models")
                else:
                    print(f"   ⚠️  Only {memory_gb:.1f} GB VRAM detected (expected 16GB)")
                
                try:
                    compute_cap_float = float(compute_cap)
                    if compute_cap_float >= 7.5:
                        print("   ✅ Compute capability 7.5+ - supports Tensor Cores")
                    else:
                        print("   ⚠️  Compute capability < 7.5 - may have reduced performance")
                except:
                    print("   ⚠️  Could not parse compute capability")
            else:
                print(f"   ⚠️  GPU is {name}, not RTX 5000 - script optimizations may not apply")
    
    # Check CUDA toolkit
    success, stdout, stderr = run_command("nvcc --version")
    if success:
        print(f"   ✅ CUDA toolkit installed: {stdout.split('release')[1].split(',')[0].strip() if 'release' in stdout else 'version unknown'}")
    else:
        print("   ⚠️  CUDA toolkit (nvcc) not found - will use PyTorch CUDA runtime")
        print("   💡 This is usually fine for vLLM installation")
    
    return True

def check_system_resources():
    """Check system memory and disk space"""
    print("\n💾 Checking system resources...")
    
    # Check RAM
    try:
        success, stdout, stderr = run_command("free -h")
        if success:
            lines = stdout.split('\n')
            for line in lines:
                if 'Mem:' in line:
                    parts = line.split()
                    total_ram = parts[1]
                    available_ram = parts[6] if len(parts) > 6 else parts[3]
                    print(f"   Total RAM: {total_ram}")
                    print(f"   Available RAM: {available_ram}")
                    
                    # Convert to GB for comparison (rough)
                    if 'G' in available_ram:
                        available_gb = float(available_ram.replace('G', ''))
                        if available_gb >= 8:
                            print("   ✅ Sufficient RAM for vLLM installation and operation")
                        else:
                            print("   ⚠️  Low available RAM - may need to close other applications")
    except:
        print("   ⚠️  Could not check RAM usage")
    
    # Check disk space
    success, stdout, stderr = run_command("df -h /")
    if success:
        lines = stdout.split('\n')
        if len(lines) >= 2:
            parts = lines[1].split()
            total_space = parts[1]
            available_space = parts[3]
            print(f"   Disk space: {available_space} available of {total_space} total")
            
            # Rough check for space
            if 'G' in available_space:
                available_gb = float(available_space.replace('G', ''))
                if available_gb >= 20:
                    print("   ✅ Sufficient disk space for vLLM and models")
                else:
                    print("   ⚠️  Low disk space - may need cleanup for large models")

def check_build_tools():
    """Check build tools and compilers"""
    print("\n🔧 Checking build tools...")
    
    tools = [
        ("gcc", "GCC compiler"),
        ("g++", "G++ compiler"),
        ("make", "Make build tool"),
        ("git", "Git version control"),
    ]
    
    all_good = True
    for tool, description in tools:
        if shutil.which(tool):
            success, stdout, stderr = run_command(f"{tool} --version")
            if success:
                version = stdout.split('\n')[0]
                print(f"   ✅ {description}: {version}")
            else:
                print(f"   ⚠️  {description} found but version check failed")
        else:
            print(f"   ❌ {description} not found")
            all_good = False
    
    if not all_good:
        print("   💡 Install missing tools with: sudo apt update && sudo apt install -y build-essential git")
    
    return all_good

def check_package_managers():
    """Check available package managers"""
    print("\n📦 Checking package managers...")
    
    managers = [
        ("pip", "pip --version"),
        ("uv", "uv --version"),
        ("conda", "conda --version"),
    ]
    
    available = []
    for manager, cmd in managers:
        success, stdout, stderr = run_command(cmd)
        if success:
            print(f"   ✅ {manager}: {stdout}")
            available.append(manager)
        else:
            print(f"   ❌ {manager} not available")
    
    if "uv" in available:
        print("   💡 uv detected - will use for faster package installation")
    elif "pip" in available:
        print("   💡 pip available - standard installation method")
    else:
        print("   ❌ No Python package manager found!")
        return False
    
    return True

def check_virtual_env():
    """Check virtual environment capabilities"""
    print("\n🏠 Checking virtual environment setup...")
    
    venvs_dir = Path.home() / "venvs"
    if venvs_dir.exists():
        print(f"   ✅ Virtual environments directory exists: {venvs_dir}")
        existing_venvs = list(venvs_dir.glob("*/"))
        if existing_venvs:
            print(f"   📋 Existing virtual environments: {len(existing_venvs)}")
            for venv in existing_venvs[:5]:  # Show first 5
                print(f"      - {venv.name}")
    else:
        print(f"   📝 Will create virtual environments directory: {venvs_dir}")
    
    # Test venv creation capability
    success, stdout, stderr = run_command("python3 -m venv --help")
    if success:
        print("   ✅ Python venv module available")
    else:
        print("   ❌ Python venv module not available")
        return False
    
    return True

def show_recommendations():
    """Show final recommendations"""
    print("\n🚀 Installation Recommendations:")
    print("   1. Use the provided installation script: ./install_vllm_optimized.sh")
    print("   2. The script will handle Python compatibility issues automatically")
    print("   3. Start with small models (< 1GB) for initial testing")
    print("   4. Use gpu_memory_utilization=0.85 for optimal RTX 5000 usage")
    print("   5. Monitor performance with: python vllm_monitor.py monitor")
    print("\n📋 Optimal Configuration for RTX 5000:")
    print("   - GPU Memory Utilization: 85% (13.6GB for models)")
    print("   - Max Model Length: 4096 tokens")
    print("   - Batch Size: 8 sequences")
    print("   - Enable Flash Attention for memory efficiency")

def main():
    print("🔍 vLLM Installation Readiness Check for RTX 5000")
    print("=" * 60)
    
    checks_passed = 0
    total_checks = 6
    
    if check_python():
        checks_passed += 1
    
    if check_gpu():
        checks_passed += 1
    
    if check_system_resources():
        checks_passed += 1
    
    if check_build_tools():
        checks_passed += 1
    
    if check_package_managers():
        checks_passed += 1
    
    if check_virtual_env():
        checks_passed += 1
    
    print(f"\n📊 Readiness Score: {checks_passed}/{total_checks}")
    
    if checks_passed == total_checks:
        print("✅ System is ready for vLLM installation!")
    elif checks_passed >= 4:
        print("⚠️  System mostly ready - minor issues may be automatically resolved")
    else:
        print("❌ System needs preparation before vLLM installation")
        print("   Please address the issues above before running the installation script")
    
    show_recommendations()

if __name__ == "__main__":
    main()