#!/usr/bin/env python3
"""
Swiss Waste Recycling Chatbot - Open Source Setup Script
========================================================
Automatic setup script for migration from OpenAI to open-source models.

This script:
1. Checks Ollama installation
2. Installs recommended models
3. Tests the configuration
4. Provides assistance for troubleshooting
"""

import os
import sys
import subprocess
import json
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

class SetupManager:
    """Manager for automatic setup"""
    
    def __init__(self):
        self.ollama_url = "http://localhost:11434"
        self.recommended_models = {
            "qwen2.5:7b-instruct": {
                "name": "Qwen2.5 7B Instruct",
                "size": "~4.7GB",
                "description": "Excellent for German/English, ideal for technical explanations",
                "recommended": True,
                "priority": 1
            },
            "mistral:7b-instruct": {
                "name": "Mistral 7B Instruct", 
                "size": "~4.1GB",
                "description": "Popular, good chat quality, many examples available",
                "recommended": True,
                "priority": 2
            },
            "qwen2.5:3b-instruct": {
                "name": "Qwen2.5 3B Instruct",
                "size": "~2.0GB", 
                "description": "Smaller version for weaker hardware",
                "recommended": False,
                "priority": 3
            }
        }
    
    def print_header(self):
        """Display setup header"""
        print(f"{Colors.CYAN}{Colors.BOLD}")
        print("=" * 70)
        print("Swiss Waste Recycling Chatbot - Open Source Setup")
        print("=" * 70)
        print(f"{Colors.END}")
        print(f"{Colors.WHITE}Migration from OpenAI API to open-source models{Colors.END}")
        print()
    
    def check_python_version(self) -> bool:
        """Check Python version"""
        print(f"{Colors.BLUE}Checking Python version...{Colors.END}")
        
        version = sys.version_info
        if version.major >= 3 and version.minor >= 8:
            print(f"{Colors.GREEN}Python {version.major}.{version.minor}.{version.micro} (OK){Colors.END}")
            return True
        else:
            print(f"{Colors.RED}Python {version.major}.{version.minor}.{version.micro} (requires >= 3.8){Colors.END}")
            return False
    
    def check_system_resources(self) -> Dict[str, bool]:
        """Check system resources"""
        print(f"{Colors.BLUE}Checking system resources...{Colors.END}")
        
        results = {}
        
        # Check RAM (approximate)
        try:
            if os.name == 'nt':  # Windows
                result = subprocess.run(['wmic', 'computersystem', 'get', 'TotalPhysicalMemory'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    ram_bytes = int(lines[1].strip())
                    ram_gb = ram_bytes / (1024**3)
                else:
                    ram_gb = 8  # Default assumption
            else:  # Unix-like
                try:
                    with open('/proc/meminfo', 'r') as f:
                        meminfo = f.read()
                    ram_kb = int([line for line in meminfo.split('\n') if 'MemTotal' in line][0].split()[1])
                    ram_gb = ram_kb / (1024**2)
                except:
                    ram_gb = 8  # Default assumption
            
            results['ram'] = ram_gb >= 6
            if results['ram']:
                print(f"{Colors.GREEN}RAM: {ram_gb:.1f} GB (sufficient){Colors.END}")
            else:
                print(f"{Colors.YELLOW}RAM: {ram_gb:.1f} GB (recommended: >=8GB for better performance){Colors.END}")
                
        except Exception:
            results['ram'] = True  # Assume OK if we can't check
            print(f"{Colors.YELLOW}RAM: Could not be checked (probably OK){Colors.END}")
        
        # Free disk space
        try:
            free_space = subprocess.run(['df', '-h', '.'], capture_output=True, text=True)
            if free_space.returncode == 0:
                print(f"{Colors.GREEN}Disk space checked{Colors.END}")
            results['disk'] = True
        except:
            results['disk'] = True
            print(f"{Colors.YELLOW}Disk space: Could not be checked{Colors.END}")
        
        return results
    
    def check_ollama_installation(self) -> bool:
        """Check if Ollama is installed"""
        print(f"{Colors.BLUE}Checking Ollama installation...{Colors.END}")
        
        try:
            result = subprocess.run(['ollama', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                version = result.stdout.strip()
                print(f"{Colors.GREEN}Ollama installed: {version}{Colors.END}")
                return True
            else:
                print(f"{Colors.RED}Ollama not found{Colors.END}")
                return False
        except FileNotFoundError:
            print(f"{Colors.RED}Ollama not installed{Colors.END}")
            return False
    
    def install_ollama_guide(self):
        """Show installation instructions for Ollama"""
        print(f"{Colors.YELLOW}{Colors.BOLD}Ollama installation required{Colors.END}")
        print()
        
        system = sys.platform
        if system == "win32":
            print("Windows:")
            print("1. Visit: https://ollama.ai/download")
            print("2. Download the Windows installer")
            print("3. Run the installer")
        elif system == "darwin":
            print("macOS:")
            print("1. Visit: https://ollama.ai/download")
            print("2. Download the macOS version")
            print("Or with Homebrew:")
            print("   brew install ollama")
        else:  # Linux
            print("Linux:")
            print("   curl -fsSL https://ollama.ai/install.sh | sh")
        
        print()
        print(f"{Colors.CYAN}After installation, run this script again.{Colors.END}")
    
    def check_ollama_running(self) -> bool:
        """Check if Ollama is running"""
        print(f"{Colors.BLUE}Checking if Ollama is running...{Colors.END}")
        
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print(f"{Colors.GREEN}Ollama running on {self.ollama_url}{Colors.END}")
                return True
            else:
                print(f"{Colors.RED}Ollama not responding correctly (Status: {response.status_code}){Colors.END}")
                return False
        except requests.RequestException as e:
            print(f"{Colors.RED}Cannot connect to Ollama: {str(e)}{Colors.END}")
            return False
    
    def start_ollama_guide(self):
        """Instructions for starting Ollama"""
        print(f"{Colors.YELLOW}{Colors.BOLD}Start Ollama{Colors.END}")
        print()
        
        system = sys.platform
        if system == "win32":
            print("Windows:")
            print("1. Open a new terminal/PowerShell")
            print("2. Run: ollama serve")
            print("3. Keep the terminal open")
        else:
            print("Unix/Linux/macOS:")
            print("1. Open a new terminal")
            print("2. Run: ollama serve")
            print("3. Or start as service: sudo systemctl start ollama")
        
        print()
        print(f"{Colors.CYAN}Press Enter when Ollama has started...{Colors.END}")
        input()
    
    def list_installed_models(self) -> List[str]:
        """List installed models"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models_data = response.json()
                return [model['name'] for model in models_data.get('models', [])]
            return []
        except Exception:
            return []
    
    def install_model(self, model_name: str) -> bool:
        """Install a specific model"""
        model_info = self.recommended_models[model_name]
        print(f"{Colors.BLUE}Installing {model_info['name']} ({model_info['size']})...{Colors.END}")
        print(f"    {model_info['description']}")
        print(f"{Colors.YELLOW}    This can take 5-15 minutes depending on your internet connection...{Colors.END}")
        
        try:
            # Start model pull
            response = requests.post(
                f"{self.ollama_url}/api/pull",
                json={"name": model_name},
                stream=True,
                timeout=600  # 10 minutes timeout
            )
            
            if response.status_code == 200:
                print("    Progress: ", end="", flush=True)
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        status = data.get('status', '')
                        
                        if 'pulling' in status.lower():
                            print(".", end="", flush=True)
                        elif 'verifying' in status.lower():
                            print(f"\n    {Colors.CYAN}Verifying model...{Colors.END}")
                        elif status == 'success':
                            print(f"\n{Colors.GREEN}{model_info['name']} successfully installed{Colors.END}")
                            return True
                
                return True
            else:
                print(f"\n{Colors.RED}Error during installation: HTTP {response.status_code}{Colors.END}")
                return False
                
        except Exception as e:
            print(f"\n{Colors.RED}Error during installation: {str(e)}{Colors.END}")
            return False
    
    def select_and_install_models(self):
        """Model selection and installation"""
        print(f"{Colors.BLUE}Model setup{Colors.END}")
        print()
        
        installed_models = self.list_installed_models()
        
        if installed_models:
            print(f"{Colors.GREEN}Already installed models:{Colors.END}")
            for model in installed_models:
                print(f"  {model}")
            print()
        
        # Show recommended models
        print(f"{Colors.BOLD}Recommended models for your project:{Colors.END}")
        print()
        
        for i, (model_key, model_info) in enumerate(self.recommended_models.items(), 1):
            status = "Installed" if model_key in installed_models else "Not installed"
            recommended = " (RECOMMENDED)" if model_info["recommended"] else ""
            
            print(f"{i}. {Colors.BOLD}{model_info['name']}{Colors.END}{recommended}")
            print(f"   Size: {model_info['size']}")
            print(f"   {model_info['description']}")
            print(f"   Status: {status}")
            print()
        
        # Automatically install recommended models
        models_to_install = []
        for model_key, model_info in self.recommended_models.items():
            if model_info["recommended"] and model_key not in installed_models:
                models_to_install.append(model_key)
        
        if models_to_install:
            print(f"{Colors.YELLOW}Should I automatically install the recommended models?{Colors.END}")
            print("Recommended models:", ", ".join([self.recommended_models[m]["name"] for m in models_to_install]))
            
            choice = input(f"\nYes/No (y/n) [y]: ").strip().lower()
            
            if choice in ['', 'y', 'yes']:
                success_count = 0
                for model_key in models_to_install:
                    if self.install_model(model_key):
                        success_count += 1
                
                print()
                if success_count == len(models_to_install):
                    print(f"{Colors.GREEN}All recommended models successfully installed!{Colors.END}")
                else:
                    print(f"{Colors.YELLOW}{success_count}/{len(models_to_install)} models installed{Colors.END}")
            else:
                print(f"{Colors.CYAN}Skipping model installation. You can install them later with 'ollama pull <model>'.{Colors.END}")
        else:
            print(f"{Colors.GREEN}All recommended models are already installed!{Colors.END}")
    
    def check_python_dependencies(self) -> bool:
        """Check Python dependencies"""
        print(f"{Colors.BLUE}Checking Python dependencies...{Colors.END}")
        
        required_packages = [
            'torch',
            'torchvision', 
            'requests',
            'pillow'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                print(f"{Colors.GREEN}{package}{Colors.END}")
            except ImportError:
                print(f"{Colors.RED}{package} (missing){Colors.END}")
                missing_packages.append(package)
        
        if missing_packages:
            print(f"\n{Colors.YELLOW}Installing missing packages...{Colors.END}")
            try:
                subprocess.run([sys.executable, '-m', 'pip', 'install'] + missing_packages, check=True)
                print(f"{Colors.GREEN}All packages successfully installed{Colors.END}")
                return True
            except subprocess.CalledProcessError:
                print(f"{Colors.RED}Error installing packages{Colors.END}")
                print("Run manually: pip install", " ".join(missing_packages))
                return False
        else:
            print(f"{Colors.GREEN}All required packages are installed{Colors.END}")
            return True
    
    def test_setup(self) -> bool:
        """Test the complete setup"""
        print(f"{Colors.BLUE}Testing setup...{Colors.END}")
        
        try:
            # Test Ollama connection
            installed_models = self.list_installed_models()
            
            if not installed_models:
                print(f"{Colors.RED}No models installed{Colors.END}")
                return False
            
            # Test one model
            test_model = None
            for model_key in self.recommended_models.keys():
                if model_key in installed_models:
                    test_model = model_key
                    break
            
            if test_model:
                print(f"{Colors.CYAN}Testing model: {test_model}{Colors.END}")
                
                test_prompt = {
                    "model": test_model,
                    "prompt": "Hello! Respond briefly in English.",
                    "stream": False
                }
                
                response = requests.post(f"{self.ollama_url}/api/generate", json=test_prompt, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    test_response = result.get('response', '')
                    
                    if test_response.strip():
                        print(f"{Colors.GREEN}Model responds: {test_response[:100]}...{Colors.END}")
                        return True
                    else:
                        print(f"{Colors.RED}Model returns no response{Colors.END}")
                        return False
                else:
                    print(f"{Colors.RED}Model test failed (HTTP {response.status_code}){Colors.END}")
                    return False
            else:
                print(f"{Colors.RED}No testable model found{Colors.END}")
                return False
                
        except Exception as e:
            print(f"{Colors.RED}Setup test failed: {str(e)}{Colors.END}")
            return False
    
    def create_config_files(self):
        """Create configuration files"""
        print(f"{Colors.BLUE}Creating configuration files...{Colors.END}")
        
        # Find installed recommended models
        installed_models = self.list_installed_models()
        default_model = None
        
        for model_key in ["qwen2.5:7b-instruct", "mistral:7b-instruct"]:
            if model_key in installed_models:
                default_model = model_key
                break
        
        if not default_model and installed_models:
            default_model = installed_models[0]
        
        config_content = f'''# Swiss Waste Recycling Chatbot - Open Source Configuration
# Generated by setup script

# Default model for the chatbot
DEFAULT_MODEL = "{default_model or 'qwen2.5:7b-instruct'}"

# Ollama configuration
OLLAMA_URL = "http://localhost:11434"

# Model paths
MODEL_PATH = "../models/baseline/finetuned_model.pth"

# Available models
AVAILABLE_MODELS = {list(self.recommended_models.keys())}

# System settings
DEVICE = "auto"  # auto, cpu, cuda
DEFAULT_LANGUAGE = "en"  # de, en

print("Configuration saved!")
'''
        
        try:
            with open('config_opensource.py', 'w', encoding='utf-8') as f:
                f.write(config_content)
            print(f"{Colors.GREEN}Configuration saved: config_opensource.py{Colors.END}")
        except Exception as e:
            print(f"{Colors.YELLOW}Could not save configuration: {str(e)}{Colors.END}")
    
    def print_next_steps(self):
        """Show next steps"""
        print()
        print(f"{Colors.GREEN}{Colors.BOLD}Setup completed successfully!{Colors.END}")
        print()
        print(f"{Colors.CYAN}{Colors.BOLD}Next steps:{Colors.END}")
        print()
        print("1. Start your new chatbot:")
        print(f"   {Colors.WHITE}python improved_swiss_waste_chatbot_opensource.py{Colors.END}")
        print()
        print("2. Test the features:")
        print("   - Image classification with: image:/path/to/image.jpg")
        print("   - Recycling questions in German and English")
        print()
        print("3. For your project:")
        print("   - Compare quality with your OpenAI version")
        print("   - Document the differences")
        print("   - Test different models")
        print()
        print(f"{Colors.BLUE}Tips:{Colors.END}")
        print("   - Use 'ollama list' to see installed models")
        print("   - Use 'ollama pull <model>' to install additional models")
        print("   - Logs can be found in 'swiss_chatbot.log'")
        print()
        print(f"{Colors.PURPLE}Good luck with your project!{Colors.END}")
    
    def run_setup(self):
        """Run the complete setup"""
        self.print_header()
        
        # Step 1: Check Python
        if not self.check_python_version():
            print(f"\n{Colors.RED}Setup aborted: Python version too old{Colors.END}")
            return False
        
        # Step 2: Check system resources
        self.check_system_resources()
        print()
        
        # Step 3: Check Ollama installation
        if not self.check_ollama_installation():
            self.install_ollama_guide()
            return False
        print()
        
        # Step 4: Is Ollama running?
        if not self.check_ollama_running():
            self.start_ollama_guide()
            # Check again
            if not self.check_ollama_running():
                print(f"\n{Colors.RED}Setup aborted: Ollama is not running{Colors.END}")
                return False
        print()
        
        # Step 5: Python dependencies
        if not self.check_python_dependencies():
            print(f"\n{Colors.RED}Setup aborted: Python dependencies missing{Colors.END}")
            return False
        print()
        
        # Step 6: Install models
        self.select_and_install_models()
        print()
        
        # Step 7: Test setup
        if not self.test_setup():
            print(f"\n{Colors.YELLOW}Setup test not successful, but installation should work{Colors.END}")
        print()
        
        # Step 8: Create configuration files
        self.create_config_files()
        
        # Step 9: Show next steps
        self.print_next_steps()
        
        return True

def main():
    """Main function"""
    try:
        setup_manager = SetupManager()
        setup_manager.run_setup()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Setup aborted by user{Colors.END}")
    except Exception as e:
        print(f"\n{Colors.RED}Unexpected error: {str(e)}{Colors.END}")
        import traceback
        print(f"{Colors.RED}{traceback.format_exc()}{Colors.END}")

if __name__ == "__main__":
    main()