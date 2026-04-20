#!/usr/bin/env python3
"""
Ollama Diagnostic Tool
======================
Checks Ollama status and helps with troubleshooting.
"""

import requests
import json
import subprocess
import sys

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def check_ollama_status():
    """Check Ollama status"""
    print(f"{Colors.BLUE}Checking Ollama status...{Colors.END}")
    
    try:
        # Test connection
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        
        if response.status_code == 200:
            print(f"{Colors.GREEN}Ollama running on http://localhost:11434{Colors.END}")
            return True
        else:
            print(f"{Colors.RED}Ollama not responding correctly (Status: {response.status_code}){Colors.END}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"{Colors.RED}Ollama is not running - connection failed{Colors.END}")
        return False
    except requests.exceptions.Timeout:
        print(f"{Colors.RED}Ollama timeout{Colors.END}")
        return False
    except Exception as e:
        print(f"{Colors.RED}Error: {str(e)}{Colors.END}")
        return False

def list_installed_models():
    """List installed models"""
    print(f"\n{Colors.BLUE}Installed models:{Colors.END}")
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            
            if models:
                for model in models:
                    name = model.get("name", "Unknown")
                    size = model.get("size", 0)
                    size_gb = size / (1024**3) if size > 0 else 0
                    print(f"  {Colors.GREEN}{name}{Colors.END} ({size_gb:.1f} GB)")
                return [model["name"] for model in models]
            else:
                print(f"  {Colors.YELLOW}No models installed{Colors.END}")
                return []
        else:
            print(f"  {Colors.RED}Error retrieving model list{Colors.END}")
            return []
            
    except Exception as e:
        print(f"  {Colors.RED}Error: {str(e)}{Colors.END}")
        return []

def test_model_download(model_name):
    """Test download of a specific model"""
    print(f"\n{Colors.BLUE}Testing download of {model_name}...{Colors.END}")
    
    try:
        response = requests.post(
            "http://localhost:11434/api/pull",
            json={"name": model_name},
            stream=True,
            timeout=30  # Short timeout for test
        )
        
        if response.status_code == 200:
            print(f"  {Colors.GREEN}Download started{Colors.END}")
            return True
        else:
            print(f"  {Colors.RED}Download error (Status: {response.status_code}){Colors.END}")
            print(f"    Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"  {Colors.RED}Download error: {str(e)}{Colors.END}")
        return False

def manual_install_guide():
    """Guide for manual installation"""
    print(f"\n{Colors.YELLOW}{Colors.BOLD}Manual installation guide:{Colors.END}")
    print(f"\nRun these commands in your terminal:")
    print()
    
    recommended_models = [
        ("mistral:7b-instruct", "~4.1GB", "Stable for German/English"),
        ("qwen2.5:7b-instruct", "~4.7GB", "Good for technical texts"),
        ("llama3.2:3b-instruct", "~2.0GB", "Smaller, for weaker hardware")
    ]
    
    for model, size, desc in recommended_models:
        print(f"{Colors.CYAN}# {desc} ({size}){Colors.END}")
        print(f"ollama pull {model}")
        print()
    
    print(f"{Colors.YELLOW}Recommendation: Start with mistral:7b-instruct (most stable){Colors.END}")

def check_system_resources():
    """Check system resources"""
    print(f"\n{Colors.BLUE}System check:{Colors.END}")
    
    # Free disk space
    import shutil
    free_space_gb = shutil.disk_usage('.').free / (1024**3)
    
    if free_space_gb >= 10:
        print(f"  {Colors.GREEN}Disk space: {free_space_gb:.1f} GB available{Colors.END}")
    else:
        print(f"  {Colors.YELLOW}Disk space: Only {free_space_gb:.1f} GB (recommended: >10GB){Colors.END}")
    
    # Internet connection
    try:
        response = requests.get("https://ollama.ai", timeout=5)
        if response.status_code == 200:
            print(f"  {Colors.GREEN}Internet connection active{Colors.END}")
        else:
            print(f"  {Colors.YELLOW}Internet connection slow{Colors.END}")
    except:
        print(f"  {Colors.RED}No internet connection{Colors.END}")

def recommend_solution():
    """Recommend solution based on status"""
    print(f"\n{Colors.GREEN}{Colors.BOLD}Recommended solution:{Colors.END}")
    
    if not check_ollama_status():
        print("1. Start Ollama:")
        print("   ollama serve")
        print("\n2. Wait until Ollama is running, then install a model:")
        print("   ollama pull mistral:7b-instruct")
        return
    
    models = list_installed_models()
    
    if not models:
        print("Install a model first:")
        print("   ollama pull mistral:7b-instruct")
        print("\nThen restart the chatbot and select the installed model.")
    else:
        print(f"Use one of the installed models:")
        for model in models:
            print(f"  - {model}")
        print("\nStart the chatbot and select an available model.")

def main():
    """Main function"""
    print(f"{Colors.CYAN}{Colors.BOLD}")
    print("=" * 50)
    print("Ollama Diagnostic Tool")
    print("=" * 50)
    print(f"{Colors.END}")
    
    # Comprehensive diagnosis
    ollama_running = check_ollama_status()
    
    if ollama_running:
        models = list_installed_models()
        check_system_resources()
        
        # Test a simple model
        if not models:
            test_model_download("mistral:7b-instruct")
    
    # Provide recommendation
    recommend_solution()
    
    # Manual installation guide
    manual_install_guide()
    
    print(f"\n{Colors.PURPLE}After model installation:{Colors.END}")
    print("python improved_swiss_waste_chatbot_opensource.py")

if __name__ == "__main__":
    main()