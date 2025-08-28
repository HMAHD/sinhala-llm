#!/usr/bin/env python3
"""
setup.py - Complete environment setup for Sinhala LLM training
Expected Runtime: 5-10 minutes
Expected Output: All dependencies installed, environment verified
Author: Akash Hasendra
"""

import os
import sys
import subprocess
import platform
import torch
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SinhalaLLMSetup:
    """Complete setup for Sinhala LLM training environment"""
    
    def __init__(self):
        self.venv_path = Path("/venv/main")
        self.workspace = Path.cwd()
        self.python_exec = sys.executable
        
    def check_system(self):
        """Verify system requirements"""
        logger.info("=" * 60)
        logger.info("🔍 SYSTEM VERIFICATION")
        logger.info("=" * 60)
        
        # Check Python version
        python_version = platform.python_version()
        logger.info(f"✓ Python version: {python_version}")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"✓ CUDA version: {cuda_version}")
            logger.info(f"✓ GPU: {gpu_name}")
            logger.info(f"✓ VRAM: {gpu_memory:.1f}GB")
        else:
            logger.error("✗ No CUDA GPU detected!")
            sys.exit(1)
            
        # Check disk space
        stat = os.statvfs(self.workspace)
        free_space = (stat.f_bavail * stat.f_frsize) / 1e9
        logger.info(f"✓ Free disk space: {free_space:.1f}GB")
        
        if free_space < 40:
            logger.warning(f"⚠️ Low disk space! Recommended: 40GB, Available: {free_space:.1f}GB")
        
        return True
    
    def setup_directories(self):
        """Create necessary directories"""
        logger.info("\n📁 Creating project directories...")
        
        directories = [
            "models",
            "data",
            "outputs",
            "logs",
            "checkpoints",
            "gguf",
            "evaluation_results"
        ]
        
        for dir_name in directories:
            dir_path = self.workspace / dir_name
            dir_path.mkdir(exist_ok=True)
            logger.info(f"  ✓ Created: {dir_name}/")
        
    def install_dependencies(self):
        """Install all required packages"""
        logger.info("\n📦 Installing dependencies...")
        
        # Core packages with specific versions for stability
        packages = [
            # PyTorch with CUDA 12.1 support
            ("torch==2.2.0", "--index-url https://download.pytorch.org/whl/cu121"),
            ("torchvision==0.17.0", "--index-url https://download.pytorch.org/whl/cu121"),
            
            # Unsloth for optimized training
            ("unsloth @ git+https://github.com/unslothai/unsloth.git", ""),
            
            # Training libraries
            ("transformers==4.40.0", ""),
            ("trl==0.8.6", ""),
            ("peft==0.10.0", ""),
            ("accelerate==0.30.0", ""),
            ("bitsandbytes==0.43.0", ""),
            
            # Data processing
            ("datasets==2.19.0", ""),
            ("sentencepiece==0.2.0", ""),
            ("protobuf==4.25.3", ""),
            
            # Monitoring and utilities
            ("wandb==0.16.6", ""),
            ("tensorboard==2.16.2", ""),
            ("tqdm==4.66.2", ""),
            ("psutil==5.9.8", ""),
            ("GPUtil==1.4.0", ""),
            
            # Evaluation
            ("scikit-learn==1.4.2", ""),
            ("rouge-score==0.1.2", ""),
            ("nltk==3.8.1", ""),
        ]
        
        for package, extra_args in packages:
            try:
                cmd = [self.python_exec, "-m", "pip", "install", "--no-cache-dir"]
                cmd.append(package)
                if extra_args:
                    cmd.extend(extra_args.split())
                
                logger.info(f"  Installing {package.split('@')[0].split('==')[0]}...")
                subprocess.run(cmd, check=True, capture_output=True)
                logger.info(f"  ✓ Installed {package.split('@')[0].split('==')[0]}")
            except subprocess.CalledProcessError as e:
                logger.error(f"  ✗ Failed to install {package}: {e}")
                
    def setup_llama_cpp(self):
    """Setup llama.cpp for GGUF conversion - UPDATED FOR CMAKE"""
    logger.info("\n🔧 Setting up llama.cpp for GGUF conversion...")
    
    llama_path = self.workspace / "llama.cpp"
    
    if not llama_path.exists():
        # Clone repository
        logger.info("  Cloning llama.cpp repository...")
        subprocess.run([
            "git", "clone", 
            "https://github.com/ggerganov/llama.cpp",
            str(llama_path)
        ], check=True)
    
    # Install CMake if not present
    logger.info("  Checking for CMake...")
    try:
        subprocess.run(["cmake", "--version"], check=True, capture_output=True)
        logger.info("  ✓ CMake found")
    except:
        logger.info("  Installing CMake...")
        subprocess.run(["apt-get", "update"], check=True, capture_output=True)
        subprocess.run(["apt-get", "install", "-y", "cmake"], check=True)
    
    # Build with CMake (new method)
    logger.info("  Building with CUDA support using CMake...")
    
    build_dir = llama_path / "build"
    build_dir.mkdir(exist_ok=True)
    
    try:
        # Configure with CMake
        logger.info("  Configuring build...")
        subprocess.run([
            "cmake", "..",
            "-DLLAMA_CUDA=ON",
            "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc"
        ], cwd=build_dir, check=True, capture_output=True)
        
        # Build
        logger.info("  Building (this may take a few minutes)...")
        subprocess.run([
            "cmake", "--build", ".", "--config", "Release", "-j", "8"
        ], cwd=build_dir, check=True, capture_output=True)
        
        # Copy binaries to main directory for compatibility
        logger.info("  Setting up binaries...")
        binaries = ["main", "quantize", "convert"]
        for binary in binaries:
            src = build_dir / "bin" / binary
            if src.exists():
                dst = llama_path / binary
                subprocess.run(["cp", str(src), str(dst)], check=True)
        
        logger.info("  ✓ llama.cpp ready for GGUF conversion")
        
    except subprocess.CalledProcessError as e:
        logger.warning("  ⚠️ CMake build failed, trying alternative method...")
        # Fallback to simple build
        self.setup_llama_cpp_simple()
        
    def verify_installation(self):
        """Verify all components are working"""
        logger.info("\n✅ Verifying installation...")
        
        try:
            # Test imports
            import unsloth
            import transformers
            import trl
            import datasets
            import peft
            import accelerate
            
            logger.info("  ✓ All core packages imported successfully")
            
            # Test Unsloth
            from unsloth import FastLanguageModel
            logger.info("  ✓ Unsloth FastLanguageModel ready")
            
            # Test GPU access
            test_tensor = torch.randn(100, 100).cuda()
            logger.info("  ✓ GPU computation test passed")
            
            return True
            
        except Exception as e:
            logger.error(f"  ✗ Verification failed: {e}")
            return False
    
    def save_config(self):
        """Save environment configuration"""
        config = {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "workspace": str(self.workspace),
        }
        
        import json
        with open("environment_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info("\n📄 Configuration saved to environment_config.json")
    
    def run(self):
        """Execute complete setup"""
        logger.info("🚀 SINHALA LLM ENVIRONMENT SETUP")
        logger.info("=" * 60)
        
        # Run all setup steps
        self.check_system()
        self.setup_directories()
        self.install_dependencies()
        self.setup_llama_cpp()
        
        if self.verify_installation():
            self.save_config()
            logger.info("\n" + "=" * 60)
            logger.info("✅ SETUP COMPLETE - Ready for training!")
            logger.info("=" * 60)
            logger.info("\nNext steps:")
            logger.info("1. Run: python data-process.py")
            logger.info("2. Run: python train.py")
            logger.info("3. Run: python evaluation.py")
            logger.info("4. Run: python gguf.py")
        else:
            logger.error("\n❌ Setup failed - Please check errors above")
            sys.exit(1)

if __name__ == "__main__":
    setup = SinhalaLLMSetup()
    setup.run()