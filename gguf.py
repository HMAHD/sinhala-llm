#!/usr/bin/env python3
"""
GGUF Converter - Fixed for Your Sinhala LLM Setup
"""

import os
import subprocess
import logging
import json
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ConversionConfig:
    """Configuration for GGUF conversion"""
    model_path: Path
    output_dir: Path
    quantization: str = "Q4_K_M"
    min_free_gb: float = 10.0
    cleanup_on_success: bool = True
    test_prompt: str = "ශ්‍රී ලංකාව"
    
    def __post_init__(self):
        self.model_path = Path(self.model_path)
        self.output_dir = Path(self.output_dir)
        
        # Validate paths exist
        if not self.model_path.exists():
            raise ValueError(f"Model path does not exist: {self.model_path}")
        
        # Create output dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

class DiskSpaceManager:
    """Manage disk space checks and cleanup"""
    
    @staticmethod
    def get_free_space_gb(path: Path = Path(".")) -> float:
        """Get free space in GB"""
        stat = os.statvfs(path)
        return (stat.f_bavail * stat.f_frsize) / 1e9
    
    @staticmethod
    def estimate_required_space(model_path: Path, quantization: str) -> float:
        """Estimate space needed for conversion"""
        try:
            model_size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
            model_gb = model_size / 1e9
            
            # More conservative estimates for safety
            multipliers = {
                "Q4_K_M": 2.5,  # Need more space for intermediate files
                "Q5_K_M": 2.8,
                "Q8_0": 3.0,
                "F16": 2.2
            }
            
            required = model_gb * multipliers.get(quantization, 3.0)
            logger.info(f"Model size: {model_gb:.2f}GB, Estimated requirement: {required:.2f}GB")
            return required
            
        except Exception as e:
            logger.warning(f"Could not estimate space: {e}")
            return 15.0  # Safe fallback
    
    def ensure_space(self, required_gb: float, path: Path = Path(".")) -> bool:
        """Ensure sufficient space is available"""
        free_gb = self.get_free_space_gb(path)
        logger.info(f"Available space: {free_gb:.1f}GB, Required: {required_gb:.1f}GB")
        
        if free_gb < required_gb:
            logger.error(f"Insufficient space. Need {required_gb:.1f}GB, have {free_gb:.1f}GB")
            logger.error("Free up space or use a different output directory")
            return False
        return True

class LlamaCppInterface:
    """Interface to llama.cpp tools with your specific setup"""
    
    def __init__(self, llama_cpp_path: Path = None):
        # Auto-detect llama.cpp location
        if llama_cpp_path is None:
            candidates = [
                Path("/workspace/sriai/llama.cpp"),
                Path("llama.cpp"),
                Path("../llama.cpp"),
                Path.cwd() / "llama.cpp"
            ]
            
            for candidate in candidates:
                if candidate.exists():
                    self.llama_cpp_path = candidate
                    break
            else:
                raise RuntimeError("llama.cpp not found. Install it first:\n"
                                 "git clone https://github.com/ggerganov/llama.cpp\n"
                                 "cd llama.cpp && make")
        else:
            self.llama_cpp_path = Path(llama_cpp_path)
        
        logger.info(f"Using llama.cpp at: {self.llama_cpp_path}")
        self._validate_installation()
    
    def _validate_installation(self):
        """Validate llama.cpp installation"""
        if not self.llama_cpp_path.exists():
            raise RuntimeError(f"llama.cpp directory not found: {self.llama_cpp_path}")
        
        # Check for conversion scripts (try multiple names)
        conversion_scripts = [
            "convert-hf-to-gguf.py",
            "convert_hf_to_gguf.py", 
            "convert.py"
        ]
        
        found_converter = False
        for script in conversion_scripts:
            if (self.llama_cpp_path / script).exists():
                found_converter = True
                logger.info(f"Found conversion script: {script}")
                break
        
        if not found_converter:
            logger.warning("No conversion scripts found. This might cause issues.")
            logger.warning("Try: cd llama.cpp && git pull")
    
    def _run_command(self, cmd: List[str], timeout: int = 1800, cwd: Path = None) -> subprocess.CompletedProcess:
        """Run command with proper error handling"""
        if cwd is None:
            cwd = self.llama_cpp_path
            
        logger.info(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.stdout:
                logger.debug(f"STDOUT: {result.stdout[:500]}...")
            if result.stderr and result.returncode != 0:
                logger.error(f"STDERR: {result.stderr[:500]}...")
                
            return result
            
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Command timed out after {timeout}s: {' '.join(cmd)}")
    
    def _build_if_needed(self, target: str) -> bool:
        """Build llama.cpp target if needed"""
        exe_path = self.llama_cpp_path / target
        
        if exe_path.exists():
            logger.info(f"{target} already exists")
            return True
            
        logger.info(f"Building {target}...")
        
        # Try make first, then cmake if available
        build_commands = [
            ["make", target],
            ["make", "-j4", target],  # Parallel build
        ]
        
        for cmd in build_commands:
            try:
                result = self._run_command(cmd, timeout=600)
                if result.returncode == 0:
                    logger.info(f"Successfully built {target}")
                    return True
            except Exception as e:
                logger.warning(f"Build attempt failed: {e}")
                continue
        
        logger.error(f"Failed to build {target}")
        return False
    
    def convert_to_f16(self, model_path: Path, output_path: Path) -> bool:
        """Convert model to F16 GGUF - try multiple methods"""
        
        # Method 1: Try convert-hf-to-gguf.py (newest)
        if self._try_hf_to_gguf_converter(model_path, output_path):
            return True
            
        # Method 2: Try legacy convert.py  
        if self._try_legacy_converter(model_path, output_path):
            return True
            
        # Method 3: Last resort - try with different python
        if self._try_alternate_python(model_path, output_path):
            return True
            
        logger.error("All conversion methods failed")
        return False
    
    def _try_hf_to_gguf_converter(self, model_path: Path, output_path: Path) -> bool:
        """Try convert-hf-to-gguf.py"""
        script_candidates = ["convert-hf-to-gguf.py", "convert_hf_to_gguf.py"]
        
        for script_name in script_candidates:
            script_path = self.llama_cpp_path / script_name
            if not script_path.exists():
                continue
                
            logger.info(f"Trying {script_name}...")
            
            cmd = [
                "python3", str(script_path),
                str(model_path),
                "--outfile", str(output_path),
                "--outtype", "f16"
            ]
            
            try:
                result = self._run_command(cmd)
                if result.returncode == 0 and output_path.exists():
                    logger.info(f"✅ Conversion successful with {script_name}")
                    return True
                else:
                    logger.warning(f"❌ {script_name} failed: {result.stderr[:200]}...")
            except Exception as e:
                logger.warning(f"❌ {script_name} error: {e}")
        
        return False
    
    def _try_legacy_converter(self, model_path: Path, output_path: Path) -> bool:
        """Try legacy convert.py"""
        script_path = self.llama_cpp_path / "convert.py"
        if not script_path.exists():
            return False
            
        logger.info("Trying legacy convert.py...")
        
        cmd = [
            "python3", str(script_path),
            str(model_path),
            "--outtype", "f16",
            "--outfile", str(output_path)
        ]
        
        try:
            result = self._run_command(cmd)
            if result.returncode == 0 and output_path.exists():
                logger.info("✅ Legacy converter successful")
                return True
            else:
                logger.warning(f"❌ Legacy converter failed: {result.stderr[:200]}...")
        except Exception as e:
            logger.warning(f"❌ Legacy converter error: {e}")
        
        return False
    
    def _try_alternate_python(self, model_path: Path, output_path: Path) -> bool:
        """Try with python instead of python3"""
        script_path = self.llama_cpp_path / "convert-hf-to-gguf.py"
        if not script_path.exists():
            script_path = self.llama_cpp_path / "convert.py"
            if not script_path.exists():
                return False
        
        logger.info("Trying with alternate python...")
        
        cmd = [
            "python", str(script_path),
            str(model_path),
            "--outfile", str(output_path),
            "--outtype", "f16"
        ]
        
        try:
            result = self._run_command(cmd)
            if result.returncode == 0 and output_path.exists():
                logger.info("✅ Alternate python successful")
                return True
        except Exception as e:
            logger.warning(f"❌ Alternate python failed: {e}")
        
        return False
    
    def quantize(self, input_path: Path, output_path: Path, quantization: str) -> bool:
        """Quantize GGUF model"""
        if not self._build_if_needed("quantize"):
            raise RuntimeError("Failed to build quantize tool")
        
        cmd = [
            str(self.llama_cpp_path / "quantize"),
            str(input_path),
            str(output_path),
            quantization
        ]
        
        try:
            result = self._run_command(cmd)
            success = result.returncode == 0 and output_path.exists()
            
            if success:
                logger.info(f"✅ Quantization to {quantization} successful")
            else:
                logger.error(f"❌ Quantization failed: {result.stderr}")
                
            return success
            
        except Exception as e:
            logger.error(f"❌ Quantization error: {e}")
            return False
    
    def test_model(self, model_path: Path, prompt: str, max_tokens: int = 50) -> Optional[str]:
        """Test GGUF model"""
        if not self._build_if_needed("main"):
            logger.warning("Could not build main - skipping test")
            return "BUILD_FAILED"
        
        cmd = [
            str(self.llama_cpp_path / "main"),
            "-m", str(model_path),
            "-p", prompt,
            "-n", str(max_tokens),
            "--temp", "0.7",
            "-ngl", "0",  # CPU-only for compatibility
            "--no-display-prompt"
        ]
        
        try:
            result = self._run_command(cmd, timeout=120)
            
            if result.returncode == 0:
                output = result.stdout.strip()
                logger.info("✅ Model test successful")
                return output
            else:
                logger.warning(f"⚠️ Model test failed: {result.stderr[:200]}...")
                return None
                
        except Exception as e:
            logger.warning(f"⚠️ Model test error: {e}")
            return None

class GGUFConverter:
    """Main converter with improved error handling"""
    
    def __init__(self, config: ConversionConfig):
        self.config = config
        self.disk_manager = DiskSpaceManager()
        self.llama = LlamaCppInterface()
        self.stats: Dict[str, Any] = {}
    
    def convert(self) -> Path:
        """Run conversion pipeline with better error recovery"""
        logger.info("=" * 60)
        logger.info("🚀 GGUF CONVERSION PIPELINE")
        logger.info("=" * 60)
        
        try:
            # Step 1: Validate everything
            self._validate_environment()
            
            # Step 2: Convert to F16
            f16_path = self.config.output_dir / "model_f16.gguf"
            final_path = self.config.output_dir / f"sinhala_llm_{self.config.quantization.lower()}.gguf"
            
            logger.info(f"\n📝 Converting {self.config.model_path} to F16...")
            
            if not self.llama.convert_to_f16(self.config.model_path, f16_path):
                raise RuntimeError("❌ F16 conversion failed with all methods")
            
            if not f16_path.exists():
                raise RuntimeError("❌ F16 file was not created")
                
            self.stats['f16_size_gb'] = f16_path.stat().st_size / 1e9
            logger.info(f"✅ F16 conversion complete: {self.stats['f16_size_gb']:.2f}GB")
            
            # Step 3: Quantize
            logger.info(f"\n⚡ Quantizing to {self.config.quantization}...")
            
            if not self.llama.quantize(f16_path, final_path, self.config.quantization):
                raise RuntimeError(f"❌ Quantization to {self.config.quantization} failed")
            
            if not final_path.exists():
                raise RuntimeError("❌ Quantized file was not created")
                
            self.stats['quantized_size_gb'] = final_path.stat().st_size / 1e9
            compression_ratio = (self.stats['f16_size_gb'] - self.stats['quantized_size_gb']) / self.stats['f16_size_gb']
            
            logger.info(f"✅ Quantization complete: {self.stats['quantized_size_gb']:.2f}GB")
            logger.info(f"📊 Compression: {compression_ratio*100:.1f}% size reduction")
            
            # Step 4: Cleanup intermediate file
            if self.config.cleanup_on_success and f16_path.exists():
                logger.info("🧹 Cleaning up intermediate F16 file...")
                f16_path.unlink()
            
            # Step 5: Test model
            logger.info(f"\n🧪 Testing model with prompt: '{self.config.test_prompt}'")
            test_result = self.llama.test_model(final_path, self.config.test_prompt)
            
            if test_result and test_result != "BUILD_FAILED":
                logger.info("✅ Model test successful!")
                logger.info(f"Sample output: {test_result[:150]}...")
                self.stats['test_passed'] = True
            else:
                logger.warning("⚠️ Model test failed, but conversion completed")
                self.stats['test_passed'] = False
            
            # Step 6: Save metadata
            self._save_metadata(final_path)
            
            logger.info("\n" + "=" * 60)
            logger.info(f"✨ CONVERSION SUCCESSFUL!")
            logger.info(f"📁 Output: {final_path}")
            logger.info(f"📊 Final size: {self.stats['quantized_size_gb']:.2f}GB")
            logger.info(f"🎯 Ready for: Ollama, LM Studio, etc.")
            logger.info("=" * 60)
            
            return final_path
            
        except Exception as e:
            logger.error(f"\n❌ CONVERSION FAILED: {e}")
            self._cleanup_on_failure()
            raise
    
    def _validate_environment(self):
        """Comprehensive environment validation"""
        logger.info("🔍 Validating environment...")
        
        # Check disk space
        required_gb = self.disk_manager.estimate_required_space(
            self.config.model_path,
            self.config.quantization
        )
        
        if not self.disk_manager.ensure_space(required_gb + self.config.min_free_gb):
            raise RuntimeError("Insufficient disk space")
        
        # Check model files - be flexible with file names
        model_files = list(self.config.model_path.rglob("*"))
        file_names = [f.name for f in model_files if f.is_file()]
        
        logger.info(f"Found model files: {file_names}")
        
        # Must have config and model weights
        has_config = any("config.json" in name for name in file_names)
        has_weights = any(name.endswith(('.safetensors', '.bin', '.pth')) for name in file_names)
        
        if not has_config:
            raise ValueError("No config.json found in model directory")
        if not has_weights:
            raise ValueError("No model weight files found (.safetensors, .bin, .pth)")
        
        # Check tokenizer files (warn if missing)
        has_tokenizer = any("tokenizer" in name for name in file_names)
        if not has_tokenizer:
            logger.warning("⚠️ No tokenizer files found - this might cause issues")
        
        logger.info("✅ Environment validation passed")
    
    def _save_metadata(self, output_path: Path):
        """Save conversion metadata"""
        metadata = {
            "source_model": str(self.config.model_path),
            "quantization": self.config.quantization,
            "output_file": str(output_path),
            "llama_cpp_path": str(self.llama.llama_cpp_path),
            "stats": self.stats,
            "usage_instructions": {
                "ollama": f"ollama create sinhala-llm -f <(echo 'FROM {output_path}')",
                "lm_studio": f"Copy {output_path} to LM Studio models folder",
                "llama_cpp": f"./main -m {output_path} -p 'your prompt here'"
            }
        }
        
        metadata_path = output_path.parent / f"{output_path.stem}_info.json"
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 Metadata saved: {metadata_path}")
    
    def _cleanup_on_failure(self):
        """Clean up files if conversion fails"""
        cleanup_patterns = [
            "model_f16.gguf",
            "*_temp.gguf",
            "*.tmp"
        ]
        
        for pattern in cleanup_patterns:
            for file in self.config.output_dir.glob(pattern):
                try:
                    file.unlink()
                    logger.info(f"🧹 Cleaned up: {file}")
                except Exception:
                    pass

def main():
    """Main entry point - UPDATED FOR YOUR SETUP"""
    
    # Your actual model path
    model_path = Path("/workspace/sriai/models/sinhala_llm")
    
    # Verify the model exists
    if not model_path.exists():
        logger.error(f"❌ Model not found at: {model_path}")
        logger.error("Available models:")
        models_dir = Path("/workspace/sriai/models")
        if models_dir.exists():
            for item in models_dir.iterdir():
                if item.is_dir():
                    logger.error(f"  - {item}")
        return 1
    
    # Create output directory
    output_dir = Path("/workspace/sriai/gguf")
    output_dir.mkdir(exist_ok=True)
    
    # Configuration
    config = ConversionConfig(
        model_path=model_path,
        output_dir=output_dir,
        quantization="Q4_K_M",  # Good balance of size/quality
        min_free_gb=5.0,
        cleanup_on_success=True,
        test_prompt="ශ්‍රී ලංකාව යනු ලස්සන රටකි"
    )
    
    try:
        converter = GGUFConverter(config)
        output_path = converter.convert()
        
        logger.info(f"\n🎉 SUCCESS! Your Sinhala LLM is ready:")
        logger.info(f"📁 Location: {output_path}")
        logger.info(f"💾 Size: {output_path.stat().st_size / 1e9:.2f}GB")
        logger.info(f"\n💡 Next steps:")
        logger.info(f"  1. Test with: cd llama.cpp && ./main -m {output_path} -p 'ඔබේ prompt එක'")
        logger.info(f"  2. Use with Ollama, LM Studio, or other GGUF-compatible tools")
        
        return 0
        
    except Exception as e:
        logger.error(f"\n💥 CONVERSION FAILED!")
        logger.error(f"Error: {e}")
        logger.error(f"\n🔧 Troubleshooting:")
        logger.error(f"  1. Check disk space: df -h")
        logger.error(f"  2. Verify llama.cpp: ls -la /workspace/sriai/llama.cpp/")
        logger.error(f"  3. Check model files: ls -la {model_path}/")
        logger.error(f"  4. Update llama.cpp: cd llama.cpp && git pull && make")
        
        return 1

if __name__ == "__main__":
    exit(main())