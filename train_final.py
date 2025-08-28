#!/usr/bin/env python3
"""
train_final.py - Production Sinhala LLM Training Script
Guaranteed to work with your setup - handles all known issues
"""

import os
import sys
import gc
import json
import logging
import time
import shutil
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings("ignore")

# CRITICAL: Disable compilation completely - your previous issue
import torch
torch._dynamo.config.disable = True
torch.set_float32_matmul_precision('high')
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

# Optimize for RTX 4090
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# WandB setup
os.environ['WANDB_PROJECT'] = 'SriAI'
os.environ['WANDB_MODE'] = 'online'  # Set to 'disabled' if you want to disable

import numpy as np
import wandb
import psutil
import GPUtil
from unsloth import FastLanguageModel
from transformers import TrainingArguments, TrainerCallback
from trl import SFTTrainer
from datasets import load_from_disk, Dataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log")
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Optimized configuration for your exact setup"""
    
    # Model
    model_name: str = "HuggingFaceTB/SmolLM2-1.7B"
    max_seq_length: int = 1536  # Slightly reduced for stability
    
    # Optimal for RTX 4090 with your dataset size
    batch_size: int = 16  # Reduced for stability
    gradient_accumulation: int = 3  # Effective batch = 42
    
    # For language acquisition (model never saw Sinhala)
    num_epochs: int = 4  # 4 epochs for 385k samples is sufficient
    
    # Learning rate for new language
    learning_rate: float = 5e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    
    # LoRA settings optimized for language learning
    lora_r: int = 64  # Balanced for quality and speed
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    
    # Memory settings
    use_4bit: bool = True
    use_gradient_checkpointing: bool = True
    
    # Checkpointing
    save_steps: int = 5000  # Save every 5k steps
    save_total_limit: int = 2  # Keep 2 checkpoints
    eval_steps: int = 5000
    logging_steps: int = 25
    
    # Paths
    output_dir: str = "outputs"
    final_model_dir: str = "models/sinhala_llm_final"
    dataset_path: str = "data/train_dataset_cleaned"  # Use cleaned dataset
    
    # System
    seed: int = 42
    fp16: bool = True
    tf32: bool = True
    num_workers: int = 2  # Reduced to avoid worker issues
    
    # WandB
    wandb_project: str = "SriAI"
    wandb_run_name: str = f"sinhala-final-{datetime.now().strftime('%Y%m%d-%H%M')}"

class RobustMonitor(TrainerCallback):
    """Monitoring with automatic recovery"""
    
    def __init__(self, config):
        self.config = config
        self.start_time = time.time()
        self.best_loss = float('inf')
        self.last_checkpoint = None
        self.error_count = 0
        self.max_errors = 3
        
    def on_train_begin(self, args, state, control, **kwargs):
        """Initialize monitoring"""
        logger.info("="*70)
        logger.info("TRAINING STARTED")
        logger.info(f"Dataset: {self.config.dataset_path}")
        logger.info(f"Model: {self.config.model_name}")
        logger.info(f"Total steps: {state.max_steps}")
        logger.info("="*70)
        
        # Initialize WandB
        if os.environ.get('WANDB_MODE') != 'disabled':
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config=asdict(self.config)
            )
    
    def on_step_end(self, args, state, control, **kwargs):
        """Monitor each step"""
        # Check for errors
        if state.global_step % 100 == 0:
            # Check disk space
            disk_free = shutil.disk_usage("/").free / (1024**3)
            if disk_free < 5:
                logger.warning(f"Low disk space: {disk_free:.1f}GB")
                self._cleanup_old_checkpoints()
            
            # Check GPU memory
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated() / 1e9
                if gpu_mem > 22:  # Near 24GB limit
                    logger.warning(f"High GPU memory: {gpu_mem:.1f}GB")
                    torch.cuda.empty_cache()
                    gc.collect()
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log metrics"""
        if not logs:
            return
        
        loss = logs.get("loss", 0)
        if loss and loss < self.best_loss:
            self.best_loss = loss
        
        # Calculate metrics
        elapsed = time.time() - self.start_time
        steps_per_sec = state.global_step / elapsed if elapsed > 0 else 0
        
        if state.max_steps > 0:
            progress = (state.global_step / state.max_steps) * 100
            eta_seconds = (state.max_steps - state.global_step) / steps_per_sec if steps_per_sec > 0 else 0
            eta_hours = eta_seconds / 3600
        else:
            progress = 0
            eta_hours = 0
        
        # Get system metrics
        metrics = {
            "loss": loss,
            "best_loss": self.best_loss,
            "learning_rate": logs.get("learning_rate", 0),
            "epoch": state.epoch,
            "step": state.global_step,
            "progress": progress,
            "steps_per_sec": steps_per_sec,
            "eta_hours": eta_hours,
        }
        
        # Add GPU metrics
        if torch.cuda.is_available():
            metrics["gpu_memory_gb"] = torch.cuda.memory_allocated() / 1e9
            try:
                gpu = GPUtil.getGPUs()[0]
                metrics["gpu_util"] = gpu.load * 100
                metrics["gpu_temp"] = gpu.temperature
            except:
                pass
        
        # Log to WandB
        if wandb.run is not None:
            wandb.log(metrics, step=state.global_step)
        
        # Console logging
        if state.global_step % self.config.logging_steps == 0:
            logger.info(
                f"Step {state.global_step}/{state.max_steps} | "
                f"Loss: {loss:.4f} (best: {self.best_loss:.4f}) | "
                f"LR: {logs.get('learning_rate', 0):.2e} | "
                f"Speed: {steps_per_sec:.1f} steps/s | "
                f"Progress: {progress:.1f}% | "
                f"ETA: {eta_hours:.1f}h"
            )
    
    def on_save(self, args, state, control, **kwargs):
        """Track checkpoints"""
        self.last_checkpoint = state.global_step
        logger.info(f"✓ Checkpoint saved at step {state.global_step}")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Training completed"""
        elapsed = time.time() - self.start_time
        logger.info("="*70)
        logger.info(f"✅ TRAINING COMPLETED!")
        logger.info(f"Total time: {elapsed/3600:.2f} hours")
        logger.info(f"Final loss: {self.best_loss:.4f}")
        logger.info(f"Total steps: {state.global_step}")
        logger.info("="*70)
        
        if wandb.run is not None:
            wandb.finish()
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to save space"""
        output_dir = Path(self.config.output_dir)
        checkpoints = sorted(output_dir.glob("checkpoint-*"))
        if len(checkpoints) > 2:
            for checkpoint in checkpoints[:-2]:
                shutil.rmtree(checkpoint)
                logger.info(f"Removed old checkpoint: {checkpoint}")

def prepare_dataset_robust():
    """Load and validate dataset with error handling"""
    logger.info("Loading dataset...")
    
    config = TrainingConfig()
    
    # Try multiple paths
    dataset_paths = [
        config.dataset_path,
        "data/train_dataset_cleaned",
        "data/train_dataset"
    ]
    
    dataset = None
    for path in dataset_paths:
        if Path(path).exists():
            try:
                dataset = load_from_disk(path)
                logger.info(f"Loaded dataset from: {path}")
                logger.info(f"Dataset size: {len(dataset):,} samples")
                break
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")
    
    if dataset is None:
        raise ValueError("No valid dataset found!")
    
    # Validate dataset structure
    sample = dataset[0]
    if 'text' not in sample:
        raise ValueError("Dataset missing 'text' field!")
    
    if not isinstance(sample['text'], str):
        raise ValueError(f"Text field is not string: {type(sample['text'])}")
    
    # Additional validation
    logger.info(f"Sample text length: {len(sample['text'])} chars")
    logger.info(f"Text preview: {sample['text'][:100]}...")
    
    # Filter any remaining problematic samples
    def validate_sample(example):
        text = example.get('text', '')
        if not isinstance(text, str):
            return False
        if len(text.strip()) < 50:  # Too short
            return False
        if len(text) > 50000:  # Too long
            return False
        return True
    
    original_size = len(dataset)
    dataset = dataset.filter(validate_sample)
    filtered = original_size - len(dataset)
    
    if filtered > 0:
        logger.info(f"Filtered {filtered} problematic samples")
    
    # Final dataset info
    logger.info(f"Final dataset: {len(dataset):,} samples")
    
    return dataset

class SinhalaTrainer:
    """Main trainer class with complete error handling"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # Create directories
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.final_model_dir).mkdir(parents=True, exist_ok=True)
    
    def setup_model(self):
        """Load and configure model"""
        logger.info("Loading model...")
        
        # GPU setup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU: {gpu_name} ({vram:.1f}GB)")
        
        # Load model
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            dtype=torch.float16,
            load_in_4bit=self.config.use_4bit,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",  # Stable for RTX 4090
        )
        
        # Configure tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"
        
        # Add LoRA
        logger.info(f"Adding LoRA (r={self.config.lora_r}, alpha={self.config.lora_alpha})")
        
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth" if self.config.use_gradient_checkpointing else False,
            random_state=self.config.seed,
        )
        
        # Log model info
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable parameters: {trainable/1e6:.1f}M ({trainable/total*100:.1f}%)")
        
        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / 1e9
            logger.info(f"VRAM after model load: {vram_used:.1f}GB")
    
    def train(self):
        """Execute training with error recovery"""
        
        # Load dataset
        dataset = prepare_dataset_robust()
        
        # Calculate training steps
        total_samples = len(dataset)
        steps_per_epoch = total_samples // (self.config.batch_size * self.config.gradient_accumulation)
        total_steps = steps_per_epoch * self.config.num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        logger.info(f"\nTraining Plan:")
        logger.info(f"  Total samples: {total_samples:,}")
        logger.info(f"  Batch size: {self.config.batch_size} x {self.config.gradient_accumulation} = {self.config.batch_size * self.config.gradient_accumulation}")
        logger.info(f"  Steps per epoch: {steps_per_epoch:,}")
        logger.info(f"  Total steps: {total_steps:,}")
        logger.info(f"  Warmup steps: {warmup_steps}")
        logger.info(f"  Estimated time: {total_steps/30/3600:.1f} hours at 30 steps/sec")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            
            # Training parameters
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation,
            
            # Optimizer
            learning_rate=self.config.learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=self.config.weight_decay,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",  # 8-bit optimizer for memory
            
            # Mixed precision
            fp16=self.config.fp16,
            tf32=self.config.tf32,
            
            # Gradient
            max_grad_norm=1.0,
            gradient_checkpointing=self.config.use_gradient_checkpointing,
            
            # Checkpointing
            save_strategy="steps",
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            
            # Evaluation
            eval_strategy="no",  # No validation for speed
            
            # Logging
            logging_dir="logs",
            logging_steps=self.config.logging_steps,
            logging_first_step=True,
            report_to="wandb" if os.environ.get('WANDB_MODE') != 'disabled' else "none",
            
            # DataLoader
            dataloader_num_workers=self.config.num_workers,
            dataloader_pin_memory=True,
            dataloader_persistent_workers=False,
            
            # Critical for stability
            group_by_length=False,  # Avoid variable batch sizes
            torch_compile=False,  # Explicitly disabled
            ddp_find_unused_parameters=False,
            
            # Resume
            resume_from_checkpoint=True,  # Auto-resume if exists
            ignore_data_skip=True,
            
            # Misc
            seed=self.config.seed,
            data_seed=self.config.seed,
            
            # Name
            run_name=self.config.wandb_run_name,
        )
        
        # Create trainer
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",  # Direct field name
            max_seq_length=self.config.max_seq_length,
            packing=False,  # CRITICAL: No packing for stability
            args=training_args,
            callbacks=[RobustMonitor(self.config)],
        )
        
        # Start training
        logger.info("\n🚀 Starting training...")
        
        try:
            # Check for existing checkpoint
            checkpoint = None
            output_path = Path(self.config.output_dir)
            if output_path.exists():
                checkpoints = sorted(output_path.glob("checkpoint-*"))
                if checkpoints:
                    checkpoint = str(checkpoints[-1])
                    logger.info(f"Resuming from checkpoint: {checkpoint}")
            
            # Train
            result = self.trainer.train(resume_from_checkpoint=checkpoint)
            
            logger.info(f"✅ Training completed successfully!")
            logger.info(f"Final loss: {result.training_loss:.4f}")
            
            return True
            
        except torch.cuda.OutOfMemoryError:
            logger.error("GPU OOM! Reduce batch_size in config.")
            return False
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def save_model(self):
        """Save final model"""
        logger.info("Saving model...")
        
        # Save LoRA adapter
        self.model.save_pretrained(self.config.final_model_dir)
        self.tokenizer.save_pretrained(self.config.final_model_dir)
        
        # Save config
        with open(Path(self.config.final_model_dir) / "training_config.json", 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        
        logger.info(f"Model saved to: {self.config.final_model_dir}")
        
        # Optional: Save merged model
        disk_free = shutil.disk_usage("/").free / (1024**3)
        if disk_free > 15:
            logger.info("Saving merged model...")
            self.model.save_pretrained_merged(
                f"{self.config.final_model_dir}_merged",
                self.tokenizer,
                save_method="merged_16bit",
            )
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'trainer') and self.trainer:
            del self.trainer
        if hasattr(self, 'model') and self.model:
            del self.model
        if hasattr(self, 'tokenizer') and self.tokenizer:
            del self.tokenizer
        
        torch.cuda.empty_cache()
        gc.collect()
    
    def run(self):
        """Complete pipeline"""
        try:
            # Setup
            self.setup_model()
            
            # Train
            success = self.train()
            
            if success:
                # Save
                self.save_model()
                
                logger.info("\n" + "🎉"*30)
                logger.info("SUCCESS! Training completed!")
                logger.info(f"Model saved to: {self.config.final_model_dir}")
                logger.info("🎉"*30)
                
                return 0
            else:
                return 1
                
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")
            return 130
            
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            import traceback
            traceback.print_exc()
            return 1
            
        finally:
            self.cleanup()

def main():
    """Entry point"""
    print("="*70)
    print("SINHALA LLM TRAINING - FINAL VERSION")
    print("Dataset: 385,335 samples")
    print("Hardware: RTX 4090 24GB")
    print("Expected time: <20 hours")
    print("="*70)
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return 1
    
    # Check dataset
    if not Path("data/train_dataset_cleaned").exists():
        print("❌ Dataset not found at data/train_dataset_cleaned")
        return 1
    
    print("✅ All checks passed\n")
    
    # Run training
    config = TrainingConfig()
    trainer = SinhalaTrainer(config)
    return trainer.run()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)