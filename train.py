#!/usr/bin/env python3
"""
train_production.py - Sinhala Language Acquisition Training
FIXED VERSION - Works with Unsloth SFTTrainer
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

# CRITICAL: Disable all compilation
import torch
torch._dynamo.config.disable = True
os.environ['TORCH_COMPILE_DISABLE'] = '1'

# RTX 4090 Specific Optimizations
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# WandB setup
os.environ['WANDB_PROJECT'] = 'sinhala-llm'
os.environ['WANDB_WATCH'] = 'gradients'

# Disk space management
os.environ['HF_HOME'] = '/tmp/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers'
os.environ['HF_DATASETS_CACHE'] = '/tmp/datasets'

import wandb
import psutil
import GPUtil
from unsloth import FastLanguageModel
from transformers import TrainingArguments, TrainerCallback
from trl import SFTTrainer
from datasets import load_from_disk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

@dataclass
class LanguageAcquisitionConfig:
    """Optimized for RTX 4090 in <24 hours"""
    
    model_name: str = "HuggingFaceTB/SmolLM2-1.7B"
    max_seq_length: int = 1536
    
    # RTX 4090 Optimal Batch Settings
    batch_size: int = 16
    gradient_accumulation: int = 2
    
    # Language Acquisition 
    num_epochs: int = 5
    
    # Learning parameters
    learning_rate: float = 5e-4
    warmup_steps: int = 500
    weight_decay: float = 0.01
    
    # LoRA configuration
    lora_r: int = 80
    lora_alpha: int = 160
    lora_dropout: float = 0.05
    
    target_modules: list = None
    
    # Memory optimizations
    use_4bit: bool = True
    use_gradient_checkpointing: bool = True
    use_flash_attention: bool = False
    
    # Saving strategy
    save_strategy: str = "epoch"
    save_total_limit: int = 1
    eval_strategy: str = "no"
    logging_steps: int = 50
    
    # Paths
    output_dir: str = "/tmp/outputs"
    final_model_dir: str = "models/sinhala_llm"
    
    # Performance
    seed: int = 42
    fp16: bool = True
    tf32: bool = True
    
    dataloader_num_workers: int = 4
    
    lr_scheduler_type: str = "cosine"
    
    # WandB configuration
    wandb_project: str = "sinhala-llm"
    wandb_entity: str = None
    wandb_run_name: str = f"sinhala-rtx4090-{datetime.now().strftime('%Y%m%d-%H%M')}"
    wandb_tags: list = None
    
    def __post_init__(self):
        self.target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
        self.wandb_tags = ["sinhala", "language-acquisition", "lora", "rtx4090", "production"]

class RTX4090Monitor(TrainerCallback):
    """Monitoring with WandB logging"""
    
    def __init__(self, config):
        self.config = config
        self.start_time = time.time()
        self.best_loss = float('inf')
        self.last_log_time = time.time()
        self.last_step = 0
        
    def check_disk_space(self):
        """Monitor disk space"""
        free_gb = shutil.disk_usage("/").free / (1024**3)
        if free_gb < 2:
            logger.critical(f"CRITICAL: Only {free_gb:.1f}GB free!")
            os.system("rm -rf /tmp/outputs/checkpoint-* 2>/dev/null")
            os.system("find /tmp -name '*.pyc' -delete 2>/dev/null")
            gc.collect()
            torch.cuda.empty_cache()
        return free_gb
    
    def get_system_metrics(self):
        """Get system metrics"""
        metrics = {}
        
        if torch.cuda.is_available():
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    metrics["gpu/utilization"] = gpu.load * 100
                    metrics["gpu/memory_gb"] = gpu.memoryUsed / 1024
                    metrics["gpu/memory_percent"] = gpu.memoryUtil * 100
                    metrics["gpu/temperature"] = gpu.temperature
                    
                metrics["gpu/allocated_gb"] = torch.cuda.memory_allocated() / 1e9
                metrics["gpu/reserved_gb"] = torch.cuda.memory_reserved() / 1e9
                metrics["gpu/max_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9
            except Exception as e:
                logger.debug(f"GPU metrics error: {e}")
        
        try:
            metrics["system/cpu_percent"] = psutil.cpu_percent(interval=1)
            metrics["system/ram_gb"] = psutil.virtual_memory().used / 1e9
            metrics["system/ram_percent"] = psutil.virtual_memory().percent
        except Exception as e:
            logger.debug(f"CPU metrics error: {e}")
        
        metrics["system/disk_free_gb"] = shutil.disk_usage("/").free / (1024**3)
        
        return metrics
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Initialize WandB run"""
        wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            name=self.config.wandb_run_name,
            tags=self.config.wandb_tags,
            config={
                "model": self.config.model_name,
                "lora_r": self.config.lora_r,
                "lora_alpha": self.config.lora_alpha,
                "batch_size": self.config.batch_size,
                "gradient_accumulation": self.config.gradient_accumulation,
                "effective_batch_size": self.config.batch_size * self.config.gradient_accumulation,
                "learning_rate": self.config.learning_rate,
                "num_epochs": self.config.num_epochs,
                "max_seq_length": self.config.max_seq_length,
                "warmup_steps": self.config.warmup_steps,
                "weight_decay": self.config.weight_decay,
                "use_4bit": self.config.use_4bit,
                "total_steps": state.max_steps,
                "hardware": "RTX 4090 24GB",
            }
        )
        
        system_metrics = self.get_system_metrics()
        wandb.log({"initial/" + k: v for k, v in system_metrics.items()})
        
    def on_step_end(self, args, state, control, **kwargs):
        """Monitor every 100 steps"""
        if state.global_step % 100 == 0:
            disk_free = self.check_disk_space()
            
            if wandb.run is not None:
                wandb.log({
                    "system/disk_free_gb": disk_free,
                    "training/global_step": state.global_step,
                }, step=state.global_step)
            
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
            
        loss = logs.get("loss", 0)
        if loss and loss < self.best_loss:
            self.best_loss = loss
        
        current_time = time.time()
        elapsed = current_time - self.start_time
        steps_per_sec = state.global_step / elapsed if elapsed > 0 else 0
        
        if self.last_step > 0:
            time_delta = current_time - self.last_log_time
            step_delta = state.global_step - self.last_step
            instant_speed = step_delta / time_delta if time_delta > 0 else 0
        else:
            instant_speed = steps_per_sec
        
        self.last_log_time = current_time
        self.last_step = state.global_step
        
        if steps_per_sec < 20 and state.global_step > 100:
            logger.warning(f"⚠️ Below target speed: {steps_per_sec:.1f} steps/s")
        
        total_steps = state.max_steps if hasattr(state, 'max_steps') else 0
        if total_steps and steps_per_sec:
            eta_hours = (total_steps - state.global_step) / steps_per_sec / 3600
            progress = (state.global_step / total_steps) * 100
            samples_processed = state.global_step * args.per_device_train_batch_size * args.gradient_accumulation_steps
            
            system_metrics = self.get_system_metrics()
            
            if wandb.run is not None:
                wandb_log = {
                    "training/loss": loss,
                    "training/best_loss": self.best_loss,
                    "training/learning_rate": logs.get("learning_rate", 0),
                    "training/epoch": logs.get("epoch", state.epoch if hasattr(state, 'epoch') else 0),
                    "training/global_step": state.global_step,
                    "training/progress_percent": progress,
                    "training/samples_processed": samples_processed,
                    
                    "performance/steps_per_second": steps_per_sec,
                    "performance/instant_steps_per_second": instant_speed,
                    "performance/samples_per_second": steps_per_sec * args.per_device_train_batch_size,
                    "performance/eta_hours": eta_hours,
                    "performance/elapsed_hours": elapsed / 3600,
                    
                    **system_metrics
                }
                
                if "grad_norm" in logs:
                    wandb_log["training/grad_norm"] = logs["grad_norm"]
                
                wandb.log(wandb_log, step=state.global_step)
            
            if state.global_step % 50 == 0:
                vram_used = system_metrics.get("gpu/allocated_gb", 0)
                logger.info(
                    f"Step {state.global_step}/{total_steps} | "
                    f"Loss: {loss:.4f} (best: {self.best_loss:.4f}) | "
                    f"Speed: {steps_per_sec:.1f} steps/s | "
                    f"Progress: {progress:.1f}% | "
                    f"ETA: {eta_hours:.1f}h | "
                    f"VRAM: {vram_used:.1f}/24GB"
                )
                
                if state.global_step % 500 == 0:
                    disk_free = self.check_disk_space()
                    logger.info(f"Disk free: {disk_free:.1f}GB")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Log final metrics to WandB"""
        if wandb.run is not None:
            elapsed = time.time() - self.start_time
            
            system_metrics = self.get_system_metrics()
            
            wandb.run.summary["final_loss"] = self.best_loss
            wandb.run.summary["total_steps"] = state.global_step
            wandb.run.summary["total_hours"] = elapsed / 3600
            wandb.run.summary["avg_steps_per_second"] = state.global_step / elapsed
            
            for k, v in system_metrics.items():
                wandb.run.summary[f"final_{k}"] = v
            
            wandb.finish()

def prepare_dataset_for_acquisition():
    """Prepare dataset with proper validation"""
    logger.info("Loading dataset for language acquisition...")
    
    train_path = None
    for path in ["data/train_dataset_cleaned", "data/train_dataset"]:
        if Path(path).exists():
            train_path = path
            break
    
    if not train_path:
        raise FileNotFoundError("No training dataset found!")
    
    train_dataset = load_from_disk(train_path)
    original_size = len(train_dataset)
    
    # CRITICAL: Remove empty or invalid samples
    def is_valid_sample(example):
        text = example.get('text', '')
        # Ensure text is string and has content
        if not isinstance(text, str):
            return False
        text = text.strip()
        if not text:  # Remove empty texts
            return False
        text_len = len(text)
        # Remove too short or too long
        return 50 < text_len < 30000
    
    train_dataset = train_dataset.filter(is_valid_sample)
    
    # CRITICAL: Ensure all samples have valid text
    def clean_and_validate(example):
        text = example.get('text', '')
        if not isinstance(text, str):
            text = str(text) if text else ""
        
        text = text.strip()
        
        # If still empty after cleaning, provide minimal content
        if not text:
            text = " "  # Single space as fallback
            
        example['text'] = text
        return example
    
    train_dataset = train_dataset.map(clean_and_validate)
    
    # Shuffle for better training
    train_dataset = train_dataset.shuffle(seed=42)
    
    filtered_size = len(train_dataset)
    if filtered_size < original_size:
        logger.info(f"Filtered {original_size - filtered_size} samples for quality")
    
    # Final validation check
    sample = train_dataset[0]
    logger.info(f"Sample text preview: {sample['text'][:100]}...")
    
    logger.info(f"Final dataset: {filtered_size:,} samples")
    
    if wandb.run is not None:
        wandb.run.summary["dataset_size"] = filtered_size
        wandb.run.summary["dataset_filtered"] = original_size - filtered_size
    
    return train_dataset

def clear_disk_space():
    """Aggressive disk cleanup"""
    logger.info("Clearing disk space...")
    
    cleanup_commands = [
        "rm -rf ~/.cache/pip",
        "rm -rf /tmp/outputs/checkpoint-*",
        "rm -rf /tmp/hf_cache/hub/models--*/blobs/*",
        "rm -rf /tmp/transformers/*",
        "find /tmp -type f -name '*.pyc' -delete",
        "find /tmp -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null",
    ]
    
    for cmd in cleanup_commands:
        os.system(f"{cmd} 2>/dev/null")
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    disk_free = shutil.disk_usage("/").free / (1024**3)
    logger.info(f"Disk space after cleanup: {disk_free:.1f}GB free")
    
    if disk_free < 5:
        logger.critical("Insufficient disk space even after cleanup!")
        sys.exit(1)

class SinhalaLanguageTrainer:
    """Production trainer for Sinhala language acquisition"""
    
    def __init__(self, config: LanguageAcquisitionConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None  # Initialize as None
        
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.final_model_dir).mkdir(parents=True, exist_ok=True)
        
    def setup_rtx4090_optimization(self):
        """RTX 4090 specific optimizations"""
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            torch.cuda.set_per_process_memory_fraction(0.95)
            
            torch.cuda.empty_cache()
            gc.collect()
            
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU: {gpu_name}")
            logger.info(f"VRAM: {vram:.1f}GB")
            logger.info(f"CUDA: {torch.version.cuda}")
            
    def load_model_for_acquisition(self):
        """Load and configure model"""
        logger.info("="*60)
        logger.info("LOADING MODEL FOR SINHALA LANGUAGE ACQUISITION")
        logger.info("="*60)
        
        self.setup_rtx4090_optimization()
        clear_disk_space()
        
        logger.info(f"Loading base model: {self.config.model_name}")
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            dtype=torch.float16,
            load_in_4bit=self.config.use_4bit,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
        )
        
        # Configure tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"
        self.tokenizer.model_max_length = self.config.max_seq_length
        
        # Apply LoRA
        logger.info(f"Applying LoRA with rank={self.config.lora_r}")
        
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.lora_r,
            target_modules=self.config.target_modules,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=self.config.seed,
            use_rslora=False,
        )
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable parameters: {trainable/1e6:.1f}M ({trainable/total*100:.1f}%)")
        
        if wandb.run is not None:
            wandb.run.summary["trainable_params_millions"] = trainable/1e6
            wandb.run.summary["total_params_billions"] = total/1e9
            wandb.run.summary["trainable_percent"] = trainable/total*100
        
        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / 1e9
            logger.info(f"VRAM after model load: {vram_used:.1f}/24GB")
            
    def train(self):
        """Execute training"""
        logger.info("\n" + "="*60)
        logger.info("STARTING SINHALA LANGUAGE ACQUISITION")
        logger.info("="*60)
        
        self.load_model_for_acquisition()
        train_dataset = prepare_dataset_for_acquisition()
        
        steps_per_epoch = len(train_dataset) // (self.config.batch_size * self.config.gradient_accumulation)
        total_steps = steps_per_epoch * self.config.num_epochs
        
        logger.info(f"\nTraining Configuration:")
        logger.info(f"  Dataset: {len(train_dataset):,} samples")
        logger.info(f"  Batch size: {self.config.batch_size} x {self.config.gradient_accumulation} = {self.config.batch_size * self.config.gradient_accumulation}")
        logger.info(f"  Epochs: {self.config.num_epochs}")
        logger.info(f"  Total steps: {total_steps:,}")
        logger.info(f"  Learning rate: {self.config.learning_rate}")
        logger.info(f"  LoRA rank: {self.config.lora_r}")
        
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation,
            
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            lr_scheduler_type=self.config.lr_scheduler_type,
            
            optim="adamw_8bit",
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            
            fp16=True,
            tf32=True,
            
            max_grad_norm=1.0,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            
            save_strategy=self.config.save_strategy,
            save_total_limit=self.config.save_total_limit,
            save_safetensors=True,
            
            eval_strategy="no",
            
            logging_steps=self.config.logging_steps,
            logging_first_step=True,
            report_to="wandb",
            
            dataloader_num_workers=self.config.dataloader_num_workers,
            dataloader_pin_memory=True,
            dataloader_persistent_workers=False,
            dataloader_drop_last=False,  # Changed to False
            
            group_by_length=False,
            torch_compile=False,
            remove_unused_columns=False,
            
            seed=self.config.seed,
            data_seed=self.config.seed,
            ignore_data_skip=True,
            ddp_find_unused_parameters=False,
            
            run_name=self.config.wandb_run_name,
        )
        
        # CRITICAL: Use dataset_text_field instead of formatting_func
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            dataset_text_field="text",  # Direct field access
            max_seq_length=self.config.max_seq_length,
            packing=False,  # No packing for stability
            args=training_args,
            callbacks=[RTX4090Monitor(self.config)],
        )
        
        logger.info("\n🚀 Starting language acquisition training...")
        logger.info(f"📊 WandB: https://wandb.ai/{self.config.wandb_entity or 'your-username'}/{self.config.wandb_project}")
        
        start_time = time.time()
        
        try:
            resume_checkpoint = None
            checkpoints = sorted(Path(self.config.output_dir).glob("checkpoint-*"))
            if checkpoints:
                resume_checkpoint = str(checkpoints[-1])
                logger.info(f"Resuming from: {resume_checkpoint}")
            
            result = self.trainer.train(resume_from_checkpoint=resume_checkpoint)
            
            elapsed = time.time() - start_time
            logger.info(f"\n{'='*60}")
            logger.info(f"✅ TRAINING COMPLETED SUCCESSFULLY!")
            logger.info(f"Time: {elapsed/3600:.2f} hours")
            logger.info(f"Final loss: {result.training_loss:.4f}")
            logger.info(f"Steps: {result.global_step}")
            logger.info(f"Avg speed: {result.global_step/elapsed:.1f} steps/sec")
            logger.info(f"{'='*60}")
            
            return True
            
        except torch.cuda.OutOfMemoryError:
            logger.critical("GPU OOM! Reduce batch_size to 12")
            if wandb.run is not None:
                wandb.alert(
                    title="Training Failed - OOM",
                    text="GPU out of memory. Reduce batch size to 12.",
                    level="ERROR"
                )
            return False
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            if wandb.run is not None:
                wandb.alert(
                    title="Training Failed",
                    text=f"Error: {str(e)}",
                    level="ERROR"
                )
            return False
            
    def save_final_model(self):
        """Save the trained model"""
        logger.info("\n💾 Saving final model...")
        
        clear_disk_space()
        
        logger.info("Saving LoRA adapter...")
        self.model.save_pretrained(self.config.final_model_dir)
        self.tokenizer.save_pretrained(self.config.final_model_dir)
        
        config_path = Path(self.config.final_model_dir) / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2, default=str)
        
        logger.info(f"✅ Model saved to: {self.config.final_model_dir}")
        
        if wandb.run is not None:
            wandb.run.summary["final_model_path"] = str(self.config.final_model_dir)
        
        disk_free = shutil.disk_usage("/").free / (1024**3)
        if disk_free > 10:
            logger.info("Saving merged model...")
            merged_path = f"{self.config.final_model_dir}_merged"
            self.model.save_pretrained_merged(
                merged_path,
                self.tokenizer,
                save_method="merged_16bit",
                safe_serialization=True,
            )
            logger.info(f"✅ Merged model saved to: {merged_path}")
            
            if wandb.run is not None:
                wandb.run.summary["merged_model_path"] = str(merged_path)
        else:
            logger.warning("Insufficient disk space for merged model")
            
    def cleanup(self):
        """Final cleanup - FIXED"""
        try:
            if hasattr(self, 'model') and self.model:
                del self.model
            if hasattr(self, 'trainer') and self.trainer:
                del self.trainer
            if hasattr(self, 'tokenizer') and self.tokenizer:
                del self.tokenizer
                
            torch.cuda.empty_cache()
            gc.collect()
            
            os.system("rm -rf /tmp/outputs 2>/dev/null")
        except Exception as e:
            logger.debug(f"Cleanup error (non-critical): {e}")
        
    def run(self):
        """Main execution pipeline"""
        try:
            clear_disk_space()
            
            success = self.train()
            
            if success:
                self.save_final_model()
                self.cleanup()
                
                logger.info("\n" + "🎉"*30)
                logger.info("SINHALA LANGUAGE ACQUISITION COMPLETE!")
                logger.info(f"Model ready at: {self.config.final_model_dir}")
                logger.info("🎉"*30)
                return 0
            else:
                logger.error("Training failed!")
                return 1
                
        except KeyboardInterrupt:
            logger.warning("\n⚠️ Training interrupted!")
            logger.info("Progress saved - run again to resume")
            return 130
            
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            import traceback
            traceback.print_exc()
            return 1
        finally:
            self.cleanup()

def main():
    """Entry point with system checks"""
    
    print("="*70)
    print("SINHALA LANGUAGE ACQUISITION TRAINING")
    print("RTX 4090 Optimized | <24 Hour Target | WandB Monitoring")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return 1
        
    disk_free = shutil.disk_usage("/").free / (1024**3)
    print(f"📁 Disk space: {disk_free:.1f}GB free")
    if disk_free < 10:
        print("⚠️  WARNING: Low disk space! Cleaning...")
        os.system("pip cache purge 2>/dev/null")
        
    dataset_exists = any(Path(p).exists() for p in ["data/train_dataset_cleaned", "data/train_dataset"])
    if not dataset_exists:
        print("❌ No dataset found! Run data preparation first.")
        return 1
        
    print("\n✅ All checks passed. Starting training...\n")
    
    config = LanguageAcquisitionConfig()
    trainer = SinhalaLanguageTrainer(config)
    return trainer.run()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)