#!/usr/bin/env python3
"""
data-process-fixed.py - Fixed Sinhala dataset processor
Fixes tokenization issues and ensures clean dataset output
"""

import os
import sys
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from collections import Counter

import numpy as np
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data_processing.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DataStatistics:
    """Container for dataset statistics"""
    total_samples: int = 0
    valid_samples: int = 0
    train_samples: int = 0
    val_samples: int = 0
    duplicates_removed: int = 0
    invalid_samples: int = 0
    tokenization_errors: int = 0
    
    avg_char_length: float = 0
    max_char_length: int = 0
    min_char_length: int = float('inf')
    
    avg_token_length: float = 0
    max_token_length: int = 0
    min_token_length: int = float('inf')
    truncated_samples: int = 0
    
    sinhala_samples: int = 0
    english_samples: int = 0
    mixed_samples: int = 0
    unknown_samples: int = 0
    
    empty_instructions: int = 0
    empty_outputs: int = 0
    short_outputs: int = 0
    very_long_outputs: int = 0

@dataclass
class ValidationConfig:
    """Configuration for data validation"""
    min_instruction_length: int = 3
    min_output_length: int = 5
    max_output_length: int = 10000
    min_token_length: int = 5
    max_token_length: int = 2048
    remove_duplicates: bool = True
    validate_encoding: bool = True
    test_tokenization: bool = True  # NEW: Test each sample

class FixedSinhalaDataProcessor:
    """Fixed processor that prevents tokenization errors"""
    
    def __init__(self, 
                 data_file: str = "train.json",
                 model_name: str = "HuggingFaceTB/SmolLM2-1.7B",
                 val_split: float = 0.05,
                 max_seq_length: int = 2048,
                 validation_config: Optional[ValidationConfig] = None):
        
        self.data_file = Path(data_file)
        self.model_name = model_name
        self.val_split = val_split
        self.max_seq_length = max_seq_length
        self.output_dir = Path("data")
        self.output_dir.mkdir(exist_ok=True)
        
        self.validation_config = validation_config or ValidationConfig()
        self.stats = DataStatistics()
        self.tokenizer = None
        self.problematic_samples = []
        
    def load_and_validate_json(self) -> List[Dict]:
        """Load JSON with error handling"""
        logger.info(f"📂 Loading data from {self.data_file}")
        
        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
        
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON: {e}")
            raise
        
        # Handle different JSON structures
        if isinstance(data, dict):
            for key in ['data', 'examples', 'samples']:
                if key in data:
                    data = data[key]
                    break
            else:
                # Try first key
                keys = list(data.keys())
                if keys and isinstance(data[keys[0]], list):
                    data = data[keys[0]]
        
        if not isinstance(data, list):
            raise ValueError(f"Expected list of samples, got {type(data)}")
        
        self.stats.total_samples = len(data)
        logger.info(f"✓ Loaded {len(data):,} raw samples")
        
        return data
    
    def validate_sample(self, sample: Dict, index: int) -> Tuple[bool, str]:
        """Validate a single sample"""
        
        if not isinstance(sample, dict):
            return False, "Not a dictionary"
        
        instruction = str(sample.get('instruction', '')).strip()
        output = str(sample.get('output', '')).strip()
        
        # Check empty fields
        if not instruction:
            self.stats.empty_instructions += 1
            return False, "Empty instruction"
        
        if not output:
            self.stats.empty_outputs += 1
            return False, "Empty output"
        
        # Check minimum lengths
        if len(instruction) < self.validation_config.min_instruction_length:
            return False, f"Instruction too short ({len(instruction)} chars)"
        
        if len(output) < self.validation_config.min_output_length:
            self.stats.short_outputs += 1
            return False, f"Output too short ({len(output)} chars)"
        
        if len(output) > self.validation_config.max_output_length:
            self.stats.very_long_outputs += 1
            return False, f"Output too long ({len(output)} chars)"
        
        return True, "Valid"
    
    def format_prompt_simple(self, sample: Dict) -> Optional[str]:
        """Format sample into simple prompt - FIXED VERSION"""
        
        instruction = str(sample.get('instruction', '')).strip()
        output = str(sample.get('output', '')).strip()
        input_text = str(sample.get('input', '')).strip()
        
        # Skip if no content
        if not instruction or not output:
            return None
        
        # Create simple prompt without complex template
        if input_text:
            # Include input if present
            text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            # Simpler format without input
            text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        
        # Ensure minimum length
        if len(text) < 20:
            return None
        
        # Add EOS token if available
        if self.tokenizer and self.tokenizer.eos_token:
            text = text + self.tokenizer.eos_token
        
        return text
    
    def test_tokenization(self, text: str) -> bool:
        """Test if text can be tokenized without errors"""
        if not self.validation_config.test_tokenization:
            return True
            
        try:
            # Test tokenization
            tokens = self.tokenizer(
                text, 
                truncation=True, 
                max_length=self.max_seq_length,
                padding=False,
                return_tensors=None
            )
            
            # Check if we got valid tokens
            if not tokens or not tokens.get('input_ids'):
                return False
                
            # Check minimum token length
            if len(tokens['input_ids']) < self.validation_config.min_token_length:
                return False
                
            return True
            
        except Exception as e:
            logger.debug(f"Tokenization error: {e}")
            return False
    
    def remove_duplicates(self, data: List[Dict]) -> List[Dict]:
        """Remove duplicate samples"""
        if not self.validation_config.remove_duplicates:
            return data
        
        logger.info("🔍 Removing duplicates...")
        seen_hashes: Set[str] = set()
        unique_samples = []
        
        for sample in tqdm(data, desc="Deduplicating"):
            content = f"{sample.get('instruction', '')}{sample.get('output', '')}"
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_samples.append(sample)
            else:
                self.stats.duplicates_removed += 1
        
        logger.info(f"✓ Removed {self.stats.duplicates_removed:,} duplicates")
        return unique_samples
    
    def process_dataset(self) -> Tuple[Dataset, Dataset]:
        """Main processing pipeline - FIXED"""
        logger.info("\n🔄 PROCESSING DATASET WITH TOKENIZATION FIXES")
        logger.info("=" * 60)
        
        # Load tokenizer first
        logger.info(f"Loading tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load raw data
        raw_data = self.load_and_validate_json()
        
        # Validate samples
        logger.info("✓ Validating samples...")
        valid_data = []
        
        for i, sample in enumerate(tqdm(raw_data, desc="Validating")):
            is_valid, reason = self.validate_sample(sample, i)
            if is_valid:
                valid_data.append(sample)
            else:
                self.stats.invalid_samples += 1
                if len(self.problematic_samples) < 10:
                    self.problematic_samples.append({
                        'index': i,
                        'reason': reason
                    })
        
        self.stats.valid_samples = len(valid_data)
        logger.info(f"✓ Valid samples: {len(valid_data):,}/{len(raw_data):,}")
        
        # Remove duplicates
        unique_data = self.remove_duplicates(valid_data)
        
        # Shuffle
        np.random.seed(42)
        np.random.shuffle(unique_data)
        
        # Process and format samples WITH TOKENIZATION TEST
        logger.info("📝 Formatting and testing tokenization...")
        all_texts = []
        
        for sample in tqdm(unique_data, desc="Processing samples"):
            # Format text
            text = self.format_prompt_simple(sample)
            
            if text is None:
                self.stats.tokenization_errors += 1
                continue
            
            # Test tokenization
            if self.validation_config.test_tokenization:
                if not self.test_tokenization(text):
                    self.stats.tokenization_errors += 1
                    continue
            
            # Ensure it's a proper string (not list or other type)
            if not isinstance(text, str):
                self.stats.tokenization_errors += 1
                continue
                
            all_texts.append(text)
        
        logger.info(f"✓ Successfully processed {len(all_texts):,} samples")
        logger.info(f"  Tokenization errors: {self.stats.tokenization_errors:,}")
        
        # Split into train/validation
        val_size = int(len(all_texts) * self.val_split)
        val_size = min(val_size, 10000)  # Cap at 10k
        
        train_texts = all_texts[val_size:]
        val_texts = all_texts[:val_size] if val_size > 0 else []
        
        self.stats.train_samples = len(train_texts)
        self.stats.val_samples = len(val_texts)
        
        logger.info(f"✓ Split: {len(train_texts):,} train, {len(val_texts):,} validation")
        
        # CRITICAL: Create datasets with proper structure
        # Each text must be a string, not a list
        train_dataset = Dataset.from_dict({
            "text": train_texts  # This is a list of strings
        })
        
        val_dataset = None
        if val_texts:
            val_dataset = Dataset.from_dict({
                "text": val_texts  # This is a list of strings
            })
        
        # Verify dataset structure
        logger.info("\n📊 Dataset Verification:")
        sample = train_dataset[0]
        logger.info(f"  Sample type: {type(sample['text'])}")
        logger.info(f"  Sample preview: {sample['text'][:100]}...")
        
        # Quick token statistics
        sample_size = min(1000, len(train_texts))
        token_lengths = []
        for text in train_texts[:sample_size]:
            tokens = self.tokenizer(text, truncation=True, max_length=self.max_seq_length)
            token_lengths.append(len(tokens['input_ids']))
        
        if token_lengths:
            self.stats.avg_token_length = np.mean(token_lengths)
            self.stats.max_token_length = max(token_lengths)
            self.stats.min_token_length = min(token_lengths)
            
            logger.info(f"  Avg tokens: {self.stats.avg_token_length:.0f}")
            logger.info(f"  Max tokens: {self.stats.max_token_length}")
            logger.info(f"  Min tokens: {self.stats.min_token_length}")
        
        return train_dataset, val_dataset
    
    def save_datasets_and_reports(self, train_dataset: Dataset, val_dataset: Optional[Dataset]):
        """Save datasets with verification"""
        logger.info("\n💾 Saving processed datasets...")
        
        # Save with clear naming
        train_path = self.output_dir / "train_dataset"
        train_dataset.save_to_disk(str(train_path))
        logger.info(f"  ✓ Training dataset saved to: {train_path}")
        
        if val_dataset:
            val_path = self.output_dir / "val_dataset"
            val_dataset.save_to_disk(str(val_path))
            logger.info(f"  ✓ Validation dataset saved to: {val_path}")
        
        # Save statistics
        stats_path = self.output_dir / "dataset_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(asdict(self.stats), f, indent=2)
        
        # Create simple report
        report_path = self.output_dir / "processing_report.txt"
        with open(report_path, 'w') as f:
            f.write("DATASET PROCESSING REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total samples: {self.stats.total_samples:,}\n")
            f.write(f"Valid samples: {self.stats.valid_samples:,}\n")
            f.write(f"Tokenization errors: {self.stats.tokenization_errors:,}\n")
            f.write(f"Final training: {self.stats.train_samples:,}\n")
            f.write(f"Final validation: {self.stats.val_samples:,}\n")
            f.write(f"\nAvg tokens: {self.stats.avg_token_length:.0f}\n")
            f.write(f"Max tokens: {self.stats.max_token_length}\n")
            f.write(f"Min tokens: {self.stats.min_token_length}\n")
        
        logger.info(f"  ✓ Report saved to: {report_path}")
    
    def run(self):
        """Execute pipeline"""
        logger.info("🚀 FIXED SINHALA DATASET PROCESSING")
        logger.info("="*60)
        logger.info(f"Configuration:")
        logger.info(f"  Input: {self.data_file}")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  Max length: {self.max_seq_length}")
        logger.info("="*60)
        
        try:
            # Process
            train_dataset, val_dataset = self.process_dataset()
            
            # Save
            self.save_datasets_and_reports(train_dataset, val_dataset)
            
            # Summary
            logger.info("\n" + "="*60)
            logger.info("✅ PROCESSING COMPLETE!")
            logger.info(f"Training samples: {self.stats.train_samples:,}")
            logger.info(f"Validation samples: {self.stats.val_samples:,}")
            logger.info(f"Tokenization errors fixed: {self.stats.tokenization_errors:,}")
            logger.info("="*60)
            logger.info("\n📁 Ready for training: python3 train.py")
            
        except Exception as e:
            logger.error(f"\n❌ Processing failed: {e}")
            import traceback
            traceback.print_exc()
            raise

if __name__ == "__main__":
    # Run with fixed configuration
    processor = FixedSinhalaDataProcessor(
        data_file="train.json",
        model_name="HuggingFaceTB/SmolLM2-1.7B",
        val_split=0.05,
        max_seq_length=1536,  # Slightly reduced for safety
        validation_config=ValidationConfig(
            min_instruction_length=3,
            min_output_length=5,
            max_output_length=10000,
            remove_duplicates=True,
            validate_encoding=True,
            test_tokenization=True  # Enable tokenization testing
        )
    )
    
    processor.run()