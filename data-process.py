#!/usr/bin/env python3
"""
data-process.py - Enhanced Sinhala dataset processor with validation and statistics
Handles 427k samples efficiently with comprehensive quality checks
Author: Sinhala LLM Project
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

# Set CUDA environment
os.environ['CUDA_HOME'] = '/usr/local/cuda'
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:' + os.environ.get('LD_LIBRARY_PATH', '')

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
    
    # Length statistics (characters)
    avg_char_length: float = 0
    max_char_length: int = 0
    min_char_length: int = float('inf')
    
    # Token statistics
    avg_token_length: float = 0
    max_token_length: int = 0
    min_token_length: int = float('inf')
    truncated_samples: int = 0
    
    # Language distribution
    sinhala_samples: int = 0
    english_samples: int = 0
    mixed_samples: int = 0
    unknown_samples: int = 0
    
    # Quality metrics
    empty_instructions: int = 0
    empty_outputs: int = 0
    short_outputs: int = 0  # < 10 chars
    very_long_outputs: int = 0  # > 5000 chars

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

class EnhancedSinhalaDataProcessor:
    """Enhanced processor with validation and comprehensive statistics"""
    
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
        
        # Track problematic samples for debugging
        self.problematic_samples = []
        
    def load_and_validate_json(self) -> List[Dict]:
        """Load JSON with error handling and structure validation"""
        logger.info(f"📂 Loading data from {self.data_file}")
        
        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
        
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON file: {e}")
            raise
        except UnicodeDecodeError as e:
            logger.error(f"Encoding error in file: {e}")
            raise
        
        # Handle different JSON structures
        if isinstance(data, dict):
            if 'data' in data:
                data = data['data']
            elif 'examples' in data:
                data = data['examples']
            elif 'samples' in data:
                data = data['samples']
            else:
                # Try to extract list from first key
                keys = list(data.keys())
                if keys and isinstance(data[keys[0]], list):
                    data = data[keys[0]]
        
        if not isinstance(data, list):
            raise ValueError(f"Expected list of samples, got {type(data)}")
        
        self.stats.total_samples = len(data)
        logger.info(f"✓ Loaded {len(data):,} raw samples")
        
        return data
    
    def validate_sample(self, sample: Dict, index: int) -> Tuple[bool, str]:
        """Validate a single sample and return (is_valid, reason)"""
        
        # Check required fields
        if not isinstance(sample, dict):
            return False, "Not a dictionary"
        
        instruction = sample.get('instruction', '').strip()
        output = sample.get('output', '').strip()
        
        # Check empty fields
        if not instruction:
            self.stats.empty_instructions += 1
            return False, "Empty instruction"
        
        if not output:
            self.stats.empty_outputs += 1
            return False, "Empty output"
        
        # Check length constraints
        if len(instruction) < self.validation_config.min_instruction_length:
            return False, f"Instruction too short ({len(instruction)} chars)"
        
        if len(output) < self.validation_config.min_output_length:
            self.stats.short_outputs += 1
            return False, f"Output too short ({len(output)} chars)"
        
        if len(output) > self.validation_config.max_output_length:
            self.stats.very_long_outputs += 1
            return False, f"Output too long ({len(output)} chars)"
        
        # Check encoding issues
        if self.validation_config.validate_encoding:
            try:
                instruction.encode('utf-8')
                output.encode('utf-8')
            except UnicodeEncodeError:
                return False, "Encoding error"
        
        return True, "Valid"
    
    def remove_duplicates(self, data: List[Dict]) -> List[Dict]:
        """Remove duplicate samples based on instruction+output hash"""
        if not self.validation_config.remove_duplicates:
            return data
        
        logger.info("🔍 Removing duplicates...")
        seen_hashes: Set[str] = set()
        unique_samples = []
        
        for sample in tqdm(data, desc="Deduplicating"):
            # Create hash from instruction and output
            content = f"{sample.get('instruction', '')}{sample.get('output', '')}"
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_samples.append(sample)
            else:
                self.stats.duplicates_removed += 1
        
        logger.info(f"✓ Removed {self.stats.duplicates_removed:,} duplicates")
        return unique_samples
    
    def detect_language_advanced(self, text: str) -> str:
        """Enhanced language detection with better accuracy"""
        if not text:
            return 'unknown'
        
        # Sinhala Unicode ranges
        # Main Sinhala block: U+0D80-U+0DFF
        # Also check for Zero Width Joiner (U+200D) commonly used in Sinhala
        sinhala_chars = sum(1 for c in text if '\u0D80' <= c <= '\u0DFF')
        
        # ASCII letters only (not numbers or punctuation)
        english_chars = sum(1 for c in text if c.isascii() and c.isalpha())
        
        # Get only alphabetic characters for ratio calculation
        alpha_chars = sum(1 for c in text if c.isalpha())
        
        if alpha_chars == 0:
            return 'unknown'
        
        sinhala_ratio = sinhala_chars / alpha_chars
        english_ratio = english_chars / alpha_chars
        
        # Classify based on ratios
        if sinhala_ratio > 0.7:
            return 'sinhala'
        elif english_ratio > 0.7:
            return 'english'
        elif sinhala_ratio > 0.2 and english_ratio > 0.2:
            return 'mixed'
        elif sinhala_ratio > english_ratio:
            return 'sinhala'
        elif english_ratio > sinhala_ratio:
            return 'english'
        else:
            return 'unknown'
    
    def format_prompt(self, sample: Dict) -> str:
        """Format sample into training prompt"""
        
        # Alpaca-style prompt template
        prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""
        
        # Fill template
        formatted = prompt.format(
            instruction=sample.get('instruction', '').strip(),
            input=sample.get('input', '').strip(),
            output=sample.get('output', '').strip()
        )
        
        # Add EOS token
        if self.tokenizer and self.tokenizer.eos_token:
            formatted += self.tokenizer.eos_token
        else:
            formatted += "</s>"
        
        return formatted
    
    def analyze_token_statistics(self, texts: List[str], sample_size: int = 5000) -> Dict:
        """Analyze tokenization statistics on a sample"""
        logger.info("📊 Analyzing token statistics...")
        
        # Sample texts if too many (for efficiency)
        sample_texts = texts[:sample_size] if len(texts) > sample_size else texts
        token_lengths = []
        truncated_count = 0
        
        for text in tqdm(sample_texts, desc="Tokenizing samples"):
            tokens = self.tokenizer(
                text, 
                truncation=True, 
                max_length=self.max_seq_length,
                return_length=True
            )
            
            length = len(tokens['input_ids'])
            token_lengths.append(length)
            
            if length == self.max_seq_length:
                truncated_count += 1
        
        # Calculate statistics
        stats = {
            'avg_tokens': np.mean(token_lengths),
            'median_tokens': np.median(token_lengths),
            'max_tokens': max(token_lengths),
            'min_tokens': min(token_lengths),
            'std_tokens': np.std(token_lengths),
            'truncated_ratio': truncated_count / len(sample_texts),
            'percentile_95': np.percentile(token_lengths, 95),
            'percentile_99': np.percentile(token_lengths, 99)
        }
        
        # Update global stats
        self.stats.avg_token_length = stats['avg_tokens']
        self.stats.max_token_length = stats['max_tokens']
        self.stats.min_token_length = stats['min_tokens']
        self.stats.truncated_samples = int(truncated_count * (len(texts) / len(sample_texts)))
        
        return stats
    
    def process_dataset(self) -> Tuple[Dataset, Dataset]:
        """Main processing pipeline with validation"""
        logger.info("\n🔄 PROCESSING DATASET")
        logger.info("=" * 60)
        
        # Load tokenizer
        logger.info(f"Loading tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
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
                if len(self.problematic_samples) < 10:  # Keep first 10 for debugging
                    self.problematic_samples.append({
                        'index': i,
                        'reason': reason,
                        'sample': sample
                    })
        
        self.stats.valid_samples = len(valid_data)
        logger.info(f"✓ Valid samples: {len(valid_data):,}/{len(raw_data):,}")
        
        # Remove duplicates
        unique_data = self.remove_duplicates(valid_data)
        
        # Shuffle for better split
        np.random.seed(42)
        np.random.shuffle(unique_data)
        
        # Split into train/validation
        val_size = int(len(unique_data) * self.val_split)
        val_size = min(val_size, 20000)  # Cap validation at 20k
        
        train_data = unique_data[val_size:]
        val_data = unique_data[:val_size]
        
        self.stats.train_samples = len(train_data)
        self.stats.val_samples = len(val_data)
        
        logger.info(f"✓ Split: {len(train_data):,} train, {len(val_data):,} validation")
        
        # Process and format samples
        logger.info("📝 Formatting samples...")
        train_texts = []
        val_texts = []
        char_lengths = []
        language_counter = Counter()
        
        # Process training samples
        for sample in tqdm(train_data, desc="Processing training samples"):
            text = self.format_prompt(sample)
            train_texts.append(text)
            
            # Character statistics
            char_lengths.append(len(text))
            self.stats.max_char_length = max(self.stats.max_char_length, len(text))
            self.stats.min_char_length = min(self.stats.min_char_length, len(text))
            
            # Language detection on output
            lang = self.detect_language_advanced(sample.get('output', ''))
            language_counter[lang] += 1
        
        # Process validation samples
        for sample in tqdm(val_data, desc="Processing validation samples"):
            text = self.format_prompt(sample)
            val_texts.append(text)
        
        # Update statistics
        self.stats.avg_char_length = np.mean(char_lengths)
        self.stats.sinhala_samples = language_counter.get('sinhala', 0)
        self.stats.english_samples = language_counter.get('english', 0)
        self.stats.mixed_samples = language_counter.get('mixed', 0)
        self.stats.unknown_samples = language_counter.get('unknown', 0)
        
        # Token statistics (on sample)
        token_stats = self.analyze_token_statistics(train_texts)
        
        # Create datasets
        train_dataset = Dataset.from_dict({"text": train_texts})
        val_dataset = Dataset.from_dict({"text": val_texts})
        
        # Log token statistics
        logger.info("\n📊 Token Statistics:")
        logger.info(f"  Average tokens: {token_stats['avg_tokens']:.1f}")
        logger.info(f"  Median tokens: {token_stats['median_tokens']:.1f}")
        logger.info(f"  95th percentile: {token_stats['percentile_95']:.0f}")
        logger.info(f"  99th percentile: {token_stats['percentile_99']:.0f}")
        logger.info(f"  Truncated samples: {token_stats['truncated_ratio']*100:.1f}%")
        
        return train_dataset, val_dataset
    
    def save_datasets_and_reports(self, train_dataset: Dataset, val_dataset: Dataset):
        """Save datasets and generate comprehensive reports"""
        logger.info("\n💾 Saving processed datasets...")
        
        # Save datasets
        train_path = self.output_dir / "train_dataset"
        val_path = self.output_dir / "val_dataset"
        
        train_dataset.save_to_disk(str(train_path))
        val_dataset.save_to_disk(str(val_path))
        
        logger.info(f"  ✓ Training dataset saved to: {train_path}")
        logger.info(f"  ✓ Validation dataset saved to: {val_path}")
        
        # Save statistics
        stats_path = self.output_dir / "dataset_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(asdict(self.stats), f, indent=2)
        
        # Save problematic samples for debugging
        if self.problematic_samples:
            problems_path = self.output_dir / "problematic_samples.json"
            with open(problems_path, 'w') as f:
                json.dump(self.problematic_samples, f, indent=2, ensure_ascii=False)
            logger.info(f"  ✓ Problematic samples saved to: {problems_path}")
        
        # Generate detailed report
        report_path = self.output_dir / "processing_report.txt"
        self.generate_report(report_path)
        
        logger.info(f"  ✓ Processing report saved to: {report_path}")
    
    def generate_report(self, report_path: Path):
        """Generate comprehensive processing report"""
        with open(report_path, 'w') as f:
            f.write("SINHALA DATASET PROCESSING REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("OVERVIEW\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total samples loaded: {self.stats.total_samples:,}\n")
            f.write(f"Valid samples: {self.stats.valid_samples:,}\n")
            f.write(f"Invalid samples: {self.stats.invalid_samples:,}\n")
            f.write(f"Duplicates removed: {self.stats.duplicates_removed:,}\n")
            f.write(f"Final training samples: {self.stats.train_samples:,}\n")
            f.write(f"Final validation samples: {self.stats.val_samples:,}\n\n")
            
            f.write("DATA QUALITY ISSUES\n")
            f.write("-" * 30 + "\n")
            f.write(f"Empty instructions: {self.stats.empty_instructions:,}\n")
            f.write(f"Empty outputs: {self.stats.empty_outputs:,}\n")
            f.write(f"Very short outputs (<10 chars): {self.stats.short_outputs:,}\n")
            f.write(f"Very long outputs (>5000 chars): {self.stats.very_long_outputs:,}\n\n")
            
            f.write("CHARACTER STATISTICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Average length: {self.stats.avg_char_length:.0f} characters\n")
            f.write(f"Maximum length: {self.stats.max_char_length:,} characters\n")
            f.write(f"Minimum length: {self.stats.min_char_length:,} characters\n\n")
            
            f.write("TOKEN STATISTICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Average tokens: {self.stats.avg_token_length:.0f}\n")
            f.write(f"Maximum tokens: {self.stats.max_token_length:,}\n")
            f.write(f"Minimum tokens: {self.stats.min_token_length:,}\n")
            f.write(f"Samples truncated to {self.max_seq_length}: {self.stats.truncated_samples:,}\n\n")
            
            f.write("LANGUAGE DISTRIBUTION\n")
            f.write("-" * 30 + "\n")
            total_lang = (self.stats.sinhala_samples + self.stats.english_samples + 
                         self.stats.mixed_samples + self.stats.unknown_samples)
            if total_lang > 0:
                f.write(f"Sinhala: {self.stats.sinhala_samples:,} ({self.stats.sinhala_samples/total_lang*100:.1f}%)\n")
                f.write(f"English: {self.stats.english_samples:,} ({self.stats.english_samples/total_lang*100:.1f}%)\n")
                f.write(f"Mixed: {self.stats.mixed_samples:,} ({self.stats.mixed_samples/total_lang*100:.1f}%)\n")
                f.write(f"Unknown: {self.stats.unknown_samples:,} ({self.stats.unknown_samples/total_lang*100:.1f}%)\n")
    
    def print_summary(self):
        """Print processing summary"""
        logger.info("\n" + "="*60)
        logger.info("📊 PROCESSING SUMMARY")
        logger.info("="*60)
        
        # Data overview
        logger.info(f"Total samples: {self.stats.total_samples:,}")
        logger.info(f"Valid samples: {self.stats.valid_samples:,} ({self.stats.valid_samples/self.stats.total_samples*100:.1f}%)")
        logger.info(f"Training samples: {self.stats.train_samples:,}")
        logger.info(f"Validation samples: {self.stats.val_samples:,}")
        
        # Quality metrics
        if self.stats.duplicates_removed > 0:
            logger.info(f"Duplicates removed: {self.stats.duplicates_removed:,}")
        if self.stats.invalid_samples > 0:
            logger.info(f"Invalid samples filtered: {self.stats.invalid_samples:,}")
        
        # Token information
        logger.info(f"\n📏 Sequence Lengths:")
        logger.info(f"  Average tokens: {self.stats.avg_token_length:.0f}")
        logger.info(f"  Max tokens: {self.stats.max_token_length}")
        if self.stats.truncated_samples > 0:
            logger.info(f"  ⚠️ Truncated samples: {self.stats.truncated_samples:,}")
        
        # Language distribution
        total_analyzed = (self.stats.sinhala_samples + self.stats.english_samples + 
                         self.stats.mixed_samples + self.stats.unknown_samples)
        if total_analyzed > 0:
            logger.info(f"\n🌍 Language Distribution:")
            logger.info(f"  Sinhala: {self.stats.sinhala_samples/total_analyzed*100:.1f}%")
            logger.info(f"  English: {self.stats.english_samples/total_analyzed*100:.1f}%")
            logger.info(f"  Mixed: {self.stats.mixed_samples/total_analyzed*100:.1f}%")
        
        logger.info("="*60)
    
    def run(self):
        """Execute complete data processing pipeline"""
        logger.info("🚀 ENHANCED SINHALA DATASET PROCESSING")
        logger.info("="*60)
        logger.info(f"Configuration:")
        logger.info(f"  Input file: {self.data_file}")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  Max sequence length: {self.max_seq_length}")
        logger.info(f"  Validation split: {self.val_split*100:.1f}%")
        logger.info(f"  System memory: 64GB")
        logger.info("="*60)
        
        try:
            # Process dataset
            train_dataset, val_dataset = self.process_dataset()
            
            # Save datasets and reports
            self.save_datasets_and_reports(train_dataset, val_dataset)
            
            # Print summary
            self.print_summary()
            
            logger.info("\n✅ Data processing complete!")
            logger.info(f"📁 Output saved to: {self.output_dir}")
            logger.info("Ready for training: python3 train.py")
            
        except Exception as e:
            logger.error(f"\n❌ Processing failed: {e}")
            import traceback
            traceback.print_exc()
            raise

if __name__ == "__main__":
    # Configure validation
    validation_config = ValidationConfig(
        min_instruction_length=3,
        min_output_length=5,
        max_output_length=10000,
        remove_duplicates=True,
        validate_encoding=True
    )
    
    # Initialize processor
    processor = EnhancedSinhalaDataProcessor(
        data_file="train.json",
        model_name="HuggingFaceTB/SmolLM2-1.7B",
        val_split=0.05,
        max_seq_length=2048,
        validation_config=validation_config
    )
    
    # Run processing
    processor.run()