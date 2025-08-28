#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ['CUDA_HOME'] = '/usr/local/cuda'
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:' + os.environ.get('LD_LIBRARY_PATH', '')

"""
evaluation_baseline.py - Evaluation of Thimira Sinhala LLaMA-2 for baseline comparison
Author: SAkash Hasendra
"""

import json
import logging
import torch
import time
import numpy as np
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from tqdm import tqdm

import nltk
nltk.download('punkt', quiet=True)

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk, Dataset
from rouge_score import rouge_scorer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/evaluation_baseline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SinhalaLLMBaselineEvaluator:
    """Baseline evaluator for Thimira Sinhala LLaMA-2 from HuggingFace"""

    def __init__(self, model_path: str = "Thimira/sinhala-llama-2-7b-chat-hf", max_samples: int = 1000):
        self.model_path = model_path
        self.max_samples = max_samples
        self.results_dir = Path("evaluation_results/baseline")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.metrics = {
            'model_info': {},
            'generation_quality': {},
            'language_understanding': {},
            'speed_metrics': {},
            'perplexity': None,  # Only if validation data available
            'rouge_scores': {},
            'model_size': {},
            'timestamp': datetime.now().isoformat()
        }

    def load_model(self):
        """Load the HuggingFace model"""
        logger.info(f"\n📂 Loading baseline model: {self.model_path}")

        try:
            # Load tokenizer and model from HuggingFace
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto",
                trust_remote_code=True
            )

            self.model.eval()

            # Get model info
            total_params = sum(p.numel() for p in self.model.parameters())
            model_size_mb = sum(
                p.numel() * p.element_size()
                for p in self.model.parameters()
            ) / 1024 / 1024

            self.metrics['model_size'] = {
                'total_parameters': total_params,
                'size_mb': model_size_mb
            }

            self.metrics['model_info'] = {
                'model_name': self.model_path,
                'model_type': 'Thimira Sinhala LLaMA 2 Chat',
                'vocab_size': len(self.tokenizer) if self.tokenizer else 'unknown',
                'base_model': 'LLaMA-2-7B-Chat (fine-tuned for Sinhala)'
            }

            logger.info(f"✓ Model loaded: {total_params/1e6:.1f}M parameters")
            logger.info(f"✓ Vocabulary size: {len(self.tokenizer)}")

        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise

    def check_local_datasets(self) -> bool:
        """Check if local validation datasets exist"""
        val_path = Path("data/val_dataset")
        return val_path.exists()

    def evaluate_perplexity(self):
        """Calculate perplexity if validation data available"""
        if not self.check_local_datasets():
            logger.warning("⚠️  No local validation dataset found - skipping perplexity calculation")
            self.metrics['perplexity'] = {
                'status': 'skipped',
                'reason': 'no_validation_data'
            }
            return

        logger.info("\n📊 Evaluating Perplexity...")

        try:
            val_dataset = load_from_disk("data/val_dataset")
            samples = val_dataset.select(range(min(self.max_samples, len(val_dataset))))

            total_loss = 0
            total_tokens = 0

            self.model.eval()
            with torch.no_grad():
                for sample in tqdm(samples, desc="Calculating perplexity"):
                    inputs = self.tokenizer(
                        sample['text'],
                        return_tensors="pt",
                        truncation=True,
                        max_length=2048
                    ).to(self.device)

                    outputs = self.model(**inputs, labels=inputs['input_ids'])
                    total_loss += outputs.loss.item() * inputs['input_ids'].size(1)
                    total_tokens += inputs['input_ids'].size(1)

            avg_loss = total_loss / total_tokens
            perplexity = np.exp(avg_loss)

            self.metrics['perplexity'] = {
                'value': perplexity,
                'avg_loss': avg_loss,
                'samples_evaluated': len(samples),
                'status': 'calculated'
            }

            logger.info(f"✓ Perplexity: {perplexity:.2f}")

        except Exception as e:
            logger.error(f"❌ Perplexity calculation failed: {e}")
            self.metrics['perplexity'] = {
                'status': 'failed',
                'error': str(e)
            }

    def evaluate_generation_quality(self):
        """Evaluate text generation quality with Sinhala-focused prompts"""
        logger.info("\n✍️ Evaluating Generation Quality...")

        # Enhanced Sinhala prompts based on LLaMA 2 chat format requirements
        prompts = [
            {
                "prompt": "<s>[INST] ශ්‍රී ලංකාව යනු [/INST]", 
                "type": "continuation", 
                "language": "sinhala",
                "description": "Basic Sinhala continuation"
            },
            {
                "prompt": "<s>[INST] කොළඹ නගරය ගැන කියන්න [/INST]", 
                "type": "description", 
                "language": "sinhala",
                "description": "City description"
            },
            {
                "prompt": "<s>[INST] සිංහල භාෂාව ගැන කෙටි ලිපියක් ලියන්න [/INST]", 
                "type": "instruction", 
                "language": "sinhala",
                "description": "Article writing"
            },
            {
                "prompt": "<s>[INST] අද කාලගුණය ගැන කියන්න [/INST]", 
                "type": "completion", 
                "language": "sinhala",
                "description": "Weather topic"
            },
            {
                "prompt": "<s>[INST] Python programming language ගැන කියන්න [/INST]", 
                "type": "mixed", 
                "language": "mixed",
                "description": "Technical topic in Sinhala"
            },
            {
                "prompt": "<s>[INST] සුභ උදෑසනක් සියලු දෙනාටම [/INST]", 
                "type": "greeting", 
                "language": "sinhala",
                "description": "Greeting completion"
            }
        ]

        results = []

        for item in prompts:
            logger.info(f"Testing: {item['description']}")
            
            inputs = self.tokenizer(item['prompt'], return_tensors="pt").to(self.device)

            with torch.no_grad():
                # Greedy decoding
                greedy_output = self.model.generate(
                    **inputs, 
                    max_new_tokens=150, 
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                # Sampling with temperature
                sample_output = self.model.generate(
                    **inputs, 
                    max_new_tokens=150, 
                    do_sample=True, 
                    temperature=0.7, 
                    top_p=0.95,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            greedy_text = self.tokenizer.decode(greedy_output[0], skip_special_tokens=True)
            sample_text = self.tokenizer.decode(sample_output[0], skip_special_tokens=True)

            # Remove prompt from outputs - handle INST format
            if "[/INST]" in greedy_text:
                greedy_text = greedy_text.split("[/INST]", 1)[1].strip()
            if "[/INST]" in sample_text:
                sample_text = sample_text.split("[/INST]", 1)[1].strip()

            # Analyze content
            contains_sinhala = any('\u0D80' <= c <= '\u0DFF' for c in sample_text)
            contains_english = any(c.isascii() and c.isalpha() for c in sample_text)
            
            result = {
                'prompt': item['prompt'],
                'prompt_type': item['type'],
                'prompt_language': item['language'],
                'description': item['description'],
                'greedy_output': greedy_text[:300],  # First 300 chars
                'sample_output': sample_text[:300],
                'greedy_length': len(greedy_text.split()),
                'sample_length': len(sample_text.split()),
                'contains_sinhala': contains_sinhala,
                'contains_english': contains_english,
                'output_quality': self._assess_output_quality(item, sample_text)
            }

            results.append(result)

            logger.info(f"Sample output: {sample_text[:100]}...")

        self.metrics['generation_quality'] = results

    def _assess_output_quality(self, prompt_item: Dict, output: str) -> Dict:
        """Simple quality assessment"""
        quality = {
            'length_appropriate': 10 < len(output.split()) < 200,
            'language_consistent': True,
            'repetitive': self._check_repetition(output),
            'coherent': len(output.strip()) > 5
        }
        
        # Language consistency check
        if prompt_item['language'] == 'sinhala':
            has_sinhala = any('\u0D80' <= c <= '\u0DFF' for c in output)
            quality['language_consistent'] = has_sinhala
        
        return quality

    def _check_repetition(self, text: str) -> bool:
        """Check for excessive repetition"""
        words = text.split()
        if len(words) < 10:
            return False
        
        # Check if more than 30% of words are repeated
        unique_words = set(words)
        repetition_ratio = (len(words) - len(unique_words)) / len(words)
        return repetition_ratio > 0.3

    def evaluate_language_understanding(self):
        """Evaluate understanding with targeted Sinhala tasks"""
        logger.info("\n🧠 Evaluating Language Understanding...")

        test_cases = [
            {
                'prompt': "ශ්‍රී ලංකාවේ අගනුවර කුමක්ද?",
                'expected_keywords': ['කොළඹ', 'ශ්‍රී ජයවර්ධනපුර'],
                'task': 'factual_sinhala',
                'description': 'Capital city question'
            },
            {
                'prompt': "සිංහල අක්ෂර මාලාවේ අකුරු කීයක් තිබේද?",
                'expected_keywords': ['අකුරු', '61', 'හැටએකක්'],
                'task': 'sinhala_knowledge',
                'description': 'Sinhala alphabet question'
            },
            {
                'prompt': "What is the capital of Sri Lanka? Please answer in Sinhala.",
                'expected_keywords': ['කොළඹ', 'ශ්‍රී ජයවර්ධනපුර'],
                'task': 'translation_instruction',
                'description': 'Cross-lingual instruction'
            },
            {
                'prompt': "අද කාලගුණය ගැන කියන්න",
                'expected_keywords': ['කාලගුණය', 'වැස්ස', 'හිරු', 'තාපමානය'],
                'task': 'topic_generation',
                'description': 'Weather topic generation'
            }
        ]

        results = []

        for test in test_cases:
            logger.info(f"Testing: {test['description']}")
            
            inputs = self.tokenizer(test['prompt'], return_tensors="pt").to(self.device)

            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            # Extract response after [/INST] token
            if "[/INST]" in response:
                response = response.split("[/INST]", 1)[1].strip()

            # Check for expected keywords
            found_keywords = []
            for keyword in test['expected_keywords']:
                if keyword.lower() in response.lower():
                    found_keywords.append(keyword)

            success_rate = len(found_keywords) / len(test['expected_keywords']) if test['expected_keywords'] else 0

            result = {
                'task': test['task'],
                'description': test['description'],
                'prompt': test['prompt'],
                'response': response[:200],
                'expected_keywords': test['expected_keywords'],
                'found_keywords': found_keywords,
                'success_rate': success_rate,
                'response_length': len(response.split()),
                'contains_sinhala': any('\u0D80' <= c <= '\u0DFF' for c in response)
            }

            results.append(result)

            logger.info(f"Success: {len(found_keywords)}/{len(test['expected_keywords'])} ({success_rate:.1%})")

        self.metrics['language_understanding'] = results

    def evaluate_speed(self):
        """Evaluate inference speed"""
        logger.info("\n⚡ Evaluating Inference Speed...")

        test_lengths = [10, 50, 100, 200]
        speed_results = {}

        # Warmup
        dummy_input = self.tokenizer("<s>[INST] test [/INST]", return_tensors="pt").to(self.device)
        with torch.no_grad():
            _ = self.model.generate(**dummy_input, max_new_tokens=10)

        for length in test_lengths:
            times = []

            for _ in range(3):  # Reduced runs for baseline
                inputs = self.tokenizer("<s>[INST] ශ්‍රී ලංකාව [/INST]", return_tensors="pt").to(self.device)
                
                torch.cuda.synchronize() if self.device == "cuda" else None
                start = time.time()

                with torch.no_grad():
                    _ = self.model.generate(
                        **inputs, 
                        max_new_tokens=length, 
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )

                torch.cuda.synchronize() if self.device == "cuda" else None
                elapsed = time.time() - start
                times.append(elapsed)

            avg_time = np.mean(times)
            avg_speed = length / avg_time

            speed_results[f"{length}_tokens"] = {
                'avg_time_seconds': avg_time,
                'tokens_per_second': avg_speed,
                'std_time': np.std(times)
            }

            logger.info(f"  {length} tokens: {avg_speed:.1f} tokens/s (±{np.std(times):.2f}s)")

        self.metrics['speed_metrics'] = speed_results

    def save_results(self):
        """Save evaluation results"""
        logger.info("\n💾 Saving Baseline Results...")

        # Save full metrics
        metrics_path = self.results_dir / "baseline_evaluation_metrics.json"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)

        # Create summary for comparison
        summary = {
            'model': self.model_path,
            'evaluation_type': 'baseline_comparison',
            'timestamp': self.metrics['timestamp'],
            'model_size_mb': self.metrics['model_size']['size_mb'],
            'total_parameters': self.metrics['model_size']['total_parameters'],
            'vocab_size': self.metrics['model_info'].get('vocab_size', 'unknown'),
            'avg_tokens_per_second': np.mean([
                v['tokens_per_second'] for v in self.metrics['speed_metrics'].values()
            ]) if self.metrics['speed_metrics'] else None,
            'perplexity': self.metrics['perplexity']['value'] if (
                self.metrics['perplexity'] and 
                self.metrics['perplexity'].get('status') == 'calculated'
            ) else 'not_calculated',
            'language_understanding_avg': np.mean([
                r['success_rate'] for r in self.metrics['language_understanding']
            ]) if self.metrics['language_understanding'] else 0,
            'generation_quality_summary': self._summarize_generation_quality()
        }

        summary_path = self.results_dir / "baseline_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        self.create_comparison_ready_report(summary)

    def _summarize_generation_quality(self) -> Dict:
        """Summarize generation quality metrics"""
        if not self.metrics['generation_quality']:
            return {}
        
        results = self.metrics['generation_quality']
        return {
            'avg_sample_length': np.mean([r['sample_length'] for r in results]),
            'sinhala_capable': sum(1 for r in results if r['contains_sinhala']) / len(results),
            'coherent_outputs': sum(1 for r in results if r['output_quality']['coherent']) / len(results),
            'appropriate_length': sum(1 for r in results if r['output_quality']['length_appropriate']) / len(results)
        }

    def create_comparison_ready_report(self, summary: Dict):
        """Generate markdown report optimized for comparison"""
        report_path = self.results_dir / "baseline_comparison_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Thimira Sinhala LLaMA-2 Baseline Evaluation Report\n\n")
            f.write(f"**Model:** {summary['model']}\n")
            f.write(f"**Date:** {summary['timestamp']}\n")
            f.write(f"**Type:** Baseline Comparison\n\n")
            
            f.write("## Key Metrics for Comparison\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Model Size | {summary['model_size_mb']:.1f} MB |\n")
            f.write(f"| Parameters | {summary['total_parameters']/1e6:.1f}M |\n")
            f.write(f"| Vocabulary Size | {summary['vocab_size']} |\n")
            f.write(f"| Inference Speed | {summary['avg_tokens_per_second']:.1f} tokens/s |\n")
            f.write(f"| Perplexity | {summary['perplexity']} |\n")
            f.write(f"| Language Understanding | {summary['language_understanding_avg']:.1%} |\n")
            f.write(f"| Sinhala Generation | {summary['generation_quality_summary'].get('sinhala_capable', 0):.1%} |\n\n")
            
            f.write("## Notes\n")
            f.write("- This is a baseline evaluation of Thimira's Sinhala LLaMA-2 model\n")
            f.write("- Results can be compared with your local model evaluation\n")
            f.write("- Perplexity calculation requires local validation data\n")

        logger.info(f"✓ Comparison report saved to {report_path}")

    def run(self):
        """Run complete baseline evaluation"""
        logger.info("🚀 THIMIRA SINHALA LLAMA-2 BASELINE EVALUATION")
        logger.info("=" * 60)

        try:
            self.load_model()
            self.evaluate_generation_quality()
            self.evaluate_language_understanding()
            self.evaluate_speed()
            self.evaluate_perplexity()  # Will skip if no data
            self.save_results()

            logger.info("\n" + "=" * 60)
            logger.info("✅ BASELINE EVALUATION COMPLETE!")
            logger.info("=" * 60)
            logger.info(f"Results saved to: {self.results_dir}")
            logger.info("Use these results to compare with your local model performance")

        except Exception as e:
            logger.error(f"\n❌ Baseline evaluation failed: {e}")
            raise

if __name__ == "__main__":
    evaluator = SinhalaLLMBaselineEvaluator(
        model_path="Thimira/sinhala-llama-2-7b-chat-hf",
        max_samples=500  # Reduced for baseline
    )
    evaluator.run()