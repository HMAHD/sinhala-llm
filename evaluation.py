#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ['CUDA_HOME'] = '/usr/local/cuda'
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:' + os.environ.get('LD_LIBRARY_PATH', '')

"""
evaluation.py - Comprehensive evaluation of trained Sinhala LLM
Author: SAkash Hasendra
"""

import json
import logging
import torch
import time
import numpy as np
from pathlib import Path
from typing import Dict
from datetime import datetime
from tqdm import tqdm

import nltk
nltk.download('punkt', quiet=True)

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
from rouge_score import rouge_scorer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SinhalaLLMEvaluator:
    """Comprehensive evaluator for Sinhala LLM"""

    def __init__(self, model_path: str = "models/sinhala_llm", max_samples: int = 1000):
        self.model_path = Path(model_path)
        self.max_samples = max_samples
        self.results_dir = Path("evaluation_results")
        self.results_dir.mkdir(exist_ok=True)

        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.metrics = {
            'perplexity': None,
            'generation_quality': {},
            'language_understanding': {},
            'speed_metrics': {},
            'rouge_scores': {},
            'model_size': {},
            'timestamp': datetime.now().isoformat()
        }

    def load_model(self):
        """Load the trained model"""
        logger.info(f"\n📂 Loading model from {self.model_path}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto"
            )

            self.model.eval()

            total_params = sum(p.numel() for p in self.model.parameters())
            model_size_mb = sum(
                p.numel() * p.element_size()
                for p in self.model.parameters()
            ) / 1024 / 1024

            self.metrics['model_size'] = {
                'total_parameters': total_params,
                'size_mb': model_size_mb
            }

            logger.info(f"✓ Model loaded: {total_params/1e6:.1f}M parameters")

        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise

    def evaluate_perplexity(self):
        """Calculate perplexity on validation set"""
        logger.info("\n📊 Evaluating Perplexity...")

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
            'samples_evaluated': len(samples)
        }

        logger.info(f"✓ Perplexity: {perplexity:.2f}")

    def evaluate_generation_quality(self):
        """Evaluate text generation quality"""
        logger.info("\n✍️ Evaluating Generation Quality...")

        prompts = [
            {"prompt": "ශ්‍රී ලංකාව යනු", "type": "continuation", "language": "sinhala"},
            {"prompt": "කොළඹ නගරය", "type": "description", "language": "sinhala"},
            {"prompt": "Python programming භාෂාව", "type": "mixed", "language": "mixed"},
            {"prompt": "### Instruction:\nසිංහල භාෂාව ගැන කෙටි විස්තරයක් ලියන්න\n\n### Response:", "type": "instruction", "language": "sinhala"},
            {"prompt": "Once upon a time in Sri Lanka", "type": "story", "language": "english"},
        ]

        results = []

        for item in prompts:
            inputs = self.tokenizer(item['prompt'], return_tensors="pt").to(self.device)

            with torch.no_grad():
                greedy_output = self.model.generate(**inputs, max_new_tokens=100, do_sample=False)
                sample_output = self.model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7, top_p=0.95)

            greedy_text = self.tokenizer.decode(greedy_output[0], skip_special_tokens=True)
            sample_text = self.tokenizer.decode(sample_output[0], skip_special_tokens=True)

            greedy_text = greedy_text[len(item['prompt']):]
            sample_text = sample_text[len(item['prompt']):]

            results.append({
                'prompt': item['prompt'][:50] + "...",
                'type': item['type'],
                'language': item['language'],
                'greedy_output': greedy_text[:200],
                'sample_output': sample_text[:200],
                'greedy_length': len(greedy_text.split()),
                'sample_length': len(sample_text.split()),
                'contains_sinhala': any('\u0D80' <= c <= '\u0DFF' for c in sample_text),
                'contains_english': any(c.isascii() and c.isalpha() for c in sample_text)
            })

            logger.info(f"\nPrompt: {item['prompt'][:50]}...")
            logger.info(f"Sample: {sample_text[:100]}...")

        self.metrics['generation_quality'] = results

    def evaluate_speed(self):
        """Evaluate inference speed"""
        logger.info("\n⚡ Evaluating Inference Speed...")

        test_lengths = [10, 50, 100, 200]
        speed_results = {}

        dummy_input = self.tokenizer("test", return_tensors="pt").to(self.device)
        _ = self.model.generate(**dummy_input, max_new_tokens=10)

        for length in test_lengths:
            times = []

            for _ in range(5):
                inputs = self.tokenizer("ශ්‍රී ලංකාව" * 5, return_tensors="pt").to(self.device)
                torch.cuda.synchronize() if self.device == "cuda" else None
                start = time.time()

                with torch.no_grad():
                    _ = self.model.generate(**inputs, max_new_tokens=length, do_sample=False)

                torch.cuda.synchronize() if self.device == "cuda" else None
                elapsed = time.time() - start
                times.append(elapsed)

            avg_time = np.mean(times)
            avg_speed = length / avg_time

            speed_results[f"{length}_tokens"] = {
                'avg_time_seconds': avg_time,
                'tokens_per_second': avg_speed
            }

            logger.info(f"  {length} tokens: {avg_speed:.1f} tokens/s")

        self.metrics['speed_metrics'] = speed_results

    def evaluate_rouge_scores(self):
        """Calculate ROUGE scores"""
        logger.info("\n📈 Calculating ROUGE Scores...")

        val_dataset = load_from_disk("data/val_dataset")
        samples = val_dataset.select(range(min(100, len(val_dataset))))
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores_all = {'rouge1': [], 'rouge2': [], 'rougeL': []}

        for sample in tqdm(samples, desc="Computing ROUGE"):
            text = sample['text']
            if "### Response:" not in text:
                continue
            parts = text.split("### Response:")
            if len(parts) != 2:
                continue
            prompt = parts[0] + "### Response:"
            reference = parts[1].replace("</s>", "").strip()

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=100, do_sample=False)

            prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()
            scores = scorer.score(reference[:200], prediction[:200])

            for metric in scores:
                scores_all[metric].append(scores[metric].fmeasure)

        avg_scores = {k: np.mean(v) if v else 0 for k, v in scores_all.items()}
        self.metrics['rouge_scores'] = avg_scores

        logger.info("✓ ROUGE Scores:")
        for k, v in avg_scores.items():
            logger.info(f"  {k.upper()}: {v:.3f}")

    def evaluate_language_understanding(self):
        """Evaluate understanding via targeted prompts"""
        logger.info("\n🧠 Evaluating Language Understanding...")

        cases = [
            {
                'instruction': "පහත වාක්‍යය ඉංග්‍රීසි භාෂාවට පරිවර්තනය කරන්න: ශ්‍රී ලංකාව ලස්සන රටකි",
                'expected_keywords': ['Sri Lanka', 'beautiful', 'country'],
                'task': 'translation'
            },
            {
                'instruction': "කොළඹ නගරය ගැන වාක්‍ය දෙකක් ලියන්න",
                'expected_keywords': ['කොළඹ', 'නගර'],
                'task': 'generation'
            },
            {
                'instruction': "What is the capital of Sri Lanka? Answer in Sinhala",
                'expected_keywords': ['කොළඹ', 'ශ්‍රී ජයවර්ධනපුර'],
                'task': 'qa_mixed'
            }
        ]

        results = []

        for test in cases:
            prompt = f"### Instruction:\n{test['instruction']}\n\n### Response:"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True
                )

            response = self.tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):].strip()

            found = sum(1 for k in test['expected_keywords'] if k.lower() in response.lower())
            results.append({
                'task': test['task'],
                'instruction': test['instruction'],
                'response': response[:200],
                'keywords_found': found,
                'success_rate': found / len(test['expected_keywords'])
            })

            logger.info(f"\nTask: {test['task']}")
            logger.info(f"Success: {found}/{len(test['expected_keywords'])} ({found/len(test['expected_keywords']):.1%})")

        self.metrics['language_understanding'] = results

    def save_results(self):
        """Save evaluation results to disk"""
        logger.info("\n💾 Saving Results...")

        metrics_path = self.results_dir / "evaluation_metrics.json"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)

        summary = {
            'model_path': str(self.model_path),
            'timestamp': self.metrics['timestamp'],
            'perplexity': self.metrics['perplexity']['value'] if self.metrics['perplexity'] else None,
            'model_size_mb': self.metrics['model_size']['size_mb'],
            'avg_tokens_per_second': np.mean([v['tokens_per_second'] for v in self.metrics['speed_metrics'].values()]) if self.metrics['speed_metrics'] else None,
            'rouge_scores': self.metrics['rouge_scores'],
            'language_understanding_score': np.mean([r['success_rate'] for r in self.metrics['language_understanding']]) if self.metrics['language_understanding'] else None
        }

        summary_path = self.results_dir / "evaluation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        self.create_markdown_report(summary)

    def create_markdown_report(self, summary: Dict):
        """Generate markdown evaluation report"""
        path = self.results_dir / "evaluation_report.md"
        with open(path, 'w', encoding='utf-8') as f:
            f.write("# Sinhala LLM Evaluation Report\n\n")
            f.write(f"**Date:** {summary['timestamp']}\n")
            f.write(f"**Model:** {summary['model_path']}\n\n")
            f.write("## Summary Metrics\n\n")
            f.write(f"- **Perplexity:** {summary['perplexity']:.2f}\n")
            f.write(f"- **Model Size:** {summary['model_size_mb']:.1f} MB\n")
            f.write(f"- **Inference Speed:** {summary['avg_tokens_per_second']:.1f} tokens/s\n")
            f.write(f"- **Language Understanding:** {summary['language_understanding_score']:.1%}\n\n")
            f.write("## ROUGE Scores\n")
            for k, v in summary['rouge_scores'].items():
                f.write(f"- **{k.upper()}:** {v:.3f}\n")
        logger.info(f"✓ Markdown report saved to {path}")

    def run(self):
        logger.info("🚀 SINHALA LLM EVALUATION")
        logger.info("=" * 60)

        try:
            self.load_model()
            self.evaluate_perplexity()
            self.evaluate_generation_quality()
            self.evaluate_speed()
            self.evaluate_rouge_scores()
            self.evaluate_language_understanding()
            self.save_results()

            logger.info("\n" + "=" * 60)
            logger.info("✅ EVALUATION COMPLETE!")
            logger.info("=" * 60)
            logger.info(f"Results saved to: {self.results_dir}")
            logger.info("Next step: python gguf.py")

        except Exception as e:
            logger.error(f"\n❌ Evaluation failed: {e}")
            raise

if __name__ == "__main__":
    evaluator = SinhalaLLMEvaluator(
        model_path="polyglots/SinLlama_v01",
        max_samples=1000
    )
    evaluator.run()
