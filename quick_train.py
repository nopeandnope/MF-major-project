"""
Quick Training Script - Optimized for Speed (5-10 minutes)
For faster training with minimal data
"""
import torch
import gc
import json
import os
from pathlib import Path
from datetime import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import Dataset
from loguru import logger
import warnings
warnings.filterwarnings("ignore")

class QuickTrainer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None

        # Setup logging
        logger.add("logs/quick_training.log", rotation="100 MB")

        # GPU detection with diagnostics
        if not torch.cuda.is_available():
            logger.warning("=" * 70)
            logger.warning("⚠️ NO GPU DETECTED - Using CPU (VERY SLOW)")
            logger.warning("=" * 70)
            self.batch_size = 1
            self.gradient_accumulation_steps = 8
        else:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info("=" * 70)
            logger.info(f"✅ GPU: {gpu_name}")
            logger.info(f"✅ VRAM: {gpu_memory:.2f} GB")
            logger.info("=" * 70)
            # Optimized for RTX 5070 Ti (12GB)
            if gpu_memory >= 12:
                self.batch_size = 8
                self.gradient_accumulation_steps = 2
            else:
                self.batch_size = 4
                self.gradient_accumulation_steps = 2

        # FAST training settings
        self.max_length = 256  # Shorter sequences
        self.learning_rate = 1e-3  # High LR
        self.num_epochs = 1
        self.warmup_steps = 5
        self.save_steps = 50
        self.max_samples = 50  # VERY LIMITED DATA

        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Gradient accumulation: {self.gradient_accumulation_steps}")

    def load_training_data(self) -> Dataset:
        """Load minimal training data"""
        training_dir = Path("data/training")
        training_files = list(training_dir.glob("gemma_training_*.jsonl"))

        if not training_files:
            raise FileNotFoundError("No training files found.")

        latest_file = max(training_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Loading from: {latest_file}")

        data = []
        with open(latest_file, 'r', encoding='utf-8') as f:
            for line in f:
                if len(data) >= self.max_samples:
                    break
                try:
                    item = json.loads(line.strip())
                    if 'text' in item and len(item['text']) > 50:
                        data.append({'text': item['text'][:1000]})  # Truncate long texts
                except:
                    continue

        logger.info(f"Loaded {len(data)} samples for QUICK training")
        return Dataset.from_list(data)

    def setup_model_and_tokenizer(self):
        """Setup lightweight model"""
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

        logger.info(f"Loading {model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
            ).to(self.device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
            )

        if self.device == "cuda":
            self.model.gradient_checkpointing_enable()
            self.model = prepare_model_for_kbit_training(self.model)

            lora_config = LoraConfig(
                r=2,  # Minimal rank
                lora_alpha=4,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )

            self.model = get_peft_model(self.model, lora_config)

        logger.info("Model loaded successfully")

    def tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """Quick tokenization"""
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
            )

        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
        )

        tokenized = tokenized.map(
            lambda examples: {"labels": examples["input_ids"]},
            batched=True
        )

        return tokenized

    def train(self):
        """Quick training"""
        logger.info("Starting QUICK training...")

        try:
            dataset = self.load_training_data()
            self.setup_model_and_tokenizer()
            tokenized_dataset = self.tokenize_dataset(dataset)

            # Split 90/10
            split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)

            output_dir = Path("models/quick_trained")
            output_dir.mkdir(parents=True, exist_ok=True)

            training_args = TrainingArguments(
                output_dir=str(output_dir),
                num_train_epochs=self.num_epochs,
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                warmup_steps=self.warmup_steps,
                learning_rate=self.learning_rate,
                logging_steps=5,
                eval_strategy="steps",
                eval_steps=20,
                save_steps=self.save_steps,
                save_total_limit=1,
                fp16=self.device == "cuda",
                gradient_checkpointing=self.device == "cuda",
                report_to="none",
                remove_unused_columns=False,
                dataloader_num_workers=0,
            )

            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=split['train'],
                eval_dataset=split['test'],
                data_collator=data_collator,
            )

            logger.info("Training...")
            trainer.train()

            logger.info("Saving model...")
            trainer.save_model()
            self.tokenizer.save_pretrained(str(output_dir))

            with open(output_dir / "training_info.json", "w") as f:
                json.dump({
                    "status": "completed",
                    "date": datetime.now().isoformat(),
                    "samples": len(split['train']),
                    "training_mode": "QUICK"
                }, f, indent=2)

            logger.info(f"✅ QUICK training completed! Model saved to {output_dir}")

            # Cleanup
            del trainer, self.model
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

def main():
    print("="*70)
    print("QUICK TRAINING MODE - Optimized for Speed (5-10 minutes)")
    print("="*70)
    print("\nSettings:")
    print("  - Model: TinyLlama-1.1B")
    print("  - Samples: 50")
    print("  - LoRA rank: 2")
    print("  - Sequence length: 256")
    print("  - Estimated time: 5-10 minutes\n")

    trainer = QuickTrainer()
    trainer.train()

    print("\n" + "="*70)
    print("✅ Training Complete!")
    print("="*70)

if __name__ == "__main__":
    main()
