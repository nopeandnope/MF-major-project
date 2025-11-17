"""
GPU training with open models - Fixed version with proper argument names
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
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class GemmaGPUTrainer:
    def __init__(self):
        self.device = None
        self.model = None
        self.tokenizer = None

        # Setup logging
        logger.add("logs/training.log", rotation="100 MB")

        # Check GPU availability with detailed diagnostics
        if not torch.cuda.is_available():
            logger.error("=" * 70)
            logger.error("❌ NO GPU DETECTED!")
            logger.error("=" * 70)
            logger.error("Possible causes:")
            logger.error("1. PyTorch installed without CUDA support")
            logger.error("   Solution: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
            logger.error("2. NVIDIA GPU drivers not installed")
            logger.error("   Solution: Install latest NVIDIA drivers from nvidia.com")
            logger.error("3. CUDA toolkit not installed")
            logger.error("   Solution: Install CUDA Toolkit 12.x from nvidia.com")
            logger.error("=" * 70)
            logger.warning("Falling back to CPU mode (VERY SLOW)")
            self.device = "cpu"
            self.batch_size = 1
            self.gradient_accumulation_steps = 32
            self.max_length = 256
        else:
            self.device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info("=" * 70)
            logger.info(f"✅ GPU DETECTED: {gpu_name}")
            logger.info(f"✅ VRAM: {gpu_memory:.2f} GB")
            logger.info(f"✅ CUDA Version: {torch.version.cuda}")
            logger.info("=" * 70)

            # Optimized settings for RTX 5070 Ti (12GB VRAM)
            if gpu_memory >= 12:
                self.batch_size = 8  # Increased for 12GB
                self.gradient_accumulation_steps = 2
                self.max_length = 768  # Increased context length
                logger.info("🚀 Using HIGH PERFORMANCE settings (12GB+ VRAM)")
            elif gpu_memory >= 8:
                self.batch_size = 4
                self.gradient_accumulation_steps = 4
                self.max_length = 512
                logger.info("🚀 Using MEDIUM PERFORMANCE settings (8-12GB VRAM)")
            else:
                self.batch_size = 2
                self.gradient_accumulation_steps = 8
                self.max_length = 256
                logger.info("🚀 Using LOW PERFORMANCE settings (<8GB VRAM)")
        
        # Training parameters - FAST MODE
        self.learning_rate = 5e-4  # Higher LR for faster convergence
        self.num_epochs = 1  # Single epoch
        self.warmup_steps = 10  # Minimal warmup
        self.save_steps = 50
        self.eval_steps = 25
        
        # Clear memory
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        
    def load_training_data(self) -> Dataset:
        """Load training data"""
        training_dir = Path("data/training")
        training_files = list(training_dir.glob("gemma_training_*.jsonl"))
        
        if not training_files:
            raise FileNotFoundError("No training files found.")
        
        latest_file = max(training_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Loading training data from: {latest_file}")
        
        data = []
        with open(latest_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    if 'text' in item and len(item['text']) > 100:
                        data.append({'text': item['text']})
                except:
                    continue
        
        logger.info(f"Loaded {len(data)} training examples")

        # Limit for FAST training
        if len(data) > 100:
            data = data[:100]
            logger.info("Limited to 100 examples for FAST training (5-10 min)")
        
        return Dataset.from_list(data)
    
    def setup_model_and_tokenizer(self):
        """Setup model using open models"""
        # Use an open model that doesn't require authentication
        model_name = os.getenv("MODEL_NAME", "microsoft/phi-2")
        
        logger.info(f"Loading model: {model_name}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with proper dtype argument
            if self.device == "cuda":
                # GPU mode with optimizations
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,  # Use FP16 for GPU efficiency
                    trust_remote_code=True,
                    use_cache=False,
                ).to(self.device)  # Manually move to device instead of device_map
            else:
                # CPU mode
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    use_cache=False,
                )
            
            # Apply LoRA for efficient training
            if self.device == "cuda":
                self.model.gradient_checkpointing_enable()
                self.model = prepare_model_for_kbit_training(self.model)
                
                lora_config = LoraConfig(
                    r=4,  # Minimal rank for speed
                    lora_alpha=8,
                    target_modules=["q_proj", "v_proj"],  # Minimal targets
                    lora_dropout=0.05,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                )
                
                self.model = get_peft_model(self.model, lora_config)
            
            self.print_trainable_parameters()
            
            logger.info("Model setup complete")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            logger.info("Trying fallback model...")
            
            # Fallback to tiny model
            model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
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

            logger.info(f"Using fallback model: {model_name}")
    
    def print_trainable_parameters(self):
        """Print model parameter statistics"""
        trainable_params = 0
        all_param = 0
        
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        logger.info(
            f"Trainable params: {trainable_params:,} || "
            f"All params: {all_param:,} || "
            f"Trainable %: {100 * trainable_params / all_param:.2f}"
        )
    
    def tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """Tokenize dataset"""
        logger.info("Tokenizing dataset...")
        
        def tokenize_function(examples):
            # Tokenize the text
            tokenized = self.tokenizer(
                examples['text'],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
            )
            # Return as lists, not tensors
            return tokenized
        
        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=50,
            remove_columns=dataset.column_names,
            desc="Tokenizing"
        )
        
        def add_labels(examples):
            examples["labels"] = examples["input_ids"].copy()
            return examples
        
        tokenized = tokenized.map(add_labels, batched=True)
        
        logger.info("Tokenization complete")
        return tokenized
    
    def train(self):
        """Main training function"""
        logger.info("Starting model training...")
        
        try:
            # Load data
            dataset = self.load_training_data()
            
            # Setup model
            self.setup_model_and_tokenizer()
            
            # Tokenize
            tokenized_dataset = self.tokenize_dataset(dataset)
            
            # Split dataset
            train_test_split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
            train_dataset = train_test_split['train']
            eval_dataset = train_test_split['test']
            
            logger.info(f"Training samples: {len(train_dataset)}")
            logger.info(f"Evaluation samples: {len(eval_dataset)}")
            
            # Training arguments - Fixed for compatibility
            output_dir = Path("models/fine_tuned")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Use the correct argument names for your transformers version
            training_args = TrainingArguments(
                output_dir=str(output_dir),
                num_train_epochs=self.num_epochs,
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                warmup_steps=self.warmup_steps,
                learning_rate=self.learning_rate,
                weight_decay=0.01,
                logging_dir="logs",
                logging_steps=10,
                # Changed from evaluation_strategy to eval_strategy
                eval_strategy="steps",  # This is the correct argument name
                eval_steps=self.eval_steps,
                save_steps=self.save_steps,
                save_total_limit=2,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                fp16=False,  # Disabled to avoid gradient scaling issues
                gradient_checkpointing=self.device == "cuda",
                optim="adamw_torch",
                dataloader_pin_memory=False,
                remove_unused_columns=False,
                report_to="none",
                push_to_hub=False,
                
                # Additional parameters for stability
                max_grad_norm=1.0,
                do_train=True,
                do_eval=True,
                prediction_loss_only=True,
                seed=42,
                data_seed=42,
                bf16=False,  # Disable bf16 for compatibility
                half_precision_backend="auto",
                dataloader_num_workers=0,  # Use 0 for Windows compatibility
                past_index=-1,
                disable_tqdm=False,
                length_column_name="length",
                ddp_find_unused_parameters=False if self.device == "cuda" else None,
                ddp_bucket_cap_mb=25 if self.device == "cuda" else None,
                dataloader_drop_last=False,
                eval_accumulation_steps=None,
                auto_find_batch_size=False,
                full_determinism=False,
                torchdynamo=None,
                ray_scope="last",
                ddp_timeout=1800,
                use_ipex=False,
                use_legacy_prediction_loop=False,
                jit_mode_eval=False,
                use_cpu=self.device == "cpu",
            )
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
            )
            
            # Create trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer,
            )
            
            # Train
            logger.info("Starting training loop...")
            trainer.train()
            
            # Save model
            logger.info("Saving model...")
            trainer.save_model()
            self.tokenizer.save_pretrained(str(output_dir))
            
            # Save training metrics
            with open(output_dir / "training_complete.json", "w") as f:
                json.dump({
                    "status": "completed",
                    "date": datetime.now().isoformat(),
                    "model": os.getenv("MODEL_NAME", "microsoft/phi-2"),
                    "samples_trained": len(train_dataset),
                    "epochs": self.num_epochs
                }, f, indent=2)
            
            logger.info(f"Training completed! Model saved to {output_dir}")
            
            # Test the model
            self.test_model(output_dir)
            
            # Cleanup
            del trainer
            del self.model
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def test_model(self, model_path):
        """Test the trained model"""
        try:
            logger.info("Testing trained model...")
            
            # Test prompt
            test_prompt = """I want to invest ₹100,000 for 5 years. 
My risk tolerance is Moderate and I'm 35 years old.
My investment goal is Wealth Creation.
Please recommend a mutual fund portfolio."""
            
            # Tokenize
            inputs = self.tokenizer(test_prompt, return_tensors="pt", truncation=True, max_length=512)
            
            # Move to device
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            logger.info("Model test response:")
            logger.info(response[:500])  # Log first 500 chars
            
        except Exception as e:
            logger.error(f"Model testing failed: {e}")

def main():
    trainer = GemmaGPUTrainer()
    trainer.train()

if __name__ == "__main__":
    main()