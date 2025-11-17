"""
Enhanced training with proper gradient handling - FIXED
"""
import torch
import gc
import json
from pathlib import Path
from datetime import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, prepare_model_for_kbit_training
from datasets import Dataset
from loguru import logger
import warnings
warnings.filterwarnings("ignore")

class EnhancedTrainer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Optimized settings
        self.batch_size = 2
        self.gradient_accumulation_steps = 8
        self.max_length = 512
        self.learning_rate = 5e-5
        self.num_epochs = 3
        self.warmup_steps = 100
        
        logger.add("logs/enhanced_training.log")
        
    def load_training_data(self) -> Dataset:
        """Load training data"""
        training_dir = Path("data/training")
        training_files = list(training_dir.glob("gemma_training_*.jsonl"))
        
        if not training_files:
            raise FileNotFoundError("No training files in data/training/")
        
        latest_file = max(training_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Loading from: {latest_file}")
        
        data = []
        with open(latest_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    if 'text' in item and len(item['text']) > 100:
                        text = item['text']
                        if any(keyword in text.lower() for keyword in ['invest', 'fund', 'portfolio', 'amount', '₹']):
                            data.append({'text': text})
                except:
                    continue
        
        logger.info(f"Loaded {len(data)} training examples")
        return Dataset.from_list(data)
    
    def setup_model(self):
        """Setup model with proper gradient configuration"""
        logger.info("Setting up model for training...")
        
        # Always load base model and apply LoRA for training
        model_name = "microsoft/phi-2"
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        logger.info(f"Loading base model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            use_cache=False,  # Disable cache for training
        )
        
        # Enable gradient checkpointing for memory efficiency
        if self.device == "cuda":
            self.model.gradient_checkpointing_enable()
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Apply LoRA configuration
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Get PEFT model
        self.model = get_peft_model(self.model, lora_config)
        
        # Enable training mode
        self.model.train()
        
        # Print trainable parameters
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
    
    def train(self):
        """Enhanced training"""
        logger.info("Starting enhanced training...")
        
        # Load data
        dataset = self.load_training_data()
        
        if len(dataset) == 0:
            logger.error("No data to train on!")
            return
        
        # Setup model
        self.setup_model()
        
        # Tokenize
        def tokenize(examples):
            return self.tokenizer(
                examples['text'],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
            )
        
        tokenized = dataset.map(
            tokenize, 
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Add labels
        def add_labels(examples):
            examples["labels"] = examples["input_ids"].copy()
            return examples
        
        tokenized = tokenized.map(add_labels, batched=True)
        
        # Split data
        split = tokenized.train_test_split(test_size=0.1, seed=42)
        train_dataset = split['train']
        eval_dataset = split['test']
        
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Evaluation samples: {len(eval_dataset)}")
        
        # Training arguments
        output_dir = Path("models/enhanced_model")
        output_dir.mkdir(parents=True, exist_ok=True)
        
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
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=self.device == "cuda",
            gradient_checkpointing=self.device == "cuda",
            optim="adamw_torch",
            report_to="none",
            seed=42,
            push_to_hub=False,
            remove_unused_columns=False,
            dataloader_num_workers=0,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=None,
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
        
        logger.info("Starting training loop...")
        
        try:
            # Train
            trainer.train()
            
            # Save model
            logger.info("Saving enhanced model...")
            trainer.save_model()
            self.tokenizer.save_pretrained(str(output_dir))
            
            # Save training info
            with open(output_dir / "training_info.json", "w") as f:
                json.dump({
                    "completed": datetime.now().isoformat(),
                    "epochs": self.num_epochs,
                    "train_samples": len(train_dataset),
                    "eval_samples": len(eval_dataset),
                    "final_loss": trainer.state.best_metric if trainer.state.best_metric else "N/A"
                }, f, indent=2)
            
            logger.info(f"Training complete! Model saved to {output_dir}")
            
            # Test the model
            self.test_model()
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        
        finally:
            # Cleanup
            del trainer
            del self.model
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
    
    def test_model(self):
        """Test the trained model"""
        logger.info("Testing enhanced model...")
        
        # Load the saved model for testing
        model_path = Path("models/enhanced_model")
        if not model_path.exists():
            logger.warning("Enhanced model not found, using base model")
            return
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-2",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )
        
        # Load LoRA weights
        model = PeftModel.from_pretrained(model, str(model_path))
        model.eval()
        
        # Test cases
        test_cases = [
            "Investment Profile:\nAmount: ₹100,000\nHorizon: 5 years\nRisk: Moderate\nAge: 35\nGoal: Wealth Creation\n\nRecommended Portfolio:",
            "Investment Profile:\nAmount: ₹500,000\nHorizon: 10 years\nRisk: High\nAge: 30\nGoal: Retirement\n\nRecommended Portfolio:",
        ]
        
        for test_prompt in test_cases:
            inputs = tokenizer(test_prompt, return_tensors="pt", max_length=256, truncation=True)
            
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = response[len(test_prompt):]
            
            logger.info(f"\nPrompt: {test_prompt[:100]}...")
            logger.info(f"Response: {generated_text[:300]}")

if __name__ == "__main__":
    trainer = EnhancedTrainer()
    trainer.train()