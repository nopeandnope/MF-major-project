"""
Advanced Training System for MF Advisor with 20-Year Data Integration
Includes sophisticated training techniques, data augmentation, and validation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling, EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset as HFDataset
import pandas as pd
import numpy as np
import sqlite3
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import warnings
import gc
import random

warnings.filterwarnings("ignore")

class AdvancedDataGenerator:
    """Generate sophisticated training data from 20-year historical data"""

    def __init__(self, db_path: str = "data/historical_20y.db"):
        self.db_path = db_path
        self.logger = logger

        # Market regimes for context-aware training
        self.market_regimes = {
            'bull_market': {'min_return': 15, 'description': 'Strong uptrend with low volatility'},
            'bear_market': {'max_return': -10, 'description': 'Market decline with high volatility'},
            'sideways': {'return_range': (-5, 10), 'description': 'Range-bound market'},
            'high_volatility': {'min_volatility': 20, 'description': 'Uncertain market conditions'},
            'crisis': {'max_drawdown': -30, 'description': 'Market crisis period'}
        }

    def generate_comprehensive_training_data(self, num_examples: int = 50000) -> List[Dict]:
        """Generate comprehensive training examples from historical data"""

        training_data = []

        conn = sqlite3.connect(self.db_path)

        try:
            # Get funds with good historical data
            funds_query = """
            SELECT f.scheme_code, f.scheme_name, f.category, f.risk_level,
                   p.return_1y, p.return_3y, p.return_5y, p.return_10y,
                   p.volatility_3y, p.sharpe_3y, p.max_drawdown_3y,
                   p.best_1y_return, p.worst_1y_return
            FROM funds f
            JOIN performance_metrics_20y p ON f.scheme_code = p.scheme_code
            WHERE p.return_3y IS NOT NULL AND p.volatility_3y IS NOT NULL
            ORDER BY p.years_of_data DESC
            LIMIT 500
            """

            funds_df = pd.read_sql_query(funds_query, conn)

            if funds_df.empty:
                self.logger.warning("No historical data found, generating synthetic examples")
                return self.generate_synthetic_data(num_examples)

            self.logger.info(f"Generating training data from {len(funds_df)} funds with historical data")

            # Generate diverse investment scenarios
            for _ in range(num_examples):
                try:
                    example = self.create_training_example(funds_df)
                    if example:
                        training_data.append(example)
                except Exception as e:
                    self.logger.error(f"Error creating training example: {str(e)}")
                    continue

            self.logger.info(f"Generated {len(training_data)} training examples")
            return training_data

        finally:
            conn.close()

    def create_training_example(self, funds_df: pd.DataFrame) -> Optional[Dict]:
        """Create a sophisticated training example"""

        # Random investment parameters
        amounts = [25000, 50000, 100000, 200000, 500000, 1000000, 2000000]
        horizons = [1, 2, 3, 5, 7, 10, 15, 20]
        risks = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
        goals = ['Emergency Fund', 'Short Term', 'Wealth Creation', 'Retirement', 'Child Education', 'House Purchase']
        ages = list(range(25, 65))

        amount = random.choice(amounts)
        horizon = random.choice(horizons)
        risk = random.choice(risks)
        goal = random.choice(goals)
        age = random.choice(ages)

        # Filter funds based on risk preference
        suitable_funds = self.filter_funds_by_risk(funds_df, risk)

        if suitable_funds.empty:
            return None

        # Create market context
        market_context = self.generate_market_context()

        # Generate portfolio allocation
        portfolio = self.generate_optimal_allocation(suitable_funds, amount, horizon, risk, age)

        if not portfolio:
            return None

        # Create comprehensive training text
        training_text = self.create_training_text(
            amount, horizon, risk, goal, age, portfolio, market_context, suitable_funds
        )

        return {'text': training_text}

    def filter_funds_by_risk(self, funds_df: pd.DataFrame, risk_level: str) -> pd.DataFrame:
        """Filter funds based on risk tolerance"""

        risk_mapping = {
            'Very Low': ['Liquid Fund', 'Ultra Short Duration', 'Arbitrage Fund'],
            'Low': ['Short Duration', 'Corporate Bond', 'Conservative Hybrid', 'Gilt Fund'],
            'Moderate': ['Large Cap Equity', 'Index Fund', 'Hybrid Fund', 'Balanced Fund'],
            'High': ['Multi Cap Equity', 'Mid Cap Equity', 'ELSS', 'Aggressive Hybrid'],
            'Very High': ['Small Cap Equity', 'Sectoral/Thematic', 'International Fund']
        }

        if risk_level not in risk_mapping:
            return funds_df

        suitable_categories = risk_mapping[risk_level]

        # Include adjacent risk levels for diversification
        if risk_level == 'Moderate':
            suitable_categories.extend(risk_mapping['Low'][:2] + risk_mapping['High'][:2])
        elif risk_level == 'High':
            suitable_categories.extend(risk_mapping['Moderate'][:2])
        elif risk_level == 'Low':
            suitable_categories.extend(risk_mapping['Moderate'][:1])

        return funds_df[funds_df['category'].isin(suitable_categories)]

    def generate_market_context(self) -> Dict:
        """Generate realistic market context"""
        contexts = [
            {'regime': 'bull_market', 'sentiment': 'optimistic', 'volatility': 'low'},
            {'regime': 'bear_market', 'sentiment': 'cautious', 'volatility': 'high'},
            {'regime': 'sideways', 'sentiment': 'neutral', 'volatility': 'moderate'},
            {'regime': 'recovery', 'sentiment': 'hopeful', 'volatility': 'moderate'},
            {'regime': 'crisis', 'sentiment': 'defensive', 'volatility': 'very_high'}
        ]

        return random.choice(contexts)

    def generate_optimal_allocation(self, funds_df: pd.DataFrame, amount: float,
                                  horizon: int, risk: str, age: int) -> Optional[List[Dict]]:
        """Generate optimal portfolio allocation using historical performance"""

        if funds_df.empty:
            return None

        try:
            # Age-based equity allocation adjustment
            base_equity_allocation = max(0, min(100, 100 - age))

            # Risk-based adjustment
            risk_multipliers = {
                'Very Low': 0.1, 'Low': 0.3, 'Moderate': 1.0,
                'High': 1.3, 'Very High': 1.5
            }

            equity_allocation = min(95, base_equity_allocation * risk_multipliers.get(risk, 1.0))

            # Horizon adjustment
            if horizon >= 10:
                equity_allocation = min(95, equity_allocation * 1.2)
            elif horizon <= 2:
                equity_allocation = max(10, equity_allocation * 0.6)

            portfolio = []

            # Select equity funds
            equity_funds = funds_df[funds_df['category'].str.contains('Equity|ELSS|Index', na=False)]
            if not equity_funds.empty and equity_allocation > 10:
                selected_equity = self.select_best_funds(equity_funds, 2, 'return_3y')
                for i, (_, fund) in enumerate(selected_equity.iterrows()):
                    allocation = equity_allocation / len(selected_equity)
                    portfolio.append({
                        'name': fund['scheme_name'],
                        'category': fund['category'],
                        'allocation': round(allocation, 1),
                        'amount': round(amount * allocation / 100, 0),
                        'expected_return': fund['return_3y'] if pd.notna(fund['return_3y']) else 12,
                        'risk_level': fund['risk_level'],
                        'sharpe_ratio': fund['sharpe_3y'] if pd.notna(fund['sharpe_3y']) else 0.8
                    })

            # Select debt funds
            debt_allocation = 100 - sum(f['allocation'] for f in portfolio)
            if debt_allocation > 5:
                debt_funds = funds_df[funds_df['category'].str.contains('Duration|Bond|Liquid|Gilt', na=False)]
                if not debt_funds.empty:
                    selected_debt = self.select_best_funds(debt_funds, 1, 'sharpe_3y')
                    for _, fund in selected_debt.iterrows():
                        portfolio.append({
                            'name': fund['scheme_name'],
                            'category': fund['category'],
                            'allocation': round(debt_allocation, 1),
                            'amount': round(amount * debt_allocation / 100, 0),
                            'expected_return': fund['return_3y'] if pd.notna(fund['return_3y']) else 7,
                            'risk_level': fund['risk_level'],
                            'sharpe_ratio': fund['sharpe_3y'] if pd.notna(fund['sharpe_3y']) else 0.8
                        })

            # Ensure allocations sum to 100%
            total_allocation = sum(f['allocation'] for f in portfolio)
            if total_allocation > 0 and len(portfolio) > 0:
                adjustment_factor = 100 / total_allocation
                for fund in portfolio:
                    fund['allocation'] = round(fund['allocation'] * adjustment_factor, 1)
                    fund['amount'] = round(amount * fund['allocation'] / 100, 0)

            return portfolio if portfolio else None

        except Exception as e:
            self.logger.error(f"Error generating allocation: {str(e)}")
            return None

    def select_best_funds(self, funds_df: pd.DataFrame, count: int, metric: str) -> pd.DataFrame:
        """Select best funds based on specified metric"""

        # Remove funds with missing data
        valid_funds = funds_df.dropna(subset=[metric])

        if valid_funds.empty:
            return funds_df.head(count)

        # Sort by metric (higher is better for returns and ratios)
        return valid_funds.nlargest(count, metric)

    def create_training_text(self, amount: float, horizon: int, risk: str, goal: str,
                           age: int, portfolio: List[Dict], market_context: Dict,
                           funds_df: pd.DataFrame) -> str:
        """Create comprehensive training text"""

        # Calculate portfolio metrics
        weighted_return = sum(f['expected_return'] * f['allocation'] / 100 for f in portfolio)
        total_allocation = sum(f['allocation'] for f in portfolio)

        # SIP calculation
        sip_amount = min(amount / 12, 50000)

        # Create detailed training text
        text = f"""Investment Advisory Session

Client Profile:
- Investment Amount: ₹{amount:,.0f}
- Time Horizon: {horizon} years
- Risk Tolerance: {risk}
- Age: {age} years
- Financial Goal: {goal}

Market Context:
- Current Market Regime: {market_context['regime'].replace('_', ' ').title()}
- Investor Sentiment: {market_context['sentiment'].title()}
- Market Volatility: {market_context['volatility'].replace('_', ' ').title()}

Recommended Portfolio Allocation:

"""

        for i, fund in enumerate(portfolio, 1):
            text += f"""{i}. {fund['name'][:50]}
   Category: {fund['category']}
   Allocation: {fund['allocation']:.1f}% (₹{fund['amount']:,.0f})
   Expected Return: {fund['expected_return']:.1f}% p.a.
   Risk Level: {fund['risk_level']}
   Sharpe Ratio: {fund['sharpe_ratio']:.2f}

"""

        text += f"""Portfolio Summary:
- Total Allocation: {total_allocation:.1f}%
- Expected Portfolio Return: {weighted_return:.1f}% per annum
- Investment Approach: {"Lumpsum + SIP" if amount > 100000 else "SIP"}
- Recommended SIP: ₹{sip_amount:,.0f} per month

Investment Strategy:
- Primary Focus: {self.get_strategy_focus(risk, horizon)}
- Asset Allocation: {self.get_allocation_strategy(portfolio)}
- Rebalancing: {self.get_rebalancing_advice(horizon)}

Risk Management:
- Maximum Expected Drawdown: {self.estimate_max_drawdown(portfolio):.1f}%
- Volatility Estimate: {self.estimate_volatility(portfolio):.1f}%
- Suitable for: {self.get_suitability(risk, age, horizon)}

Tax Considerations:
- Equity Taxation: Long-term capital gains > ₹1L taxed at 10%
- Debt Taxation: Long-term gains taxed with indexation
- ELSS Benefits: 80C deduction up to ₹1.5L available

Action Plan:
1. Start with {portfolio[0]['name'][:30]}... (₹{portfolio[0]['amount']:,.0f})
2. {"Set up SIP" if sip_amount > 1000 else "Consider lumpsum investment"}
3. Review and rebalance {"annually" if horizon > 5 else "every 6 months"}
4. Monitor performance against benchmark indices

This portfolio is designed based on historical performance analysis of {len(funds_df)} funds with average {self.get_avg_track_record(funds_df)} years of track record.

Disclaimer: Past performance does not guarantee future results. Please consult with a SEBI-registered investment advisor for personalized advice."""

        return text

    def get_strategy_focus(self, risk: str, horizon: int) -> str:
        """Get investment strategy focus"""
        if horizon >= 10:
            return f"Long-term wealth creation with {risk.lower()} risk profile"
        elif horizon >= 5:
            return f"Medium-term growth with {risk.lower()} risk tolerance"
        else:
            return f"Short to medium-term goals with {risk.lower()} risk approach"

    def get_allocation_strategy(self, portfolio: List[Dict]) -> str:
        """Describe allocation strategy"""
        equity_allocation = sum(f['allocation'] for f in portfolio if 'Equity' in f['category'] or 'ELSS' in f['category'])
        debt_allocation = 100 - equity_allocation

        if equity_allocation > 70:
            return f"Aggressive growth ({equity_allocation:.0f}% equity, {debt_allocation:.0f}% debt)"
        elif equity_allocation > 40:
            return f"Balanced approach ({equity_allocation:.0f}% equity, {debt_allocation:.0f}% debt)"
        else:
            return f"Conservative strategy ({equity_allocation:.0f}% equity, {debt_allocation:.0f}% debt)"

    def get_rebalancing_advice(self, horizon: int) -> str:
        """Get rebalancing frequency advice"""
        if horizon >= 10:
            return "Annual rebalancing to maintain target allocation"
        elif horizon >= 3:
            return "Semi-annual review and rebalancing"
        else:
            return "Quarterly monitoring with tactical adjustments"

    def estimate_max_drawdown(self, portfolio: List[Dict]) -> float:
        """Estimate portfolio maximum drawdown"""
        # Simplified estimation based on allocation
        equity_allocation = sum(f['allocation'] for f in portfolio if 'Equity' in f['category'])

        if equity_allocation > 80:
            return -25.0
        elif equity_allocation > 60:
            return -20.0
        elif equity_allocation > 40:
            return -15.0
        else:
            return -8.0

    def estimate_volatility(self, portfolio: List[Dict]) -> float:
        """Estimate portfolio volatility"""
        equity_allocation = sum(f['allocation'] for f in portfolio if 'Equity' in f['category'])

        if equity_allocation > 80:
            return 18.0
        elif equity_allocation > 60:
            return 14.0
        elif equity_allocation > 40:
            return 10.0
        else:
            return 6.0

    def get_suitability(self, risk: str, age: int, horizon: int) -> str:
        """Get investment suitability description"""
        descriptors = []

        if age < 35:
            descriptors.append("young investors")
        elif age < 50:
            descriptors.append("middle-aged professionals")
        else:
            descriptors.append("pre-retirement investors")

        if horizon >= 10:
            descriptors.append("long-term wealth builders")
        elif horizon >= 5:
            descriptors.append("medium-term goal planners")
        else:
            descriptors.append("short-term investors")

        if risk == 'High' or risk == 'Very High':
            descriptors.append("growth-oriented individuals")
        elif risk == 'Low' or risk == 'Very Low':
            descriptors.append("conservative investors")

        return ", ".join(descriptors)

    def get_avg_track_record(self, funds_df: pd.DataFrame) -> int:
        """Get average track record of funds"""
        return random.randint(8, 15)  # Simplified for training data

    def generate_synthetic_data(self, num_examples: int) -> List[Dict]:
        """Generate synthetic training data when no historical data available"""

        synthetic_data = []

        for _ in range(num_examples):
            # Generate random but realistic parameters
            amount = random.choice([50000, 100000, 200000, 500000, 1000000])
            horizon = random.choice([1, 3, 5, 7, 10, 15])
            risk = random.choice(['Low', 'Moderate', 'High'])
            age = random.randint(25, 60)
            goal = random.choice(['Wealth Creation', 'Retirement', 'Child Education'])

            # Simple synthetic portfolio
            if risk == 'High':
                allocation = "70% Large Cap Equity, 20% Mid Cap, 10% Debt"
                expected_return = "12-15% annually"
            elif risk == 'Low':
                allocation = "20% Equity, 70% Debt, 10% Liquid"
                expected_return = "7-9% annually"
            else:
                allocation = "50% Large Cap, 30% Debt, 20% Hybrid"
                expected_return = "9-12% annually"

            text = f"""Investment Profile: ₹{amount:,} for {horizon} years, {risk} risk, Age {age}, Goal: {goal}
Recommended Allocation: {allocation}
Expected Returns: {expected_return}
Strategy: Balanced approach suitable for {risk.lower()} risk investors"""

            synthetic_data.append({'text': text})

        return synthetic_data


class AdvancedTrainingSystem:
    """Advanced training system with sophisticated techniques"""

    def __init__(self, model_name: str = "microsoft/phi-2"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = logger

        # Enhanced training configuration
        self.config = {
            'learning_rate': 2e-4,
            'batch_size': 2 if self.device == "cuda" else 1,
            'gradient_accumulation_steps': 16,
            'max_length': 1024,  # Increased for detailed responses
            'num_epochs': 5,
            'warmup_ratio': 0.1,
            'weight_decay': 0.01,
            'max_grad_norm': 1.0,
            'fp16': self.device == "cuda",
            'gradient_checkpointing': True,
            'save_steps': 100,
            'eval_steps': 100,
            'logging_steps': 25
        }

        # LoRA configuration for efficient training
        self.lora_config = LoraConfig(
            r=32,  # Increased rank for better performance
            lora_alpha=64,  # Increased alpha for stronger adaptation
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            modules_to_save=["embed_tokens", "lm_head"]  # Fine-tune embeddings and head
        )

        self.logger.add("logs/advanced_training.log", rotation="100 MB")

    def prepare_training_data(self) -> Tuple[HFDataset, HFDataset]:
        """Prepare comprehensive training and validation data"""

        self.logger.info("Preparing comprehensive training data...")

        # Generate data from historical records
        data_generator = AdvancedDataGenerator()
        training_examples = data_generator.generate_comprehensive_training_data(num_examples=30000)

        if not training_examples:
            raise ValueError("No training data generated!")

        self.logger.info(f"Generated {len(training_examples)} training examples")

        # Split data
        train_data, val_data = train_test_split(
            training_examples, test_size=0.15, random_state=42, shuffle=True
        )

        self.logger.info(f"Training samples: {len(train_data)}")
        self.logger.info(f"Validation samples: {len(val_data)}")

        # Create HuggingFace datasets
        train_dataset = HFDataset.from_list(train_data)
        val_dataset = HFDataset.from_list(val_data)

        return train_dataset, val_dataset

    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer with advanced configuration"""

        self.logger.info(f"Loading model: {self.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left"
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load model with optimized settings
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            use_cache=False,  # Disable during training
            attn_implementation="flash_attention_2" if self.device == "cuda" else None
        )

        # Enable gradient checkpointing for memory efficiency
        if self.device == "cuda":
            self.model.gradient_checkpointing_enable()
            self.model = prepare_model_for_kbit_training(self.model)

        # Apply LoRA
        self.model = get_peft_model(self.model, self.lora_config)
        self.model.train()

        # Print trainable parameters
        self.print_trainable_parameters()

    def print_trainable_parameters(self):
        """Print the number of trainable parameters"""
        trainable_params = 0
        all_param = 0

        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        self.logger.info(
            f"Trainable parameters: {trainable_params:,} || "
            f"All parameters: {all_param:,} || "
            f"Trainable %: {100 * trainable_params / all_param:.4f}"
        )

    def tokenize_data(self, dataset: HFDataset) -> HFDataset:
        """Tokenize dataset with advanced preprocessing"""

        def tokenize_function(examples):
            # Add instruction prompting for better learning
            texts = []
            for text in examples['text']:
                prompted_text = f"<|system|>You are a professional mutual fund advisor with 20 years of experience. Provide detailed, accurate investment advice based on Indian market conditions.<|user|>{text}<|assistant|>"
                texts.append(prompted_text)

            # Tokenize
            tokenized = self.tokenizer(
                texts,
                max_length=self.config['max_length'],
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )

            # Set labels (same as input_ids for causal LM)
            tokenized["labels"] = tokenized["input_ids"].clone()

            return tokenized

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset"
        )

        return tokenized_dataset

    def create_training_arguments(self) -> TrainingArguments:
        """Create advanced training arguments"""

        output_dir = Path("models/advanced_mf_advisor")
        output_dir.mkdir(parents=True, exist_ok=True)

        return TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config['num_epochs'],
            per_device_train_batch_size=self.config['batch_size'],
            per_device_eval_batch_size=self.config['batch_size'],
            gradient_accumulation_steps=self.config['gradient_accumulation_steps'],
            learning_rate=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            warmup_ratio=self.config['warmup_ratio'],
            max_grad_norm=self.config['max_grad_norm'],

            # Logging and saving
            logging_dir="logs/tensorboard",
            logging_steps=self.config['logging_steps'],
            save_steps=self.config['save_steps'],
            save_total_limit=3,

            # Evaluation
            evaluation_strategy="steps",
            eval_steps=self.config['eval_steps'],
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,

            # Optimization
            fp16=self.config['fp16'] and self.device == "cuda",
            gradient_checkpointing=self.config['gradient_checkpointing'],
            dataloader_num_workers=2,
            dataloader_pin_memory=True,

            # Advanced features
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            seed=42,
            data_seed=42,

            # Reporting
            report_to="tensorboard",
            run_name=f"advanced_mf_advisor_{datetime.now().strftime('%Y%m%d_%H%M%S')}",

            # Memory optimization
            remove_unused_columns=False,
            push_to_hub=False
        )

    def train_model(self):
        """Train the model with advanced techniques"""

        self.logger.info("Starting advanced model training...")

        try:
            # Prepare data
            train_dataset, val_dataset = self.prepare_training_data()

            # Setup model
            self.setup_model_and_tokenizer()

            # Tokenize datasets
            tokenized_train = self.tokenize_data(train_dataset)
            tokenized_val = self.tokenize_data(val_dataset)

            # Create data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
                pad_to_multiple_of=8 if self.config['fp16'] else None,
            )

            # Training arguments
            training_args = self.create_training_arguments()

            # Create trainer with callbacks
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_val,
                data_collator=data_collator,
                tokenizer=self.tokenizer,
                callbacks=[
                    EarlyStoppingCallback(
                        early_stopping_patience=3,
                        early_stopping_threshold=0.001
                    )
                ]
            )

            # Train model
            self.logger.info("Starting training process...")
            train_result = trainer.train()

            # Save final model
            trainer.save_model()
            self.tokenizer.save_pretrained(training_args.output_dir)

            # Save training metrics
            self.save_training_metrics(train_result, training_args.output_dir)

            # Test the trained model
            self.test_trained_model(training_args.output_dir)

            self.logger.info("Advanced training completed successfully!")

        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise

        finally:
            # Cleanup
            if hasattr(self, 'model'):
                del self.model
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    def save_training_metrics(self, train_result, output_dir: str):
        """Save comprehensive training metrics"""

        metrics = {
            'training_completed': datetime.now().isoformat(),
            'train_runtime': train_result.training_time,
            'train_samples_per_second': train_result.train_samples_per_second,
            'final_loss': float(train_result.training_loss),
            'model_name': self.model_name,
            'config': self.config,
            'lora_config': {
                'r': self.lora_config.r,
                'lora_alpha': self.lora_config.lora_alpha,
                'lora_dropout': self.lora_config.lora_dropout,
                'target_modules': self.lora_config.target_modules
            }
        }

        metrics_path = Path(output_dir) / "training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        self.logger.info(f"Training metrics saved to {metrics_path}")

    def test_trained_model(self, model_path: str):
        """Test the trained model with sample inputs"""

        self.logger.info("Testing trained model...")

        test_cases = [
            {
                'amount': 500000,
                'horizon': 7,
                'risk': 'Moderate',
                'age': 35,
                'goal': 'Wealth Creation'
            },
            {
                'amount': 100000,
                'horizon': 3,
                'risk': 'Low',
                'age': 50,
                'goal': 'Emergency Fund'
            },
            {
                'amount': 1000000,
                'horizon': 15,
                'risk': 'High',
                'age': 30,
                'goal': 'Retirement'
            }
        ]

        for i, case in enumerate(test_cases, 1):
            prompt = f"""Investment Profile:
Amount: ₹{case['amount']:,}
Horizon: {case['horizon']} years
Risk Tolerance: {case['risk']}
Age: {case['age']} years
Goal: {case['goal']}

Recommended Portfolio:"""

            response = self.generate_response(prompt, model_path)

            self.logger.info(f"\nTest Case {i}:")
            self.logger.info(f"Input: {prompt[:100]}...")
            self.logger.info(f"Response: {response[:200]}...")

    def generate_response(self, prompt: str, model_path: str) -> str:
        """Generate response from trained model"""

        try:
            # Load trained model for inference
            from peft import PeftModel

            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )

            model = PeftModel.from_pretrained(base_model, model_path)
            model.eval()

            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True
            )

            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_part = full_response[len(prompt):].strip()

            return generated_part

        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return "Error generating response"


def run_advanced_training():
    """Run the advanced training system"""

    logger.info("=" * 60)
    logger.info("ADVANCED MF ADVISOR TRAINING SYSTEM")
    logger.info("=" * 60)

    try:
        # First, collect 20-year data
        logger.info("Step 1: Collecting 20-year historical data...")

        from src.enhanced_20_year_collector import run_20_year_collection
        asyncio.run(run_20_year_collection())

        logger.info("Step 2: Training advanced model...")

        # Train advanced model
        trainer = AdvancedTrainingSystem()
        trainer.train_model()

        logger.info("=" * 60)
        logger.info("ADVANCED TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Advanced training failed: {str(e)}")
        raise


if __name__ == "__main__":
    import asyncio
    run_advanced_training()