"""
Fine-tuning Script for Qwen3-4B Banking Chatbot
Script chính để fine-tune model với LoRA
"""

import os
import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    Trainer
)
from peft import get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from typing import Dict, List, Any
import logging
from datetime import datetime

# Import custom configurations
from lora_config import LoRAConfigManager, get_recommended_config
from training_config import TrainingConfigManager, get_recommended_training_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QwenBankingFineTuner:
    def __init__(self, 
                 model_name: str = "Qwen/Qwen3-4B",
                 data_dir: str = "e:/HDBank_Hackathon/source/data/processed/train_split",
                 output_dir: str = "./qwen-banking-lora"):
        
        self.model_name = model_name
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None
        
        logger.info(f"Initializing fine-tuner for {model_name}")
        logger.info(f"Device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def load_tokenizer_and_model(self):
        """Load tokenizer and model"""
        logger.info("Loading tokenizer and model...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # Add pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else None
        )
        
        # Prepare model for LoRA
        self.model = prepare_model_for_kbit_training(self.model)
        
        logger.info("Model and tokenizer loaded successfully")
    
    def load_datasets(self):
        """Load training and validation datasets"""
        logger.info("Loading datasets...")
        
        def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
            data = []
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            return data
        
        # Load data
        train_data = load_jsonl(os.path.join(self.data_dir, "train.jsonl"))
        val_data = load_jsonl(os.path.join(self.data_dir, "validation.jsonl"))
        
        # Convert to datasets
        self.train_dataset = Dataset.from_list(train_data)
        self.eval_dataset = Dataset.from_list(val_data)
        
        logger.info(f"Loaded {len(self.train_dataset)} training samples")
        logger.info(f"Loaded {len(self.eval_dataset)} validation samples")
    
    def preprocess_function(self, examples):
        """Preprocess examples for training"""
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
        
        for messages in examples["messages"]:
            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            
            # Tokenize
            tokenized = self.tokenizer(
                text,
                truncation=True,
                max_length=1024,
                padding=False,
                return_tensors=None
            )
            
            model_inputs["input_ids"].append(tokenized["input_ids"])
            model_inputs["attention_mask"].append(tokenized["attention_mask"])
            
            # Labels are the same as input_ids for causal LM
            model_inputs["labels"].append(tokenized["input_ids"].copy())
        
        return model_inputs
    
    def setup_lora(self, available_vram_gb: float = 12.0):
        """Setup LoRA configuration"""
        logger.info("Setting up LoRA...")
        
        # Get recommended LoRA config
        lora_config = get_recommended_config(available_vram_gb)
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        logger.info("LoRA setup completed")
    
    def train(self, available_vram_gb: float = 12.0):
        """Main training function"""
        logger.info("Starting training...")
        
        # Preprocess datasets
        train_dataset = self.train_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=self.train_dataset.column_names
        )
        
        eval_dataset = self.eval_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=self.eval_dataset.column_names
        )
        
        # Get training arguments
        training_args = get_recommended_training_config(
            available_vram_gb, 
            len(train_dataset)
        )
        training_args.output_dir = self.output_dir
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            return_tensors="pt"
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Start training
        start_time = datetime.now()
        logger.info(f"Training started at {start_time}")
        
        trainer.train()
        
        end_time = datetime.now()
        training_duration = end_time - start_time
        logger.info(f"Training completed at {end_time}")
        logger.info(f"Training duration: {training_duration}")
        
        # Save the final model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        logger.info(f"Model saved to {self.output_dir}")
        
        return trainer
    
    def run_fine_tuning(self, available_vram_gb: float = 12.0):
        """Complete fine-tuning pipeline"""
        try:
            # Load model and tokenizer
            self.load_tokenizer_and_model()
            
            # Load datasets
            self.load_datasets()
            
            # Setup LoRA
            self.setup_lora(available_vram_gb)
            
            # Train model
            trainer = self.train(available_vram_gb)
            
            logger.info("Fine-tuning completed successfully!")
            return trainer
            
        except Exception as e:
            logger.error(f"Error during fine-tuning: {str(e)}")
            raise

def main():
    """Main function"""
    # Configuration
    MODEL_NAME = "Qwen/Qwen3-4B"
    DATA_DIR = "e:/HDBank_Hackathon/source/data/processed/train_split"
    OUTPUT_DIR = "./qwen-banking-lora"
    
    # Detect available VRAM
    if torch.cuda.is_available():
        available_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"Available VRAM: {available_vram:.1f} GB")
    else:
        available_vram = 8.0  # Default for CPU
        logger.warning("CUDA not available, using CPU")
    
    # Initialize and run fine-tuning
    fine_tuner = QwenBankingFineTuner(
        model_name=MODEL_NAME,
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR
    )
    
    trainer = fine_tuner.run_fine_tuning(available_vram)
    
    # Print final statistics
    if trainer:
        logger.info("=== Training Statistics ===")
        logger.info(f"Final training loss: {trainer.state.log_history[-1].get('train_loss', 'N/A')}")
        logger.info(f"Final eval loss: {trainer.state.log_history[-1].get('eval_loss', 'N/A')}")

if __name__ == "__main__":
    main()