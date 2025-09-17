"""
Complete Fine-tuning Pipeline for Qwen2-4B Banking Chatbot
Script tổng hợp chạy toàn bộ pipeline từ preprocessing đến merge model
"""

import os
import sys
import torch
import logging
from datetime import datetime
import argparse

# Add scripts directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import pipeline components
from data_preprocessing import BankingDataPreprocessor
from fine_tune_qwen import QwenBankingFineTuner
from merge_model import ModelMerger

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fine_tuning_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CompletePipeline:
    def __init__(self, config: dict):
        self.config = config
        self.start_time = datetime.now()
        
        # Paths
        self.csv_path = config['csv_path']
        self.processed_data_dir = config['processed_data_dir']
        self.lora_output_dir = config['lora_output_dir']
        self.merged_output_dir = config['merged_output_dir']
        
        # Model config
        self.model_name = config.get('model_name', 'Qwen/Qwen2-4B')
        self.available_vram = self._detect_vram()
        
        logger.info("=== HDBank Chatbot Fine-tuning Pipeline ===")
        logger.info(f"Start time: {self.start_time}")
        logger.info(f"Available VRAM: {self.available_vram:.1f} GB")
        logger.info(f"Model: {self.model_name}")
    
    def _detect_vram(self) -> float:
        """Detect available VRAM"""
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            return vram
        else:
            logger.warning("CUDA not available, using CPU")
            return 8.0  # Default for CPU
    
    def step_1_preprocess_data(self) -> bool:
        """Step 1: Data Preprocessing"""
        logger.info("=== STEP 1: DATA PREPROCESSING ===")
        
        try:
            preprocessor = BankingDataPreprocessor(
                csv_path=self.csv_path,
                output_dir=self.processed_data_dir
            )
            
            train_data, val_data, test_data = preprocessor.run_preprocessing()
            
            logger.info(f"✓ Preprocessing completed")
            logger.info(f"  - Train samples: {len(train_data)}")
            logger.info(f"  - Validation samples: {len(val_data)}")
            logger.info(f"  - Test samples: {len(test_data)}")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Preprocessing failed: {str(e)}")
            return False
    
    def step_2_fine_tune_model(self) -> bool:
        """Step 2: Fine-tune Model with LoRA"""
        logger.info("=== STEP 2: FINE-TUNING WITH LORA ===")
        
        try:
            fine_tuner = QwenBankingFineTuner(
                model_name=self.model_name,
                data_dir=self.processed_data_dir,
                output_dir=self.lora_output_dir
            )
            
            trainer = fine_tuner.run_fine_tuning(self.available_vram)
            
            if trainer:
                # Log training statistics
                final_train_loss = trainer.state.log_history[-1].get('train_loss', 'N/A')
                final_eval_loss = trainer.state.log_history[-1].get('eval_loss', 'N/A')
                
                logger.info(f"✓ Fine-tuning completed")
                logger.info(f"  - Final train loss: {final_train_loss}")
                logger.info(f"  - Final eval loss: {final_eval_loss}")
                
                return True
            else:
                logger.error("✗ Fine-tuning failed: No trainer returned")
                return False
                
        except Exception as e:
            logger.error(f"✗ Fine-tuning failed: {str(e)}")
            return False
    
    def step_3_merge_model(self) -> bool:
        """Step 3: Merge LoRA weights with base model"""
        logger.info("=== STEP 3: MERGING MODEL ===")
        
        try:
            merger = ModelMerger(
                base_model_name=self.model_name,
                lora_model_path=self.lora_output_dir,
                output_path=self.merged_output_dir
            )
            
            success = merger.merge_and_save()
            
            if success:
                logger.info(f"✓ Model merging completed")
                logger.info(f"  - Merged model saved to: {self.merged_output_dir}")
                return True
            else:
                logger.error("✗ Model merging failed")
                return False
                
        except Exception as e:
            logger.error(f"✗ Model merging failed: {str(e)}")
            return False
    
    def step_4_validate_pipeline(self) -> bool:
        """Step 4: Validate complete pipeline"""
        logger.info("=== STEP 4: PIPELINE VALIDATION ===")
        
        try:
            # Check if all required files exist
            required_files = [
                os.path.join(self.processed_data_dir, 'train.jsonl'),
                os.path.join(self.processed_data_dir, 'validation.jsonl'),
                os.path.join(self.processed_data_dir, 'test.jsonl'),
                os.path.join(self.lora_output_dir, 'adapter_config.json'),
                os.path.join(self.lora_output_dir, 'adapter_model.safetensors'),
                os.path.join(self.merged_output_dir, 'config.json'),
                os.path.join(self.merged_output_dir, 'model.safetensors.index.json'),
                os.path.join(self.merged_output_dir, 'tokenizer.json'),
                os.path.join(self.merged_output_dir, 'model_info.json'),
                os.path.join(self.merged_output_dir, 'inference.py')
            ]
            
            missing_files = []
            for file_path in required_files:
                if not os.path.exists(file_path):
                    missing_files.append(file_path)
            
            if missing_files:
                logger.error(f"✗ Validation failed - Missing files:")
                for file_path in missing_files:
                    logger.error(f"  - {file_path}")
                return False
            
            # Test model loading (quick test)
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    self.merged_output_dir,
                    trust_remote_code=True
                )
                logger.info("✓ Model loading test passed")
            except Exception as e:
                logger.error(f"✗ Model loading test failed: {str(e)}")
                return False
            
            logger.info("✓ Pipeline validation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"✗ Pipeline validation failed: {str(e)}")
            return False
    
    def run_complete_pipeline(self) -> bool:
        """Run complete fine-tuning pipeline"""
        logger.info("Starting complete fine-tuning pipeline...")
        
        steps = [
            ("Data Preprocessing", self.step_1_preprocess_data),
            ("Fine-tuning with LoRA", self.step_2_fine_tune_model),
            ("Model Merging", self.step_3_merge_model),
            ("Pipeline Validation", self.step_4_validate_pipeline)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"\n{'='*50}")
            logger.info(f"Starting: {step_name}")
            logger.info(f"{'='*50}")
            
            success = step_func()
            
            if not success:
                logger.error(f"Pipeline failed at step: {step_name}")
                return False
            
            logger.info(f"✓ {step_name} completed successfully")
        
        # Calculate total time
        end_time = datetime.now()
        total_time = end_time - self.start_time
        
        logger.info(f"\n{'='*50}")
        logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY! 🎉")
        logger.info(f"{'='*50}")
        logger.info(f"Start time: {self.start_time}")
        logger.info(f"End time: {end_time}")
        logger.info(f"Total duration: {total_time}")
        logger.info(f"Final model location: {self.merged_output_dir}")
        logger.info("Model is ready for RAG integration!")
        
        return True

def create_default_config() -> dict:
    """Create default configuration"""
    return {
        'csv_path': 'e:/HDBank_Hackathon/source/data/raw/csv/final_sua_mapped_v2.csv',
        'processed_data_dir': 'e:/HDBank_Hackathon/source/data/processed/train_split',
        'lora_output_dir': './qwen-banking-lora',
        'merged_output_dir': './qwen-banking-merged',
        'model_name': 'Qwen/Qwen2-4B'
    }

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='HDBank Chatbot Fine-tuning Pipeline')
    parser.add_argument('--csv-path', type=str, help='Path to CSV data file')
    parser.add_argument('--output-dir', type=str, default='./qwen-banking-merged', 
                       help='Output directory for merged model')
    parser.add_argument('--model-name', type=str, default='Qwen/Qwen2-4B',
                       help='Base model name')
    parser.add_argument('--skip-preprocessing', action='store_true',
                       help='Skip data preprocessing step')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training step')
    parser.add_argument('--skip-merging', action='store_true',
                       help='Skip model merging step')
    
    args = parser.parse_args()
    
    # Create configuration
    config = create_default_config()
    
    # Override with command line arguments
    if args.csv_path:
        config['csv_path'] = args.csv_path
    if args.output_dir:
        config['merged_output_dir'] = args.output_dir
    if args.model_name:
        config['model_name'] = args.model_name
    
    # Initialize and run pipeline
    pipeline = CompletePipeline(config)
    
    # Run selected steps
    if args.skip_preprocessing and args.skip_training and args.skip_merging:
        logger.info("All steps skipped, running validation only...")
        success = pipeline.step_4_validate_pipeline()
    else:
        success = pipeline.run_complete_pipeline()
    
    if success:
        logger.info("Pipeline completed successfully!")
        sys.exit(0)
    else:
        logger.error("Pipeline failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()