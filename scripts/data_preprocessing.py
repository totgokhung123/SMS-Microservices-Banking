"""
Data Preprocessing Script for Qwen2-4B Banking Chatbot Fine-tuning
Xử lý dữ liệu CSV thành format ChatML phù hợp với Qwen2-4B
"""

import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
from typing import List, Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BankingDataPreprocessor:
    def __init__(self, csv_path: str, output_dir: str):
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.system_prompt = """Bạn là trợ lý tư vấn tài chính ngân hàng chuyên nghiệp của HDBank. Bạn có kiến thức sâu về các sản phẩm và dịch vụ ngân hàng, luôn hỗ trợ khách hàng một cách tận tình và chính xác. Hãy trả lời các câu hỏi một cách chi tiết, dễ hiểu và thân thiện."""
        
    def load_data(self) -> pd.DataFrame:
        """Load CSV data"""
        logger.info(f"Loading data from {self.csv_path}")
        df = pd.read_csv(self.csv_path, encoding='utf-8')
        logger.info(f"Loaded {len(df)} records")
        return df
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if pd.isna(text):
            return ""
        
        # Remove extra whitespaces and normalize
        text = str(text).strip()
        text = ' '.join(text.split())
        
        # Remove quotes that might interfere with JSON
        text = text.replace('""', '"')
        
        return text
    
    def create_chat_format(self, instruction: str, response: str) -> Dict[str, Any]:
        """Convert instruction-response pair to ChatML format"""
        return {
            "messages": [
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user", 
                    "content": self.clean_text(instruction)
                },
                {
                    "role": "assistant",
                    "content": self.clean_text(response)
                }
            ]
        }
    
    def process_data(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Process DataFrame to ChatML format"""
        processed_data = []
        
        for idx, row in df.iterrows():
            if pd.isna(row['instruction']) or pd.isna(row['response']):
                logger.warning(f"Skipping row {idx} due to missing data")
                continue
                
            chat_data = self.create_chat_format(
                instruction=row['instruction'],
                response=row['response']
            )
            processed_data.append(chat_data)
            
        logger.info(f"Processed {len(processed_data)} valid records")
        return processed_data
    
    def split_data(self, data: List[Dict[str, Any]], 
                   train_ratio: float = 0.8, 
                   val_ratio: float = 0.1,
                   test_ratio: float = 0.1) -> tuple:
        """Split data into train/validation/test sets"""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        # First split: train + val vs test
        train_val, test = train_test_split(
            data, 
            test_size=test_ratio, 
            random_state=42,
            shuffle=True
        )
        
        # Second split: train vs val
        val_size = val_ratio / (train_ratio + val_ratio)
        train, val = train_test_split(
            train_val,
            test_size=val_size,
            random_state=42,
            shuffle=True
        )
        
        logger.info(f"Data split - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        return train, val, test
    
    def save_jsonl(self, data: List[Dict[str, Any]], filename: str):
        """Save data in JSONL format"""
        filepath = os.path.join(self.output_dir, filename)
        os.makedirs(self.output_dir, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(data)} records to {filepath}")
    
    def generate_statistics(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate dataset statistics"""
        total_samples = len(data)
        
        # Calculate text lengths
        instruction_lengths = []
        response_lengths = []
        
        for item in data:
            messages = item['messages']
            user_msg = next(msg for msg in messages if msg['role'] == 'user')
            assistant_msg = next(msg for msg in messages if msg['role'] == 'assistant')
            
            instruction_lengths.append(len(user_msg['content']))
            response_lengths.append(len(assistant_msg['content']))
        
        stats = {
            'total_samples': total_samples,
            'avg_instruction_length': sum(instruction_lengths) / len(instruction_lengths),
            'avg_response_length': sum(response_lengths) / len(response_lengths),
            'max_instruction_length': max(instruction_lengths),
            'max_response_length': max(response_lengths),
            'min_instruction_length': min(instruction_lengths),
            'min_response_length': min(response_lengths)
        }
        
        return stats
    
    def run_preprocessing(self):
        """Main preprocessing pipeline"""
        logger.info("Starting data preprocessing...")
        
        # Load and process data
        df = self.load_data()
        processed_data = self.process_data(df)
        
        # Split data
        train_data, val_data, test_data = self.split_data(processed_data)
        
        # Save datasets
        self.save_jsonl(train_data, 'train.jsonl')
        self.save_jsonl(val_data, 'validation.jsonl')
        self.save_jsonl(test_data, 'test.jsonl')
        
        # Generate and save statistics
        stats = self.generate_statistics(processed_data)
        stats_path = os.path.join(self.output_dir, 'dataset_stats.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info("Preprocessing completed successfully!")
        logger.info(f"Dataset statistics: {stats}")
        
        return train_data, val_data, test_data

if __name__ == "__main__":
    # Configuration
    CSV_PATH = "e:/HDBank_Hackathon/source/data/raw/csv/final_sua_mapped_v2.csv"
    OUTPUT_DIR = "e:/HDBank_Hackathon/source/data/processed/train_split"
    
    # Run preprocessing
    preprocessor = BankingDataPreprocessor(CSV_PATH, OUTPUT_DIR)
    train_data, val_data, test_data = preprocessor.run_preprocessing()