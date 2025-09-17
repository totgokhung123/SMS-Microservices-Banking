"""
Model Merging Script for Qwen2-4B Banking Chatbot
Merge LoRA weights với base model và chuẩn bị cho RAG integration
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import logging
from typing import Optional
import json
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelMerger:
    def __init__(self, 
                 base_model_name: str = "Qwen/Qwen2-4B",
                 lora_model_path: str = "./qwen-banking-lora",
                 output_path: str = "./qwen-banking-merged"):
        
        self.base_model_name = base_model_name
        self.lora_model_path = lora_model_path
        self.output_path = output_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Initializing model merger")
        logger.info(f"Base model: {base_model_name}")
        logger.info(f"LoRA model: {lora_model_path}")
        logger.info(f"Output path: {output_path}")
        logger.info(f"Device: {self.device}")
    
    def load_base_model(self):
        """Load base model and tokenizer"""
        logger.info("Loading base model and tokenizer...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True
        )
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        logger.info("Base model and tokenizer loaded successfully")
        return base_model, tokenizer
    
    def merge_lora_weights(self, base_model, tokenizer):
        """Merge LoRA weights with base model"""
        logger.info("Loading LoRA model and merging weights...")
        
        # Load LoRA model
        lora_model = PeftModel.from_pretrained(
            base_model,
            self.lora_model_path,
            torch_dtype=torch.float16
        )
        
        logger.info("LoRA model loaded, starting merge...")
        
        # Merge weights
        merged_model = lora_model.merge_and_unload()
        
        logger.info("LoRA weights merged successfully")
        return merged_model, tokenizer
    
    def save_merged_model(self, merged_model, tokenizer):
        """Save merged model and tokenizer"""
        logger.info(f"Saving merged model to {self.output_path}...")
        
        # Create output directory
        os.makedirs(self.output_path, exist_ok=True)
        
        # Save model with safe serialization
        merged_model.save_pretrained(
            self.output_path,
            safe_serialization=True,
            max_shard_size="2GB"
        )
        
        # Save tokenizer
        tokenizer.save_pretrained(self.output_path)
        
        logger.info("Merged model saved successfully")
    
    def create_model_info(self, merged_model, tokenizer):
        """Create model information file"""
        logger.info("Creating model information file...")
        
        # Get model info
        model_info = {
            "base_model": self.base_model_name,
            "lora_model": self.lora_model_path,
            "merged_at": torch.datetime.now().isoformat(),
            "model_type": "qwen2-banking-chatbot",
            "task": "conversational-banking-assistant",
            "language": "vietnamese",
            "parameters": {
                "vocab_size": tokenizer.vocab_size,
                "model_size": "4B",
                "precision": "float16"
            },
            "usage": {
                "max_length": 1024,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True
            },
            "placeholders": {
                "description": "Model uses placeholders for sensitive information",
                "examples": [
                    "{{CARD_PIN_FIELD}}",
                    "{{ACCOUNT_LOGIN_URL}}",
                    "{{SUPPORT_PHONE}}",
                    "{{RESET_PASSWORD_URL}}",
                    "{{BILL_PAY_URL}}",
                    "{{SUPPORT_WEBSITE}}",
                    "{{WORKING_HOURS}}",
                    "{{CREDIT_LIMIT_INFO}}",
                    "{{MOBILE_APP_APPSTORE_URL}}",
                    "{{MOBILE_APP_PLAYSTORE_URL}}"
                ]
            }
        }
        
        # Save model info
        info_path = os.path.join(self.output_path, "model_info.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Model information saved to {info_path}")
    
    def create_inference_script(self):
        """Create inference script template"""
        logger.info("Creating inference script template...")
        
        inference_script = '''"""
Inference Script for Qwen2-4B Banking Chatbot
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class BankingChatbot:
    def __init__(self, model_path="./qwen-banking-merged"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.system_prompt = """Bạn là trợ lý tư vấn tài chính ngân hàng chuyên nghiệp của HDBank. Bạn có kiến thức sâu về các sản phẩm và dịch vụ ngân hàng, luôn hỗ trợ khách hàng một cách tận tình và chính xác. Hãy trả lời các câu hỏi một cách chi tiết, dễ hiểu và thân thiện."""
    
    def chat(self, user_input: str, max_length: int = 1024) -> str:
        """Generate response for user input"""
        
        # Create messages
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()

if __name__ == "__main__":
    # Example usage
    chatbot = BankingChatbot()
    
    while True:
        user_input = input("Bạn: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            break
        
        response = chatbot.chat(user_input)
        print(f"HDBank Assistant: {response}")
'''
        
        script_path = os.path.join(self.output_path, "inference.py")
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(inference_script)
        
        logger.info(f"Inference script saved to {script_path}")
    
    def validate_merged_model(self, merged_model, tokenizer):
        """Validate merged model functionality"""
        logger.info("Validating merged model...")
        
        try:
            # Test tokenization
            test_text = "Tôi muốn kích hoạt thẻ tín dụng"
            tokens = tokenizer(test_text, return_tensors="pt")
            
            # Test model forward pass
            with torch.no_grad():
                outputs = merged_model(**tokens)
            
            logger.info("Model validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Model validation failed: {str(e)}")
            return False
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        logger.info("Cleaning up temporary files...")
        
        # List of temporary directories/files to clean
        temp_paths = [
            os.path.join(self.output_path, "__pycache__"),
            os.path.join(self.output_path, ".git"),
        ]
        
        for temp_path in temp_paths:
            if os.path.exists(temp_path):
                if os.path.isdir(temp_path):
                    shutil.rmtree(temp_path)
                else:
                    os.remove(temp_path)
        
        logger.info("Cleanup completed")
    
    def merge_and_save(self):
        """Complete merge and save pipeline"""
        try:
            # Load base model
            base_model, tokenizer = self.load_base_model()
            
            # Merge LoRA weights
            merged_model, tokenizer = self.merge_lora_weights(base_model, tokenizer)
            
            # Validate merged model
            if not self.validate_merged_model(merged_model, tokenizer):
                raise RuntimeError("Model validation failed")
            
            # Save merged model
            self.save_merged_model(merged_model, tokenizer)
            
            # Create additional files
            self.create_model_info(merged_model, tokenizer)
            self.create_inference_script()
            
            # Cleanup
            self.cleanup_temp_files()
            
            logger.info("Model merging completed successfully!")
            logger.info(f"Merged model ready at: {self.output_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error during model merging: {str(e)}")
            return False

def main():
    """Main function"""
    # Configuration
    BASE_MODEL = "Qwen/Qwen2-4B"
    LORA_MODEL_PATH = "./qwen-banking-lora"
    OUTPUT_PATH = "./qwen-banking-merged"
    
    # Initialize merger
    merger = ModelMerger(
        base_model_name=BASE_MODEL,
        lora_model_path=LORA_MODEL_PATH,
        output_path=OUTPUT_PATH
    )
    
    # Run merging
    success = merger.merge_and_save()
    
    if success:
        logger.info("=== Merge Summary ===")
        logger.info(f"✓ Base model: {BASE_MODEL}")
        logger.info(f"✓ LoRA model: {LORA_MODEL_PATH}")
        logger.info(f"✓ Output: {OUTPUT_PATH}")
        logger.info("✓ Model ready for RAG integration")
    else:
        logger.error("Model merging failed!")

if __name__ == "__main__":
    main()