"""
LoRA Configuration for Qwen2-4B Banking Chatbot Fine-tuning
Cấu hình LoRA được tối ưu cho VRAM và chất lượng training
"""

from peft import LoraConfig, TaskType
from dataclasses import dataclass
from typing import List, Optional
import torch

@dataclass
class LoRAConfigManager:
    """
    LoRA Configuration Manager với các tham số được tối ưu cho Qwen2-4B
    
    Phân tích tham số:
    - r=16: Rank thấp vừa đủ để capture patterns quan trọng, tiết kiệm VRAM
    - lora_alpha=32: Scaling factor = 2*r, cân bằng tốt giữa stability và learning capacity
    - target_modules: Tập trung vào attention và MLP layers quan trọng nhất
    - lora_dropout=0.1: Regularization vừa phải, tránh overfitting
    """
    
    # Core LoRA parameters
    r: int = 16                    # Rank - cân bằng hiệu suất/chất lượng
    lora_alpha: int = 32           # Scaling factor (2*r)
    lora_dropout: float = 0.1      # Dropout cho LoRA layers
    bias: str = "none"             # Không train bias để tiết kiệm memory
    
    # Target modules cho Qwen2-4B architecture
    target_modules: List[str] = None
    
    # Task configuration
    task_type: TaskType = TaskType.CAUSAL_LM
    
    def __post_init__(self):
        """Initialize target modules if not provided"""
        if self.target_modules is None:
            self.target_modules = [
                # Attention layers - quan trọng nhất cho language understanding
                "q_proj",      # Query projection
                "k_proj",      # Key projection  
                "v_proj",      # Value projection
                "o_proj",      # Output projection
                
                # MLP layers - quan trọng cho knowledge representation
                "gate_proj",   # Gate projection in MLP
                "up_proj",     # Up projection in MLP
                "down_proj",   # Down projection in MLP
            ]
    
    def get_lora_config(self) -> LoraConfig:
        """Create LoRA configuration"""
        return LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            bias=self.bias,
            task_type=self.task_type,
            inference_mode=False,
        )
    
    def get_memory_efficient_config(self) -> LoraConfig:
        """Get memory-efficient LoRA config for limited VRAM"""
        return LoraConfig(
            r=8,                    # Giảm rank để tiết kiệm memory
            lora_alpha=16,          # Tương ứng giảm alpha
            target_modules=[
                "q_proj", "v_proj",  # Chỉ train query và value
                "down_proj"          # Và output MLP
            ],
            lora_dropout=0.1,
            bias="none",
            task_type=self.task_type,
            inference_mode=False,
        )
    
    def get_high_quality_config(self) -> LoraConfig:
        """Get high-quality LoRA config for better performance"""
        return LoraConfig(
            r=32,                   # Rank cao hơn cho chất lượng tốt hơn
            lora_alpha=64,          # Alpha tương ứng
            target_modules=self.target_modules + [
                "embed_tokens",     # Thêm embedding layer
                "lm_head"          # Thêm language model head
            ],
            lora_dropout=0.05,      # Dropout thấp hơn
            bias="none",
            task_type=self.task_type,
            inference_mode=False,
        )
    
    def estimate_memory_usage(self, model_size_gb: float = 8.0) -> dict:
        """Estimate memory usage for different configurations"""
        
        # Base model memory
        base_memory = model_size_gb
        
        # LoRA parameters estimation
        # Rough calculation: r * (input_dim + output_dim) * num_layers * num_target_modules
        qwen_hidden_size = 3584  # Qwen2-4B hidden size
        num_layers = 40          # Qwen2-4B layers
        num_target_modules = len(self.target_modules)
        
        # Standard config
        standard_params = self.r * qwen_hidden_size * num_layers * num_target_modules
        standard_memory = standard_params * 4 / (1024**3)  # 4 bytes per param, convert to GB
        
        # Memory efficient config  
        efficient_params = 8 * qwen_hidden_size * num_layers * 3  # r=8, 3 modules
        efficient_memory = efficient_params * 4 / (1024**3)
        
        # High quality config
        quality_params = 32 * qwen_hidden_size * num_layers * (num_target_modules + 2)
        quality_memory = quality_params * 4 / (1024**3)
        
        return {
            "base_model_memory_gb": base_memory,
            "standard_lora_memory_gb": standard_memory,
            "efficient_lora_memory_gb": efficient_memory,
            "quality_lora_memory_gb": quality_memory,
            "total_standard_gb": base_memory + standard_memory,
            "total_efficient_gb": base_memory + efficient_memory,
            "total_quality_gb": base_memory + quality_memory
        }
    
    def print_config_summary(self):
        """Print configuration summary"""
        print("=== LoRA Configuration Summary ===")
        print(f"Rank (r): {self.r}")
        print(f"Alpha: {self.lora_alpha}")
        print(f"Dropout: {self.lora_dropout}")
        print(f"Target modules: {len(self.target_modules)}")
        print(f"Modules: {', '.join(self.target_modules)}")
        
        memory_info = self.estimate_memory_usage()
        print(f"\n=== Memory Estimation ===")
        print(f"Standard config total: {memory_info['total_standard_gb']:.2f} GB")
        print(f"Efficient config total: {memory_info['total_efficient_gb']:.2f} GB") 
        print(f"Quality config total: {memory_info['total_quality_gb']:.2f} GB")

def get_recommended_config(available_vram_gb: float) -> LoraConfig:
    """Get recommended LoRA config based on available VRAM"""
    config_manager = LoRAConfigManager()
    
    if available_vram_gb >= 16:
        print("Using high-quality LoRA configuration")
        return config_manager.get_high_quality_config()
    elif available_vram_gb >= 12:
        print("Using standard LoRA configuration")
        return config_manager.get_lora_config()
    else:
        print("Using memory-efficient LoRA configuration")
        return config_manager.get_memory_efficient_config()

if __name__ == "__main__":
    # Test configurations
    config_manager = LoRAConfigManager()
    config_manager.print_config_summary()
    
    # Test memory estimation
    print("\n=== Testing different VRAM scenarios ===")
    for vram in [8, 12, 16, 24]:
        print(f"\nVRAM: {vram}GB")
        config = get_recommended_config(vram)
        print(f"Selected rank: {config.r}, alpha: {config.lora_alpha}")