"""
Training Configuration for Qwen2.5-4B Banking Chatbot Fine-tuning
Cấu hình training được tối ưu cho chất lượng và hiệu suất
"""

from transformers import TrainingArguments
from dataclasses import dataclass
from typing import Optional
import torch
import os

@dataclass
class TrainingConfigManager:
    """
    Training Configuration Manager với các tham số được nghiên cứu kỹ lưỡng
    
    Phân tích tham số:
    - learning_rate=2e-4: Optimal cho LoRA fine-tuning, không quá cao gây instability
    - batch_size=4: Cân bằng giữa memory usage và gradient stability
    - gradient_accumulation=4: Effective batch size = 16, đủ lớn cho stable training
    - warmup_steps=100: Warm-up để tránh gradient explosion ở đầu training
    - max_steps: Được tính dựa trên dataset size và epochs
    """
    
    # Core training parameters
    output_dir: str = "./qwen-banking-lora"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    
    # Learning rate and optimization
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    lr_scheduler_type: str = "cosine"
    
    # Mixed precision and optimization
    fp16: bool = True
    bf16: bool = False  # Set to True if using Ampere+ GPUs
    gradient_checkpointing: bool = True
    dataloader_pin_memory: bool = False
    
    # Evaluation and logging
    evaluation_strategy: str = "steps"
    eval_steps: int = 100
    logging_steps: int = 10
    save_strategy: str = "steps"
    save_steps: int = 200
    save_total_limit: int = 3
    
    # Early stopping and best model
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Memory optimization
    remove_unused_columns: bool = False
    dataloader_num_workers: int = 0
    
    def __post_init__(self):
        """Post-initialization setup"""
        # Auto-detect bf16 support
        if torch.cuda.is_available():
            device_capability = torch.cuda.get_device_capability()
            if device_capability[0] >= 8:  # Ampere or newer
                self.bf16 = True
                self.fp16 = False
                print("Using BF16 precision (Ampere+ GPU detected)")
            else:
                print("Using FP16 precision")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def get_training_args(self, max_steps: Optional[int] = None) -> TrainingArguments:
        """Create TrainingArguments object"""
        
        args_dict = {
            "output_dir": self.output_dir,
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_steps": self.warmup_steps,
            "lr_scheduler_type": self.lr_scheduler_type,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "dataloader_pin_memory": self.dataloader_pin_memory,
            "evaluation_strategy": self.evaluation_strategy,
            "eval_steps": self.eval_steps,
            "logging_steps": self.logging_steps,
            "save_strategy": self.save_strategy,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "load_best_model_at_end": self.load_best_model_at_end,
            "metric_for_best_model": self.metric_for_best_model,
            "greater_is_better": self.greater_is_better,
            "remove_unused_columns": self.remove_unused_columns,
            "dataloader_num_workers": self.dataloader_num_workers,
            "report_to": "none",  # Disable wandb/tensorboard by default
            "seed": 42,
        }
        
        if max_steps is not None:
            args_dict["max_steps"] = max_steps
            args_dict.pop("num_train_epochs")  # Remove epochs if using max_steps
        
        return TrainingArguments(**args_dict)
    
    def get_memory_efficient_args(self, max_steps: Optional[int] = None) -> TrainingArguments:
        """Get memory-efficient training arguments"""
        # Reduce batch sizes for memory efficiency
        self.per_device_train_batch_size = 2
        self.per_device_eval_batch_size = 2
        self.gradient_accumulation_steps = 8  # Keep effective batch size = 16
        self.gradient_checkpointing = True
        self.dataloader_pin_memory = False
        
        return self.get_training_args(max_steps)
    
    def get_high_performance_args(self, max_steps: Optional[int] = None) -> TrainingArguments:
        """Get high-performance training arguments"""
        # Increase batch sizes for better performance
        self.per_device_train_batch_size = 8
        self.per_device_eval_batch_size = 8
        self.gradient_accumulation_steps = 2  # Effective batch size = 16
        self.dataloader_num_workers = 4
        self.dataloader_pin_memory = True
        
        return self.get_training_args(max_steps)
    
    def calculate_training_steps(self, dataset_size: int) -> dict:
        """Calculate training steps and duration"""
        effective_batch_size = (
            self.per_device_train_batch_size * 
            self.gradient_accumulation_steps
        )
        
        steps_per_epoch = dataset_size // effective_batch_size
        total_steps = steps_per_epoch * self.num_train_epochs
        
        # Estimate training time (rough approximation)
        # Assuming ~1.5 seconds per step on RTX 4090
        estimated_time_hours = (total_steps * 1.5) / 3600
        
        return {
            "dataset_size": dataset_size,
            "effective_batch_size": effective_batch_size,
            "steps_per_epoch": steps_per_epoch,
            "total_training_steps": total_steps,
            "estimated_time_hours": estimated_time_hours,
            "warmup_ratio": self.warmup_steps / total_steps if total_steps > 0 else 0
        }
    
    def print_config_summary(self, dataset_size: Optional[int] = None):
        """Print training configuration summary"""
        print("=== Training Configuration Summary ===")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Batch size per device: {self.per_device_train_batch_size}")
        print(f"Gradient accumulation: {self.gradient_accumulation_steps}")
        print(f"Effective batch size: {self.per_device_train_batch_size * self.gradient_accumulation_steps}")
        print(f"Epochs: {self.num_train_epochs}")
        print(f"Warmup steps: {self.warmup_steps}")
        print(f"Weight decay: {self.weight_decay}")
        print(f"Precision: {'BF16' if self.bf16 else 'FP16' if self.fp16 else 'FP32'}")
        print(f"Gradient checkpointing: {self.gradient_checkpointing}")
        
        if dataset_size:
            training_info = self.calculate_training_steps(dataset_size)
            print(f"\n=== Training Schedule ===")
            print(f"Dataset size: {training_info['dataset_size']}")
            print(f"Steps per epoch: {training_info['steps_per_epoch']}")
            print(f"Total training steps: {training_info['total_training_steps']}")
            print(f"Estimated time: {training_info['estimated_time_hours']:.1f} hours")
            print(f"Warmup ratio: {training_info['warmup_ratio']:.3f}")

def get_recommended_training_config(available_vram_gb: float, dataset_size: int) -> TrainingArguments:
    """Get recommended training config based on available VRAM"""
    config_manager = TrainingConfigManager()
    
    # Calculate max steps for better control
    training_info = config_manager.calculate_training_steps(dataset_size)
    max_steps = training_info['total_training_steps']
    
    if available_vram_gb >= 16:
        print("Using high-performance training configuration")
        return config_manager.get_high_performance_args(max_steps)
    elif available_vram_gb >= 12:
        print("Using standard training configuration")
        return config_manager.get_training_args(max_steps)
    else:
        print("Using memory-efficient training configuration")
        return config_manager.get_memory_efficient_args(max_steps)

if __name__ == "__main__":
    # Test configurations
    config_manager = TrainingConfigManager()
    config_manager.print_config_summary(dataset_size=1000)
    
    # Test different VRAM scenarios
    print("\n=== Testing different VRAM scenarios ===")
    for vram in [8, 12, 16, 24]:
        print(f"\nVRAM: {vram}GB")
        args = get_recommended_training_config(vram, 1000)
        print(f"Batch size: {args.per_device_train_batch_size}")
        print(f"Gradient accumulation: {args.gradient_accumulation_steps}")