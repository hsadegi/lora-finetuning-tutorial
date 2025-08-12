"""
تنظیمات کلی پروژه LoRA Fine-tuning
Configuration settings for LoRA Fine-tuning project
"""

import torch
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class ModelConfig:
    """تنظیمات مدل پایه"""
    model_name: str = "microsoft/DialoGPT-medium"
    max_length: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

@dataclass
class LoRAConfig:
    """تنظیمات LoRA"""
    r: int = 16                          # رتبه تجزیه (کلیدی‌ترین پارامتر)
    lora_alpha: int = 32                 # ضریب مقیاس‌بندی
    lora_dropout: float = 0.1            # نرخ dropout
    bias: str = "none"                   # نوع bias: "none", "all", "lora_only"
    task_type: str = "CAUSAL_LM"         # نوع کار
    target_modules: List[str] = None     # لایه‌های هدف
    
    def __post_init__(self):
        if self.target_modules is None:
            # لایه‌های پیش‌فرض برای مدل‌های مختلف
            self.target_modules = ["c_attn", "c_proj"]  # برای GPT-2 و DialoGPT

@dataclass
class TrainingConfig:
    """تنظیمات آموزش"""
    output_dir: str = "./models/lora_output"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    
    # تنظیمات پیشرفته
    fp16: bool = True                    # استفاده از precision کمتر
    gradient_checkpointing: bool = True  # بهینه‌سازی حافظه
    dataloader_num_workers: int = 4
    
    # مانیتورینگ
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    save_strategy: str = "epoch"
    evaluation_strategy: str = "steps"
    
    # Weights & Biases
    use_wandb: bool = False
    wandb_project: str = "lora-finetuning"
    wandb_run_name: Optional[str] = None

@dataclass
class DataConfig:
    """تنظیمات داده"""
    train_file: str = "data/train.json"
    validation_file: str = "data/validation.json"
    test_file: str = "data/test.json"
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    
    # پیش‌پردازش
    padding: str = "max_length"
    truncation: bool = True
    return_overflowing_tokens: bool = False

@dataclass
class QLoRAConfig:
    """تنظیمات QLoRA برای بهینه‌سازی حافظه"""
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: torch.dtype = torch.float16
    bnb_4bit_use_double_quant: bool = True

# تنظیمات پیش‌فرض
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_LORA_CONFIG = LoRAConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_DATA_CONFIG = DataConfig()
DEFAULT_QLORA_CONFIG = QLoRAConfig()

# نگاشت مدل‌ها به target_modules مناسب
MODEL_TARGET_MODULES = {
    "gpt2": ["c_attn", "c_proj"],
    "microsoft/DialoGPT-medium": ["c_attn", "c_proj"],
    "microsoft/DialoGPT-large": ["c_attn", "c_proj"],
    "meta-llama/Llama-2-7b-hf": ["q_proj", "v_proj"],
    "meta-llama/Llama-2-13b-hf": ["q_proj", "v_proj"],
    "mistralai/Mistral-7B-v0.1": ["q_proj", "v_proj"],
    "google/flan-t5-base": ["q", "v"],
    "google/flan-t5-large": ["q", "v"],
}

def get_target_modules(model_name: str) -> List[str]:
    """
    تشخیص target_modules مناسب برای مدل
    
    Args:
        model_name: نام مدل
        
    Returns:
        لیست target_modules مناسب
    """
    for key, modules in MODEL_TARGET_MODULES.items():
        if key in model_name.lower():
            return modules
    
    # پیش‌فرض برای مدل‌های ناشناخته
    print(f"⚠️  Target modules برای {model_name} شناخته نشده. از modules پیش‌فرض استفاده می‌شود.")
    return ["q_proj", "v_proj"]

# پیکربندی logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lora_training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
