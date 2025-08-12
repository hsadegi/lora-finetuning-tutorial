# آموزش کامل LoRA Fine-tuning 🚀

**راهنمای جامع آموزش مدل‌های زبانی بزرگ با تکنیک LoRA**

## 📋 فهرست مطالب

- [مقدمه](#مقدمه)
- [نصب و راه‌اندازی](#نصب-و-راه‌اندازی)
- [درک LoRA](#درک-lora)
- [آموزش قدم به قدم](#آموزش-قدم-به-قدم)
- [مثال‌های عملی](#مثال‌های-عملی)
- [بهینه‌سازی و نکات](#بهینه‌سازی-و-نکات)

## 🎯 مقدمه

**LoRA (Low-Rank Adaptation)** یکی از موثرترین روش‌های آموزش مدل‌های زبانی بزرگ است که:

- ✅ **99% کاهش** پارامترهای قابل آموزش
- ✅ **سرعت بالا** در آموزش و استنتاج
- ✅ **استفاده بهینه** از حافظه GPU
- ✅ **عملکرد مشابه** Fine-tuning کامل

## 🔧 نصب و راه‌اندازی

### پیش‌نیازها
```bash
Python >= 3.8
CUDA >= 11.0 (برای GPU)
RAM >= 8GB
GPU Memory >= 4GB (توصیه شده)
```

### نصب پکیج‌ها
```bash
pip install -r requirements.txt
```

## 🧠 درک LoRA

### چگونه LoRA کار می‌کند؟

```python
# بجای آموزش کل ماتریس W (بزرگ و پرهزینه)
W_updated = W_original + ΔW

# LoRA ΔW را به دو ماتریس کوچک تجزیه می‌کند
ΔW = A × B
# جایی که A: [d, r] و B: [r, d] و r << d
```

### مزایای کلیدی:
- 🔹 **کاهش پارامترها**: از میلیون‌ها به هزارها
- 🔹 **سرعت آموزش**: 3-5 برابر سریع‌تر
- 🔹 **حافظه کمتر**: قابل اجرا روی GPU‌های کوچک
- 🔹 **مدولار**: قابل ترکیب و تعویض

## 🚀 آموزش قدم به قدم

### گام ۱: بارگذاری مدل پایه
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# بارگذاری مدل پایه
model_name = "microsoft/DialoGPT-medium"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

### گام ۲: تنظیم LoRA
```python
# پیکربندی LoRA
lora_config = LoraConfig(
    r=16,                    # رتبه تجزیه (کلید اصلی)
    lora_alpha=32,          # ضریب مقیاس‌بندی
    target_modules=["c_attn", "c_proj"],  # لایه‌های هدف
    lora_dropout=0.1,       # جلوگیری از overfitting
    bias="none",            # نوع bias
    task_type="CAUSAL_LM"   # نوع کار
)

# اعمال LoRA به مدل
model = get_peft_model(model, lora_config)
```

### گام ۳: آماده‌سازی داده
```python
# مثال ساده داده‌های گفتگو
conversations = [
    {"input": "سلام، حال شما چطور است؟", "output": "سلام! من خوبم، متشکرم. شما چطورید؟"},
    {"input": "امروز هوا چطور است؟", "output": "هوا آفتابی و دلپذیر است. روز خوبی برای پیاده‌روی!"}
]

# تبدیل به فرمت آموزش
def prepare_dataset(conversations):
    inputs, outputs = [], []
    for conv in conversations:
        inputs.append(conv["input"])
        outputs.append(conv["output"])
    return inputs, outputs
```

### گام ۴: آموزش مدل
```python
from transformers import Trainer, TrainingArguments

# تنظیمات آموزش
training_args = TrainingArguments(
    output_dir="./lora_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    warmup_steps=100,
    learning_rate=2e-4,
    fp16=True,                    # استفاده از precision کمتر
    logging_steps=10,
    save_strategy="epoch"
)

# شروع آموزش
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer
)

trainer.train()
```

## 📊 مثال‌های عملی

### ۱. آموزش مدل گفتگو فارسی
```python
# اجرای notebook آموزشی
jupyter notebook notebooks/persian_chat_lora.ipynb
```

### ۲. Fine-tuning برای خلاصه‌سازی
```python
# اجرای اسکریپت تخصصی
python src/summarization_lora.py
```

### ۳. آموزش مدل سوال-پاسخ
```python
# اجرای آموزش Q&A
python src/qa_lora.py --dataset_path data/qa_dataset.json
```

## ⚡ بهینه‌سازی و نکات

### نکات عملکردی:
```python
# ۱. انتخاب r مناسب
r = 8    # برای کارهای ساده
r = 16   # برای اکثر کاربردها (توصیه شده)
r = 32   # برای کارهای پیچیده

# ۲. بهینه‌سازی حافظه
torch.backends.cudnn.benchmark = True
model.gradient_checkpointing_enable()

# ۳. استفاده از QLoRA برای GPU کوچک
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
```

### عیب‌یابی رایج:
- 🔧 **CUDA out of memory**: کاهش batch_size یا استفاده از gradient_checkpointing
- 🔧 **آموزش آهسته**: افزایش batch_size یا gradient_accumulation_steps
- 🔧 **عملکرد ضعیف**: افزایش r یا تغییر target_modules

## 📈 ارزیابی عملکرد

```python
# ارزیابی کیفیت مدل
from src.evaluation import evaluate_model

results = evaluate_model(
    model=trained_model,
    test_dataset=test_data,
    metrics=["bleu", "rouge", "perplexity"]
)

print(f"BLEU Score: {results['bleu']:.3f}")
print(f"ROUGE Score: {results['rouge']:.3f}")
print(f"Perplexity: {results['perplexity']:.3f}")
```

## 🎯 استفاده از مدل آموزش‌یافته

```python
# بارگذاری مدل ذخیره شده
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
model = PeftModel.from_pretrained(base_model, "./lora_model")

# تولید پاسخ
def generate_response(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=100, do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# تست
response = generate_response("سلام، امروز چه خبر؟")
print(response)
```

## 📚 منابع اضافی

- 📖 [مقاله اصلی LoRA](https://arxiv.org/abs/2106.09685)
- 🔗 [مستندات PEFT](https://huggingface.co/docs/peft)
- 💡 [نمونه‌های بیشتر](examples/)
- 🎥 [ویدیوهای آموزشی](docs/videos.md)

## 🤝 مشارکت

برای مشارکت در این پروژه:
1. Fork کنید
2. Branch جدید بسازید
3. تغییرات را commit کنید
4. Pull Request ارسال کنید

## 📞 پشتیبانی

سوالات خود را در [Issues](../../issues) مطرح کنید یا با ما در ارتباط باشید.

---

**ساخته شده با ❤️ برای جامعه AI ایران**
