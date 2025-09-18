# HDBank Chatbot Fine-tuning với Qwen3-4B

Hệ thống fine-tuning hoàn chỉnh cho chatbot tư vấn tài chính ngân hàng sử dụng Qwen3-4B với LoRA.

## 🎯 Tổng quan

Dự án này fine-tune mô hình Qwen3-4B để tạo ra một chatbot chuyên biệt cho tư vấn tài chính ngân hàng. Sử dụng kỹ thuật LoRA (Low-Rank Adaptation) để tối ưu VRAM và chất lượng training.

## 📋 Yêu cầu hệ thống

### Phần cứng tối thiểu:
- **GPU**: RTX 3080 (10GB VRAM) hoặc tương đương
- **RAM**: 16GB
- **Storage**: 50GB trống

### Phần cứng khuyến nghị:
- **GPU**: RTX 4090 (24GB VRAM) hoặc A100
- **RAM**: 32GB+
- **Storage**: 100GB+ SSD

## 🚀 Cài đặt

### 1. Clone repository và cài đặt dependencies

```bash
cd e:/HDBank_Hackathon/source
pip install -r requirements_finetune.txt
```

### 2. Chuẩn bị dữ liệu

Đảm bảo file CSV `final_sua_mapped_v2.csv` có cấu trúc:
```
tags,instruction,category,intent,response
BCIPZ,"Tôi muốn kích hoạt thẻ...",CARD,activate_card,"Tôi ở đây để giúp..."
```

## 📊 Cấu trúc dữ liệu

### Format đầu vào (CSV):
- `instruction`: Câu hỏi của khách hàng
- `response`: Câu trả lời của chatbot (có thể chứa placeholder)

### Format sau xử lý (ChatML):
```json
{
  "messages": [
    {"role": "system", "content": "System prompt..."},
    {"role": "user", "content": "Câu hỏi khách hàng"},
    {"role": "assistant", "content": "Câu trả lời chatbot"}
  ]
}
```

## ⚙️ Cấu hình

### LoRA Configuration
```python
# Cấu hình chuẩn (12GB VRAM)
r=16, lora_alpha=32, dropout=0.1
target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Cấu hình tiết kiệm memory (8GB VRAM)  
r=8, lora_alpha=16, dropout=0.1
target_modules=["q_proj", "v_proj", "down_proj"]

# Cấu hình chất lượng cao (16GB+ VRAM)
r=32, lora_alpha=64, dropout=0.05
```

### Training Configuration
```python
# Tham số training được tối ưu
learning_rate=2e-4
batch_size=4 (per device)
gradient_accumulation=4 (effective batch size = 16)
epochs=3
warmup_steps=100
```

## 🔧 Sử dụng

### Chạy pipeline hoàn chỉnh:

```bash
cd scripts
python run_complete_pipeline.py
```

### Chạy từng bước riêng lẻ:

#### 1. Tiền xử lý dữ liệu:
```bash
python data_preprocessing.py
```

#### 2. Fine-tuning với LoRA:
```bash
python fine_tune_qwen.py
```

#### 3. Merge model:
```bash
python merge_model.py
```

### Tùy chọn command line:

```bash
# Chỉ định file CSV khác
python run_complete_pipeline.py --csv-path /path/to/your/data.csv

# Chỉ định thư mục output
python run_complete_pipeline.py --output-dir ./my-custom-model

# Sử dụng model khác
python run_complete_pipeline.py --model-name Qwen/Qwen3-7B

# Bỏ qua bước preprocessing
python run_complete_pipeline.py --skip-preprocessing
```

## 📈 Monitoring và Logging

### Logs được lưu tại:
- `fine_tuning_pipeline.log`: Log tổng quát
- Console output: Real-time progress

### Thông tin được track:
- Training/validation loss
- Memory usage
- Training time
- Model parameters count
- Dataset statistics

## 🎛️ Tối ưu cho VRAM khác nhau

### 8GB VRAM (RTX 3070/4060 Ti):
```python
# Tự động chọn cấu hình memory-efficient
batch_size=2, gradient_accumulation=8
r=8, target_modules=3
```

### 12GB VRAM (RTX 3080 Ti/4070 Ti):
```python
# Cấu hình chuẩn
batch_size=4, gradient_accumulation=4  
r=16, target_modules=7
```

### 16GB+ VRAM (RTX 4080/4090):
```python
# Cấu hình chất lượng cao
batch_size=8, gradient_accumulation=2
r=32, target_modules=9
```

## 📁 Cấu trúc output

Sau khi hoàn thành, bạn sẽ có:

```
qwen-banking-merged/
├── config.json                 # Model config
├── model.safetensors.index.json # Model weights index
├── model-00001-of-00002.safetensors # Model weights
├── model-00002-of-00002.safetensors
├── tokenizer.json              # Tokenizer
├── tokenizer_config.json       # Tokenizer config
├── model_info.json            # Model metadata
└── inference.py               # Inference script
```

## 🧪 Testing và Validation

### Test model sau khi merge:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("./qwen-banking-merged")
model = AutoModelForCausalLM.from_pretrained("./qwen-banking-merged")

# Test inference
messages = [
    {"role": "system", "content": "Bạn là trợ lý tư vấn tài chính..."},
    {"role": "user", "content": "Tôi muốn kích hoạt thẻ tín dụng"}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=512)
response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(response)
```

## 🔗 Tích hợp RAG

Model sau khi merge sẵn sàng tích hợp với hệ thống RAG:

```python
# Sử dụng merged model trong RAG pipeline
from your_rag_system import RAGPipeline

rag = RAGPipeline(
    model_path="./qwen-banking-merged",
    vector_store_path="./vector_db/",
    embedding_model="your-embedding-model"
)
```

## 🐛 Troubleshooting

### Lỗi CUDA Out of Memory:
```bash
# Giảm batch size
export CUDA_VISIBLE_DEVICES=0
python run_complete_pipeline.py  # Sẽ tự động chọn config phù hợp
```

### Lỗi Flash Attention:
```bash
# Nếu không hỗ trợ flash attention
pip uninstall flash-attn
# Hoặc set attn_implementation=None trong code
```

### Lỗi Model Loading:
```bash
# Kiểm tra HuggingFace token
huggingface-cli login
```

## 📊 Kết quả mong đợi

### Training metrics:
- **Training loss**: ~1.5-2.0 (cuối training)
- **Validation loss**: ~1.8-2.5 (tùy dataset)
- **Training time**: 2-4 giờ (RTX 4090)

### Model performance:
- **Response quality**: Cải thiện đáng kể so với base model
- **Domain knowledge**: Chuyên biệt hóa cho banking
- **Placeholder handling**: Xử lý đúng các placeholder

## 🤝 Đóng góp

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push và tạo Pull Request

## 📄 License

MIT License - xem file LICENSE để biết thêm chi tiết.

## 📞 Hỗ trợ

Nếu gặp vấn đề, vui lòng:
1. Kiểm tra logs trong `fine_tuning_pipeline.log`
2. Tạo issue trên GitHub với thông tin chi tiết
3. Liên hệ team phát triển

---

**Lưu ý**: Đây là hệ thống fine-tuning được tối ưu cho dự án HDBank Hackathon. Các tham số đã được nghiên cứu và test kỹ lưỡng để đạt chất lượng tốt nhất với tài nguyên có hạn.