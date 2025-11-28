import torch
import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_from_disk

def main():
    # ==========================================
    # 配置
    # ==========================================
    MODEL_PATH = "models/sft_1/checkpoint/checkpoint-249840"
    
    # 数据集路径
    DATA_PATH = "data/tldr_cleaned"
    
    # 测试样本数
    NUM_SAMPLES = 2
    # ==========================================

    print(f"Loading model from: {MODEL_PATH} ...")
    
    # 1. 加载模型和分词器
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)
        model.eval() # 切换到评估模式
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. 加载数据集 (从验证集或测试集取，看看泛化能力)
    print(f"Loading dataset from: {DATA_PATH} ...")
    try:
        # 尝试从测试集取数据，如果没有就用训练集
        dataset = load_from_disk(DATA_PATH)
        if "test" in dataset:
            data_split = dataset["test"]
            print("Using 'test' split.")
        else:
            data_split = dataset["train"]
            print("Using 'train' split.")
    except:
        print("Dataset loading failed.")
        return

    # 3. 随机抽取样本
    # 过滤掉太短的文本，保证测试效果
    valid_samples = [x for x in data_split if len(x["post_for_model"]) > 200]
    samples = random.sample(valid_samples, NUM_SAMPLES)

    print("\n" + "="*50)
    print("       SFT MODEL INFERENCE TEST       ")
    print("="*50)

    # 4. 推理生成
    for i, item in enumerate(samples):
        input_text = item["post_for_model"]
        gold_summary = item["gold_tldr"]
        
        # 编码
        inputs = tokenizer(
            input_text, 
            return_tensors="pt", 
            max_length=1024, 
            truncation=True
        ).to(device)

        # 生成 (使用 Beam Search 以获得较高质量)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                num_beams=5,
                no_repeat_ngram_size=3,
                early_stopping=True
            )
        
        # 解码
        generated_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 5. 打印结果
        print(f"\n[Sample {i+1}]")
        print("-" * 20)
        # 打印部分原文（太长就截断）
        print(f"Input Post (Truncated):\n{input_text[:300]}...") 
        print("-" * 20)
        print(f"Human Reference:\n{gold_summary}")
        print("-" * 20)
        print(f"Model Prediction:\n{generated_summary}")
        print("="*50)

if __name__ == "__main__":
    main()