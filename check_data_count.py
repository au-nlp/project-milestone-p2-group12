import json
import os

def main():
    # 文件路径
    FILE_PATH = "data/candidates/train_candidates_hybrid.json"
    
    if not os.path.exists(FILE_PATH):
        print(f"❌ Error: File not found at {FILE_PATH}")
        return

    print(f"Loading data from {FILE_PATH} ...")
    try:
        with open(FILE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error reading JSON: {e}")
        return

    total_prompts = len(data)
    
    # 初始化统计字典
    stats = {
        "sft": {"covered_prompts": 0, "total_responses": 0},
        "bart": {"covered_prompts": 0, "total_responses": 0},
        "pegasus": {"covered_prompts": 0, "total_responses": 0}
    }

    print("Analyzing...")
    
    for item in data:
        responses = item.get("responses", {})
        
        # 检查 SFT
        sft_keys = [k for k in responses.keys() if k.startswith("sft")]
        if sft_keys:
            stats["sft"]["covered_prompts"] += 1
            stats["sft"]["total_responses"] += len(sft_keys)
            
        # 检查 BART
        bart_keys = [k for k in responses.keys() if k.startswith("bart")]
        if bart_keys:
            stats["bart"]["covered_prompts"] += 1
            stats["bart"]["total_responses"] += len(bart_keys)
            
        # 检查 Pegasus
        pegasus_keys = [k for k in responses.keys() if k.startswith("pegasus")]
        if pegasus_keys:
            stats["pegasus"]["covered_prompts"] += 1
            stats["pegasus"]["total_responses"] += len(pegasus_keys)

    # 打印报表
    print("\n" + "="*60)
    print(f"{'DATA INTEGRITY CHECK REPORT':^60}")
    print("="*60)
    print(f"Total Unique Prompts (Entries): {total_prompts}")
    print("-" * 60)
    print(f"{'Model Name':<15} | {'Prompts Covered':<15} | {'Coverage %':<10} | {'Total Summaries':<15}")
    print("-" * 60)
    
    for model in ["sft", "bart", "pegasus"]:
        covered = stats[model]["covered_prompts"]
        coverage = (covered / total_prompts * 100) if total_prompts > 0 else 0
        total_resp = stats[model]["total_responses"]
        print(f"{model:<15} | {covered:<15} | {coverage:>9.1f}% | {total_resp:<15}")
    
    print("-" * 60)
    
    # 给出建议
    print("\n>>> Analysis & Recommendation:")
    if stats["sft"]["covered_prompts"] > 0 and stats["bart"]["covered_prompts"] > 0:
        if stats["pegasus"]["covered_prompts"] == 0:
            print("✅ Status: GOOD to proceed.")
            print("   You have SFT and BART data. Missing Pegasus is NOT a problem.")
            print("   Action: Run Step 3 (Calibration) directly.")
        elif stats["pegasus"]["covered_prompts"] > 0:
            print("✅ Status: EXCELLENT.")
            print("   You have data from all three models.")
    else:
        print("⚠️ Status: WARNING.")
        print("   It seems you are missing SFT or BART data. Please verify your generation steps.")

if __name__ == "__main__":
    main()