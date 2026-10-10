"""Chinese sentiment entry point; trained local checkpoint."""
import argparse
from pathlib import Path
from bert_chinese_sentiment import predict
ROOT = Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description="本机中文教材情感实验模型")
    parser.add_argument("--text")
    args=parser.parse_args()
    text=args.text or input("请输入要分析的中文句段：")
    if not text.strip():parser.error("文本不能为空")
    args.text=text
    args.model=str(ROOT/"output"/"github_experiment_v2"/"sentiment_model")
    predict(args)

if __name__=="__main__":main()
