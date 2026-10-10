"""Local Chinese sentiment fine-tuning. Original project files are unchanged."""
import argparse
import csv
import json
import random
from pathlib import Path

MODEL_NAME = "google-bert/bert-base-chinese"
LABELS = {0: "负面", 1: "中性", 2: "正面"}


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not {"text", "label"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV 必须有 text,label 两列")
        rows = []
        for line, row in enumerate(reader, 2):
            text = (row.get("text") or "").strip()
            try:
                label = int(row["label"])
            except (ValueError, TypeError):
                raise ValueError(f"{path} 第 {line} 行标签不是整数") from None
            if not text or label not in LABELS:
                raise ValueError(f"{path} 第 {line} 行需非空文本和 0/1/2 标签")
            rows.append((text, label))
    if not rows:
        raise ValueError(f"{path} 没有数据")
    return rows


def train(args):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"训练设备: {device}；基础模型: {MODEL_NAME}")
    train_rows, val_rows = read_csv(args.train), read_csv(args.val)
    if set(t for t, _ in train_rows) & set(t for t, _ in val_rows):
        raise ValueError("训练集与验证集有相同文本，请先消除数据泄漏")
    if set(y for _, y in train_rows) != set(LABELS):
        raise ValueError("当前版本为三分类，训练集必须包含负面、中性、正面三类")
    if set(y for _, y in val_rows) != set(LABELS):
        raise ValueError("验证集也必须包含三类，以便评估每类表现")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3, id2label=LABELS,
        label2id={v: k for k, v in LABELS.items()},
        attn_implementation="eager",
    ).to(device)

    def collate(rows):
        batch = tokenizer([t for t, _ in rows], padding=True, truncation=True,
                          max_length=args.max_length, return_tensors="pt")
        batch["labels"] = torch.tensor([y for _, y in rows], dtype=torch.long)
        return {k: v.to(device) for k, v in batch.items()}

    train_loader = DataLoader(train_rows, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_rows, batch_size=args.batch_size, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("输出目录非空，请换一个 --output，避免覆盖已有模型")
    output.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(batch["labels"])
        model.eval()
        val_loss, correct = 0.0, 0
        confusion = [[0] * 3 for _ in range(3)]
        with torch.no_grad():
            for batch in val_loader:
                result = model(**batch)
                val_loss += result.loss.item() * len(batch["labels"])
                pred = result.logits.argmax(-1)
                correct += (pred == batch["labels"]).sum().item()
                for actual, guessed in zip(batch["labels"].tolist(), pred.tolist()):
                    confusion[actual][guessed] += 1
        f1s = []
        for k in range(3):
            tp = confusion[k][k]
            fp = sum(confusion[i][k] for i in range(3)) - tp
            fn = sum(confusion[k]) - tp
            f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
        metrics = {"epoch": epoch, "train_loss": train_loss / len(train_rows),
                   "val_loss": val_loss / len(val_rows),
                   "val_accuracy": correct / len(val_rows),
                   "val_macro_f1": sum(f1s) / 3, "val_confusion_matrix": confusion}
        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False))
        if metrics["val_loss"] < best_loss:
            best_loss = metrics["val_loss"]
            model.save_pretrained(output)
            tokenizer.save_pretrained(output)
            (output / "training_info.json").write_text(json.dumps(
                {"base_model": MODEL_NAME, "best_epoch": epoch,
                 "max_length": args.max_length, "labels": LABELS,
                 "train_size": len(train_rows), "val_size": len(val_rows),
                 "seed": args.seed, "lr": args.lr, "batch_size": args.batch_size,
                 "metrics": metrics}, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "training_history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已保存验证损失最低的模型: {output.resolve()}")
    print("验证集用于选模型；最终成果请另用未参与训练和选模型的测试集评估。")


def predict(args):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    path = Path(args.model)
    if not (path / "training_info.json").is_file():
        raise ValueError("请指定本脚本训练并保存的目录，不能直接用基础 BERT 做情感预测")
    info = json.loads((path / "training_info.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        path, local_files_only=True, attn_implementation="eager")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    full_length = len(tokenizer(args.text)["input_ids"])
    if full_length > info["max_length"]:
        print("提示：文本过长，本次只分析截断后的开头部分；课文请按段落分析。")
    batch = tokenizer(args.text, return_tensors="pt", truncation=True,
                      max_length=info["max_length"])
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        probabilities = model(**batch).logits.softmax(-1)[0].cpu().tolist()
    index = max(range(len(probabilities)), key=probabilities.__getitem__)
    print(f"预测: {model.config.id2label[index]}")
    print(json.dumps({model.config.id2label[i]: p for i, p in enumerate(probabilities)},
                     ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="中文 BERT 情感三分类：本地训练与预测")
    commands = parser.add_subparsers(dest="command", required=True)
    t = commands.add_parser("train")
    t.add_argument("--train", required=True, help="训练 CSV，列名 text,label")
    t.add_argument("--val", required=True, help="独立验证 CSV，列名 text,label")
    t.add_argument("--output", default="output/bert_chinese_sentiment")
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--batch-size", type=int, default=4)
    t.add_argument("--max-length", type=int, default=256)
    t.add_argument("--lr", type=float, default=2e-5)
    t.add_argument("--seed", type=int, default=42)
    p = commands.add_parser("predict")
    p.add_argument("--model", default="output/bert_chinese_sentiment")
    p.add_argument("--text", required=True)
    args = parser.parse_args()
    if args.command == "train":
        if args.epochs < 1 or args.batch_size < 1 or not 2 <= args.max_length <= 512 or args.lr <= 0:
            parser.error("epochs/batch-size/lr 须大于 0；max-length 须在 2 到 512 之间")
        train(args)
    else:
        if not args.text.strip():
            parser.error("text 不能为空")
        predict(args)


if __name__ == "__main__":
    main()
