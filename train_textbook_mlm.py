"""Reproduce the saved 60-step textbook MLM stage using the recorded partition."""
import os,json,random
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME',str(ROOT/'models'/'hf_cache'))
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer,AutoModelForMaskedLM,DataCollatorForLanguageModeling

def main():
    random.seed(42);torch.manual_seed(42);torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    raw=ROOT/'data'/'github_experiment'
    rows=json.loads((raw/'textbook_train.json').read_text(encoding='utf-8'))
    base=ROOT/'models'/'bert-base-chinese'
    tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True)
    encoded=[tokenizer(r['text'],truncation=True,max_length=128) for r in rows]
    loader=DataLoader(encoded,batch_size=4,shuffle=True,
        collate_fn=DataCollatorForLanguageModeling(tokenizer,mlm_probability=0.15))
    model=AutoModelForMaskedLM.from_pretrained(base,local_files_only=True).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=0.01)
    step=0;history=[];model.train()
    while step<60:
        for batch in loader:
            batch={k:v.to(device) for k,v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                loss=model(**batch).loss
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
            step+=1;history.append(loss.item())
            if step%10==0: print('MLM',step,loss.item(),flush=True)
            if step>=60:break
    out=ROOT/'output'/'github_experiment_v1'/'textbook_mlm'
    model.save_pretrained(out);tokenizer.save_pretrained(out)
    (out.parent/'mlm_history.json').write_text(json.dumps(history),encoding='utf-8')
    print('Saved',out)

if __name__=='__main__':main()
