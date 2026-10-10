import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME',str(ROOT/'models'/'hf_cache'))
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
import csv,json,re,random,time,hashlib,urllib.request,collections
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer,AutoModelForMaskedLM,AutoModelForSequenceClassification,DataCollatorForLanguageModeling

SEED=42
D2L='e6b18ccea71451a55fcd861d7b96fddf2587b09a'
SENT='0ce815566c8b19fb678d2b37eb9809ce94df7848'
LABELS={0:'负面',1:'中性',2:'正面'}
RAW=ROOT/'data'/'github_experiment'
OUT=ROOT/'output'/'github_experiment_v2'

def fetch(url,path):
    if path.exists(): return path.read_bytes()
    for attempt in range(3):
        try:
            content=urllib.request.urlopen(url,timeout=60).read()
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(content)
            return content
        except Exception:
            if attempt==2: raise
            time.sleep(2)

def save_json(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def metrics(actual,pred):
    cm=[[0]*3 for _ in range(3)]
    for a,p in zip(actual,pred): cm[a][p]+=1
    f1=[]
    for k in range(3):
        tp=cm[k][k]; fp=sum(cm[i][k] for i in range(3))-tp; fn=sum(cm[k])-tp
        f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
    return {'accuracy':sum(a==p for a,p in zip(actual,pred))/len(actual),
            'macro_f1_labeled_classes':sum(f1[k] for k in sorted(set(actual)))/len(set(actual)), 'evaluated_labels':sorted(set(actual)),'per_class_f1':dict(zip(LABELS.values(),f1)),
            'confusion_matrix_true_rows':cm,'size':len(actual)}

def main():
    random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(4)
    RAW.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('DEVICE',device,flush=True)
    sources=[]; paragraphs=[]
    for chapter in ['linear-algebra','calculus','probability']:
        url=f'https://raw.githubusercontent.com/d2l-ai/d2l-zh/{D2L}/chapter_preliminaries/{chapter}.md'
        content=fetch(url,RAW/f'{chapter}.md')
        sources.append({'url':url,'sha256':hashlib.sha256(content).hexdigest()})
        text=content.decode('utf-8')
        text=re.sub(r'```.*?```','',text,flags=re.S)
        text=re.sub(r'\$\$.*?\$\$','',text,flags=re.S)
        text=re.sub(r'\$[^$\n]*\$','',text)
        for p in re.split(r'\n\s*\n',text):
            p=re.sub(r':(?:label|eqref|numref|cite):`[^`]*`','',p)
            p=re.sub(r'!\[[^\]]*\]\([^)]*\)','',p)
            p=re.sub(r'\[([^\]]+)\]\([^)]*\)',r'\1',p)
            p=re.sub(r'\s+',' ',p).strip()
            if len(re.findall(r'[\u4e00-\u9fff]',p))>=25:
                paragraphs.append({'chapter':chapter,'text':p[:800]})
    fetch(f'https://raw.githubusercontent.com/d2l-ai/d2l-zh/{D2L}/LICENSE',RAW/'D2L_LICENSE')
    commit='34065bb439077d1affe8ae8d6dca20af9081f7bd'
    corpus='https://raw.githubusercontent.com/qweraqq/NLPCC2014_sentiment/'+commit+'/'
    fetch(corpus+'README.md',RAW/'NLPCC_README.md')
    buckets={0:[],2:[]}
    for label,name in [(0,'sample.negative.txt'),(2,'sample.positive.txt')]:
        content=fetch(corpus+name,RAW/name)
        sources.append({'url':corpus+name,'sha256':hashlib.sha256(content).hexdigest()})
        texts=[]
        for line in content.decode('utf-8-sig').splitlines():
            line=re.sub(r'\s+',' ',line).strip()
            if line and not line.startswith('<') and len(line)>=5: texts.append(line)
        buckets[label]=sorted(set(texts))
    overlap=set(buckets[0])&set(buckets[2])
    splits={k:[] for k in ['train','val','test']}
    for label,texts in buckets.items():
        texts=[t for t in texts if t not in overlap];random.shuffle(texts)
        if len(texts)<1400:raise ValueError('Too few sentiment samples')
        splits['train'] += [(t,label) for t in texts[:1000]]
        splits['val'] += [(t,label) for t in texts[1000:1200]]
        splits['test'] += [(t,label) for t in texts[1200:1400]]
    # Reuse the exact partition used by the saved MLM checkpoint.
    heldout=json.loads((RAW/'textbook_holdout.json').read_text(encoding='utf-8'))
    textbook_train=json.loads((RAW/'textbook_train.json').read_text(encoding='utf-8'))
    neutral=[]
    for p in textbook_train:
        for sentence in re.split(r'(?<=[。！？])',p['text']):
            sentence=sentence.strip()
            if 25<=len(sentence)<=160 and re.search('定义|称为|表示|等于|概率|向量|矩阵|导数',sentence) and not re.search('喜欢|惊讶|希望|有趣|漂亮|糟|遗憾|高兴|令人',sentence):
                neutral.append(sentence)
    neutral=sorted(set(neutral))
    if len(neutral)<10:raise ValueError('Insufficient textbook neutral candidates')
    # Weak labels: these are factual-looking textbook sentences, not human gold labels.
    save_json(RAW/'weak_neutral_training.json',{'label_source':'heuristic, not human verified','texts':neutral})
    splits['train'] += [(neutral[i%len(neutral)],1) for i in range(1000)]
    for name,rows in splits.items():
        random.shuffle(rows)
        with (RAW/f'{name}_v2.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.writer(f);w.writerow(['text','label']);w.writerows(rows)
    save_json(RAW/'textbook_train_v2.json',textbook_train);save_json(RAW/'textbook_holdout_v2.json',heldout)
    save_json(OUT/'provenance.json',{'sources':sources,'seed':SEED,'train_rows':3000,'val':400,'test':400,
        'train_gold_sentiment':2000,'weak_neutral_unique':len(neutral),'weak_neutral_oversampled':1000,
        'textbook_train_paragraphs':len(textbook_train),'textbook_holdout_paragraphs':len(heldout),
        'sentiment_labels':'NLPCC2014 positive/negative source files; 0 negative, 2 positive',
        'neutral_labels':'Heuristically selected factual textbook sentences; weak labels, not gold',
        'textbook_test_labels':'None; predictions are not accuracy evaluation',
        'evaluation_scope':'Gold validation/test include only positive and negative; neutral accuracy unknown',
        'split':'Exact deduplication; conflicting texts removed; random disjoint subsets; near duplicates may remain',
        'rejected_source':'lights-think dataset: README mapping contradicted by inspected samples',
        'mlm_checkpoint':'github_experiment_v1/textbook_mlm; 60 updates on textbook training paragraphs'})
    base=ROOT/'models'/'bert-base-chinese'
    tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True)
    adapted=ROOT/'output'/'github_experiment_v1'/'textbook_mlm'
    model=AutoModelForSequenceClassification.from_pretrained(adapted,num_labels=3,id2label=LABELS,
          label2id={v:k for k,v in LABELS.items()},attn_implementation='sdpa').to(device)
    def collate(rows):
        batch=tokenizer([r[0] for r in rows],padding=True,truncation=True,max_length=128,return_tensors='pt')
        batch['labels']=torch.tensor([r[1] for r in rows]);return {k:v.to(device) for k,v in batch.items()}
    loaders={k:DataLoader(v,batch_size=8,shuffle=k=='train',collate_fn=collate) for k,v in splits.items()}
    def evaluate(loader):
        model.eval(); actual=[];pred=[];total_loss=0
        with torch.no_grad():
            for batch in loader:
                result=model(**batch);total_loss+=result.loss.item()*len(batch['labels'])
                actual+=batch['labels'].tolist();pred+=result.logits.argmax(-1).tolist()
        result=metrics(actual,pred);result['loss']=total_loss/len(actual)
        return result
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=0.01)
    history=[];best=float('inf');saved=OUT/'sentiment_model'
    start=time.time()
    for epoch in range(1,4):
        model.train();running=0
        for step,batch in enumerate(loaders['train'],1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                loss=model(**batch).loss
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
            running+=loss.item()*len(batch['labels'])
            if step%50==0:print('SFT',epoch,step,'/',len(loaders['train']),'elapsed',round(time.time()-start),flush=True)
        result=evaluate(loaders['val']);result['epoch']=epoch;result['train_loss']=running/len(splits['train'])
        history.append(result);print('VAL',json.dumps(result,ensure_ascii=False),flush=True)
        save_json(OUT/'training_history.json',history)
        if result['loss']<best:
            best=result['loss'];model.save_pretrained(saved);tokenizer.save_pretrained(saved)
            save_json(saved/'training_info.json',{'base_model':'google-bert/bert-base-chinese','max_length':128,'best_epoch':epoch,'labels':LABELS})
    del model,optimizer
    if device.type=='cuda':torch.cuda.empty_cache()
    model=AutoModelForSequenceClassification.from_pretrained(saved,local_files_only=True).to(device)
    test=evaluate(loaders['test']);save_json(OUT/'test_metrics.json',test)
    print('TEST',json.dumps(test,ensure_ascii=False),flush=True)
    model.eval();results=[]
    with torch.no_grad():
        for row in heldout:
            batch=tokenizer(row['text'],truncation=True,max_length=128,return_tensors='pt').to(device)
            probs=model(**batch).logits.softmax(-1)[0].cpu().tolist();pred=max(range(3),key=probs.__getitem__)
            results.append({**row,'prediction':LABELS[pred],'probabilities':dict(zip(LABELS.values(),probs)),
                            'truncated':len(tokenizer(row['text'])['input_ids'])>128})
    save_json(OUT/'textbook_predictions.json',results)
    print('COMPLETE',str(OUT),flush=True)

if __name__=='__main__':main()
