import pandas as pd 
import json 

with open('./data/2wikimqa_e.jsonl', 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

print(data[12])
d = []
fixed = '2wikimqa_e'
iter_ = 0
for i in data:
   iter_ +=1
   tmp = {
           "id": f"{fixed}_{iter_}",
           "input": i["input"]
           } 
   d.append(tmp)
#raw = pd.read_json('./data/2wikimqa_e.jsonl', lines=True)
#print(raw.head(5))
#
#new_gold_test = raw[['input', '_id']]
#print(new_gold_test.head(4))

d = pd.DataFrame(d)
#
#new_gold_test.to_csv('llama2-gold-test.csv', index=False)
d.to_csv('llama2-gold-test_custom_id.csv', index=False)
