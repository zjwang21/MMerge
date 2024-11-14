import json

with open("./sft_enis.json",'r') as f:
    data = json.load(f)

with open("./sft_en_is.is.txt", 'w') as f:
    for k in data:
        f.write(k['hyp'].strip() + "\n")