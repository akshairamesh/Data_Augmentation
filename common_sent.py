import re
fpath = '/home/rameshak/spinning-storage/rameshak/wmt2020/scripts/masked_word_substitution/'
entalist = []
mask = '[MASK]'
engkey = 'English'
tamkey = 'German'
common_words = open(fpath+'common_words.en','r',encoding='utf-8').readlines()
english_data = open(fpath+'train.en','r',encoding='utf-8').readlines()
german_data = open(fpath+'train.ta','r',encoding='utf-8').readlines()
for l in range(0,len(english_data)):
    for cw in common_words:
        dicts = {}
        if re.search(r'\b'+cw+r'\b',english_data[l]):
            sentence = re.sub(r'\b'+cw+r'\b',mask,english_data[l],count=1)
            dicts.update({engkey:sentence})
            dicts.update({gerkey:german_data[l]})
            entalist.append(dicts)
print(entalist)
