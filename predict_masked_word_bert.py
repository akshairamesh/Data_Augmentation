import os
import re
import torch
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM, BertModel

top_k = 250
syn_corpus_size = 0
rarewords_dict = {}
giza_dict = {}
PAD, MASK, CLS, SEP = '[PAD]', '[MASK]', '[CLS]', '[SEP]'
fpath = '/home/rameshak/spinning-storage/rameshak/wmt2020/scripts/masked_word_substitution/'
src_vocab_list = []
trg_vocab_list = []

def to_bert_input(tokens, bert_tokenizer):
    token_idx = torch.tensor(bert_tokenizer.convert_tokens_to_ids(tokens))
    sep_idx = tokens.index('[SEP]')
    segment_idx = token_idx * 0
    segment_idx[(sep_idx + 1):] = 1
    mask = (token_idx != 0)
    return token_idx.unsqueeze(0), segment_idx.unsqueeze(0), mask.unsqueeze(0)

def topk_suggestions_bertlm(src_message):
    tokens = bert_tokenizer.tokenize(src_message)
    #Add CLS and SEP if needed.
    if tokens[0] != CLS:
        tokens = [CLS] + tokens
    if tokens[-1] != SEP:
        tokens.append(SEP)
    token_idx, segment_idx, mask = to_bert_input(tokens, bert_tokenizer)
    with torch.no_grad():
        logits = bert_model(token_idx, segment_idx, mask, masked_lm_labels=None)
    logits = logits.squeeze(0)
    probs = torch.softmax(logits, dim=-1)

    topk_tokens = []
    for idx, token in enumerate(tokens):
        if token == MASK:
            topk_prob, topk_indices = torch.topk(probs[idx, :], top_k)
            topk_tokens = bert_tokenizer.convert_ids_to_tokens(topk_indices.cpu().numpy())

    topk_set = set(topk_tokens)

    return topk_set

def load_lex_dict(fpath,filename):
    lex_dict = {}
    file_content = open(fpath+filename,'r',encoding='utf-8')
    for line in file_content:
        line = line.strip()
        words = line.split(',')
        src,tgt = words[0],words[1]
        lex_dict[src] = tgt
    return lex_dict

def load_giza_dict(fpath,filename):
    #src_tgt_dict = {}
    src_list = []
    tgt_list = []
    snt_src_list = []
    snt_tgt_list = []
    stlist = []
    chck_str = '!@#$%^&*()_+.-,;:—?&()"/[]{}\|='
    spl_str = '##at##-##at##'
    giza_contents = open(fpath+'ende_alignment.txt','r',encoding='utf-8').readlines()
    for i in range(0,len(giza_contents)):
        words = giza_contents[i].split(' ')
        #print(len(words))
        for j in range(0,len(words),2):
            tgt_word = words[j+1][words[j+1].find("(")+1:words[j+1].rfind(")")]
            #print(tgt_word)
            if ((tgt_word in chck_str) or (tgt_word in spl_str) or (tgt_word == '') or len(tgt_word.split(','))>1):
                tgt_word = ''
            src_list.append(words[j])
            #print(src_list)
            tgt_list.append(tgt_word)
        snt_src_list.append(src_list)
        snt_tgt_list.append(tgt_list)
        src_list = []
        tgt_list = []
    return snt_tgt_list

def search_common_key(sntno,idx):
    tgt_snt = giza_dict[sntno]
    tgt_word = tgt_snt[idx]
    return tgt_word

def search_rare_key(key):
    chck_str = '!@#$%^&*()_+.-,;:—?&()"/[]{}\|='
    spl_str = '##at##-##at##'
    new_aug_word = ''
    #word = [tgt for src,tgt in lex_contents.items() if src == key]
    if key in rarewords_dict:
        new_aug_word = rarewords_dict[key]
    if ((new_aug_word in chck_str) or (new_aug_word in spl_str) or (new_aug_word == '')):
        new_aug_word = ''
    return new_aug_word


def generate_synthetic_corpus( src_sent, trg_sent, masked_index, trg_index, rareword, trg_augword):
    #print('size of src sent vocab : ',len(src_sent))
    #print('size of trg sent vocab : ',len(trg_sent))
    src_sent[masked_index] = rareword
    trg_sent[trg_index] = trg_augword
    src_message = ' '.join(src_sent)
    trg_message = ' '.join(trg_sent)
    print('added in src side :',src_message)
    print('added in trg side :',trg_message)
    with open("syn_english.txt", "a") as src_file, open("syn_german.txt", "a") as trg_file:
        src_file.write(src_message+'\n')
        trg_file.write(trg_message+'\n')

def generate_rarewords_subset():
    #rare words stored in rare_words_set
    filename = 'RareWords.txt'
    rare_words_file = open(fpath+filename,'r',encoding='utf-8')
    rare_words = rare_words_file.readlines()
    rare_words = ' '.join(rare_words).replace('\n','').split()
    rare_words_set = set(rare_words)

    return rare_words_set


if __name__ == '__main__':
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    #lex dictionary
    rarewords_dict = load_lex_dict(fpath,'lex_dict.txt')
    #giza dictionary
    giza_dict = load_giza_dict(fpath, 'ende_alignment.txt')
    #generate rarewords subset
    rare_words_set = generate_rarewords_subset()
    mask_cnt = 0
    with open('src_corpus.en','r') as srcfile, open('trg_corpus.de','r') as trgfile:
        while True :
            src_message = srcfile.readline().rstrip()
            trg_message = trgfile.readline().rstrip()
            src_sent_list = src_message.split(' ')
            trg_sent_list = trg_message.split(' ')
            mask_cnt = mask_cnt + 1
            #print line number
            print('For line num : ', mask_cnt)
            #print src-target sentence
            print(src_sent_list)
            print(trg_sent_list)

            #bert language model suggests top_k words
            topk_set = topk_suggestions_bertlm(src_message)
            #rare_subset i.e intersection of top_k words and rarewords list.
            rare_subset = topk_set.intersection(rare_words_set)
            if(len(rare_subset) == 0):
                print('No word suggestions are available in training data' )
                continue
            else :
                masked_index = src_sent_list.index(MASK)
                #print('token position is : ', masked_index)
                #identify target side word index corresponding to given value.
                trg_word = search_common_key( mask_cnt - 1, masked_index-1)
                print('token found at :',masked_index)
                print('word to be replaced :',trg_word)
                if(trg_word == '' ):
                    continue
                else :
                    trg_index = trg_sent_list.index(trg_word)
                    print('corresponding target side index is : ',trg_index)
                    print('word suggestions by bert are :',rare_subset)
                    for word in list(rare_subset) :
                        trg_augword = search_rare_key(word)
                        if(trg_augword != ''):
                            generate_synthetic_corpus(src_sent_list, trg_sent_list, masked_index, trg_index, word, trg_augword)
                        else :
                            continue

