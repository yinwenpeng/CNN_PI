# -*- coding: utf-8 -*- 




import os


root="/mounts/data/proj/wenpeng/Dataset/StanfordSentiment/stanfordSentimentTreebank/"

sentid2sent={}
sentid2split={}

phraseid2label={}
phrase2id={}
phrase2label={}
sent_label={}

task=2

file_id2sent=open(root+"datasetSentences.txt", 'r')
line_no=0
for line in file_id2sent:
    if line_no >0:
        tokens=line.strip().split('\t')
        sentid2sent[tokens[0]]=tokens[1].replace('-LRB-', '(').replace('-RRB-', ')')
    line_no=line_no+1
file_id2sent.close()
print 'Totally, '+str(len(sentid2sent))+' samples.'
file_id2split=open(root+"datasetSplit.txt", 'r')
line_no=0
train_count=0
dev_count=0
test_count=0
for line in file_id2split:
    if line_no >0:
        tokens=line.strip().split(',')
        sentid2split[tokens[0]]=tokens[1]
        if tokens[1] == '1':
            train_count=train_count+1
        elif tokens[1] == '2':
            test_count=test_count+1           
        elif tokens[1] =='3':
            dev_count=dev_count+1
    line_no=line_no+1
#print '\t there should be '+str(train_count)+' train, '+str(dev_count)+' dev, '+str(test_count)+' test samples.'
file_id2split.close()
#split into three dataset: train, dev, test
train_sents=[] # can not use set(), for some samples repeat
test_sents=[]
dev_sents=[]
for (sentid, sent) in sentid2sent.items():
    if sentid2split[sentid]=='1':
        train_sents.append(sent)
    elif sentid2split[sentid]=='2':
        test_sents.append(sent)
    else:
        dev_sents.append(sent)
print '\t there should be '+str(len(train_sents))+' train, '+str(len(dev_sents))+' dev, '+str(len(test_sents))+' test samples.'


#map sentence to label now
file_phrase2id=open(root+"dictionary.txt", 'r')
for line in file_phrase2id:
    tokens=line.strip().split('|')
    #phrase2id[tokens[0].replace('(', '-LRB-').replace(')','-RRB-')]=tokens[1]
    phrase2id[tokens[0]]=tokens[1]
file_phrase2id.close()

file_phraseid2label=open(root+"sentiment_labels.txt", 'r')
line_no=0
for line in file_phraseid2label:
    if line_no >0:
        tokens=line.strip().split('|')
        phraseid2label[tokens[0]]=float(tokens[1])
    line_no=line_no+1
file_phraseid2label.close()
#integrate above two
for phrase in phrase2id:
    if task==5:
        if phraseid2label[phrase2id[phrase]]<=0.2 :
            phrase2label[phrase]=1
        elif phraseid2label[phrase2id[phrase]]<=0.4 :
            phrase2label[phrase]=2
        elif phraseid2label[phrase2id[phrase]]<=0.6 :
            phrase2label[phrase]=3
        elif phraseid2label[phrase2id[phrase]]<=0.8 :
            phrase2label[phrase]=4
        else:
            phrase2label[phrase]=5
    elif task ==2:
        if phraseid2label[phrase2id[phrase]]<=0.4 :
            phrase2label[phrase]=1
        elif phraseid2label[phrase2id[phrase]]> 0.6 :
            phrase2label[phrase]=2
        #else:
        #    phrase2label[phrase]=0 # 0 means neutral
    
#creat three files to store: train.txt, dev.txt, test.txt
train=open(root+str(task)+"classes/rich_train.txt", 'w')
dev=open(root+str(task)+"classes/rich_dev.txt", 'w')
test=open(root+str(task)+"classes/rich_test.txt", 'w')
train_no=0
dev_no=0
test_no=0
i=0
total=len(phrase2label)
print '\t Totally, '+str(total)+' phrases.'
train_set=set()
'''
for sent in train_sents:
    for (phrase, label) in phrase2label.items():
        if phrase in sent:
            if phrase not in train_set:
                train_no=train_no+1
                train.write(str(label)+'\t'+phrase+'\n')
                train_set.add(phrase)
'''
for (phrase, label) in phrase2label.items():
    if phrase not in test_sents and phrase not in dev_sents and phrase not in train_set:
        train_no=train_no+1
        train.write(str(label)+'\t'+phrase+'\n')
        train_set.add(phrase)           
for sent in test_sents:
    if phrase2label.has_key(sent):
        test_no=test_no+1
        test.write(str(phrase2label[sent])+'\t'+sent+'\n') 
for sent in dev_sents:
    if phrase2label.has_key(sent):
        dev_no=dev_no+1
        dev.write(str(phrase2label[sent])+'\t'+sent+'\n') 
          
print '\t Actually, train '+str(train_no)+', dev '+str(dev_no)+', test '+str(test_no)
train.close()
dev.close()
test.close()


    












