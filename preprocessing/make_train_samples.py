import torch, sys
import pickle
from transformers import LongformerTokenizer
from tqdm import tqdm
import os
o_path=os.path.abspath(os.path.dirname(os.getcwd()))
sys.path.append(os.path.join(o_path,'utils'))

from decode import  get_arg_span, spans_to_tags

tags2id = {'O': 0, 'B-Review': 1, 'I-Review': 2, 'E-Review': 3, 'S-Review': 4,
           'B-Reply': 1, 'I-Reply': 2, 'E-Reply': 3, 'S-Reply': 4,
           'B': 1, 'I': 2, 'E': 3, 'S': 4}
special_tokens_yi = ['[TAB]', '[LINE]',
                                  '[EQU]', '[URL]', '[NUM]',
                                  '[SPE]', '<sep>', '[q]']
special_tokens_dict_yi = {'additional_special_tokens': special_tokens_yi}

longtokenizer = LongformerTokenizer.from_pretrained(os.path.join(o_path,'longformer-base/'))
longtokenizer.add_special_tokens(special_tokens_dict_yi)
def load_data_new_sample(file_path):

    sample_list_task1_for_review = []
    sample_list_task1_for_reply = []
    sample_list_task2_for_review_dir = []
    sample_list_task2_for_reply_dir = []

    with open(file_path, 'r') as fp:
        rr_pair_list = fp.read().split('\n\n\n')
        for rr_pair in rr_pair_list:
            if rr_pair == '':
                continue
            review, reply = rr_pair.split('\n\n')
            sample_review = {'sentences': [], 'bio_tags': [],
                             'pair_tags': [], 'text_type': None, 'sub_ids': [], 'arg_spans': []}
            for line in review.strip().split('\n'):
                sent, bio_tag, pair_tag, text_type, sub_id = line.strip().split('\t')
                sample_review['sentences'].append(sent)
                sample_review['bio_tags'].append(bio_tag)
                sample_review['pair_tags'].append(pair_tag)
                sample_review['text_type'] = text_type
                sample_review['sub_ids'] = sub_id
            tags_ids = [tags2id[t] for t in sample_review['bio_tags']]

            review_spans=get_arg_span(tags_ids)

            sample_review['arg_spans'] = review_spans

            seq_len = len(tags_ids)

            review_start_positions = []
            review_end_positions = []
            match_labels = torch.zeros([seq_len, seq_len], dtype=torch.long)
            for start, end in review_spans:
                review_start_positions.append(start)
                review_end_positions.append(end)
                if start >= seq_len or end >= seq_len:
                    continue
                match_labels[start, end] = 1

            start_labels = torch.LongTensor([(1 if idx in review_start_positions else 0) for idx in range(
                seq_len)])
            end_labels = torch.LongTensor([(1 if idx in review_end_positions else 0) for idx in range(
                seq_len)])

            sample_review['match_labels'] = match_labels
            sample_review['start_labels'] = start_labels
            sample_review['end_labels'] = end_labels

            sample_review['tag']="task1_review"

            sample_list_task1_for_review.append(sample_review)

            #________________________________________________________________

            sample_reply = {'sentences': [], 'bio_tags': [],
                            'pair_tags': [], 'text_type': None, 'sub_ids': [], 'arg_spans': []}
            for line in reply.strip().split('\n'):
                sent, bio_tag, pair_tag, text_type, sub_id = line.strip().split('\t')
                sample_reply['sentences'].append(sent)
                sample_reply['bio_tags'].append(bio_tag)
                sample_reply['pair_tags'].append(pair_tag)
                sample_reply['text_type'] = text_type
                sample_reply['sub_ids'] = sub_id
            tags_ids = [tags2id[t] for t in sample_reply['bio_tags']]


            reply_spans = get_arg_span(tags_ids)

            sample_reply['arg_spans'] = reply_spans

            seq_len = len(tags_ids)

            reply_start_positions = []
            reply_end_positions = []
            match_labels = torch.zeros([seq_len, seq_len], dtype=torch.long)
            for start, end in reply_spans:
                reply_start_positions.append(start)
                reply_end_positions.append(end)
                if start >= seq_len or end >= seq_len:
                    continue
                match_labels[start, end] = 1

            start_labels = torch.LongTensor([(1 if idx in reply_start_positions else 0) for idx in
                                             range(
                                                 seq_len)])
            end_labels = torch.LongTensor([(1 if idx in reply_end_positions else 0) for idx in
                                           range(
                                               seq_len)])

            sample_reply['match_labels'] = match_labels
            sample_reply['start_labels'] = start_labels
            sample_reply['end_labels'] = end_labels


            sample_reply['tag'] = "task1_reply"
            sample_list_task1_for_reply.append(sample_reply)

            rev_arg_2_rep_arg_dict = {}
            for rev_arg_span in sample_review['arg_spans']:
                rev_arg_pair_id = int(sample_review['pair_tags'][rev_arg_span[0]].split('-')[-1])
                rev_arg_2_rep_arg_dict[rev_arg_span] = []
                for rep_arg_span in sample_reply['arg_spans']:
                    rep_arg_pair_id = int(sample_reply['pair_tags'][rep_arg_span[0]].split('-')[-1])
                    if rev_arg_pair_id == rep_arg_pair_id:
                        rev_arg_2_rep_arg_dict[rev_arg_span].append(rep_arg_span)
            sample_review['rev_arg_2_rep_arg_dict'] = rev_arg_2_rep_arg_dict


            rep_seq_len = len(sample_reply['bio_tags'])



            for rev_arg_span, rep_arg_spans in rev_arg_2_rep_arg_dict.items():

                pair_reply_start_positions = []
                pair_reply_end_positions = []
                pair_match_labels = torch.zeros([rep_seq_len, rep_seq_len], dtype=torch.long)
                for start, end in rep_arg_spans:
                    pair_reply_start_positions.append(start)
                    pair_reply_end_positions.append(end)
                    if start >= rep_seq_len or end >= rep_seq_len:
                        continue
                    pair_match_labels[start, end] = 1

                pair_start_labels = torch.LongTensor([(1 if idx in pair_reply_start_positions else 0) for idx in range(rep_seq_len)])
                pair_end_labels = torch.LongTensor([(1 if idx in pair_reply_end_positions else 0) for idx in range(rep_seq_len)])


                sample_review_dir_temp={}
                sample_review_dir_temp['review_sentences']=sample_review['sentences']
                sample_review_dir_temp['reply_sentences'] = sample_reply['sentences']
                sample_review_dir_temp['match_labels']=pair_match_labels
                sample_review_dir_temp['start_labels'] = pair_start_labels
                sample_review_dir_temp['end_labels'] = pair_end_labels

                sample_review_dir_temp['tag'] ="task2_review"

                temp_rr_dict={}
                tags = spans_to_tags(rep_arg_spans, rep_seq_len)
                temp_rr_dict[rev_arg_span] = tags
                sample_review_dir_temp['rr_arg_dict']=temp_rr_dict

                sample_list_task2_for_review_dir.append(sample_review_dir_temp)


            rep_arg_2_rev_arg_dict = {}


            for rep_arg_span in sample_reply['arg_spans']:
                rep_arg_pair_id = int(sample_reply['pair_tags'][rep_arg_span[0]].split('-')[-1])
                rep_arg_2_rev_arg_dict[rep_arg_span] = []
                for rev_arg_span in sample_review['arg_spans']:
                    rev_arg_pair_id = int(sample_review['pair_tags'][rev_arg_span[0]].split('-')[-1])
                    if rep_arg_pair_id == rev_arg_pair_id:
                        rep_arg_2_rev_arg_dict[rep_arg_span].append(rev_arg_span)
            sample_reply['rep_arg_2_rev_arg_dict'] = rep_arg_2_rev_arg_dict




            rev_seq_len = len(sample_review['bio_tags'])


            for rep_arg_span, rev_arg_spans in rep_arg_2_rev_arg_dict.items():

                pair_review_start_positions = []
                pair_review_end_positions = []
                pair_match_labels = torch.zeros([rev_seq_len, rev_seq_len], dtype=torch.long)
                for start, end in rev_arg_spans:
                    pair_review_start_positions.append(start)
                    pair_review_end_positions.append(end)
                    if start >= rev_seq_len or end >= rev_seq_len:
                        continue
                    pair_match_labels[start, end] = 1

                pair_start_labels = torch.LongTensor([(1 if idx in pair_review_start_positions else 0) for idx in range( rev_seq_len)])
                pair_end_labels = torch.LongTensor([(1 if idx in pair_review_end_positions else 0) for idx in range(rev_seq_len)])


                sample_reply_dir_temp = {}
                sample_reply_dir_temp['review_sentences'] = sample_review['sentences']
                sample_reply_dir_temp['reply_sentences'] = sample_reply['sentences']
                sample_reply_dir_temp['match_labels'] = pair_match_labels
                sample_reply_dir_temp['start_labels'] = pair_start_labels
                sample_reply_dir_temp['end_labels'] = pair_end_labels

                sample_reply_dir_temp['tag'] = "task2_reply"

                temp_rr_dict = {}
                tags = spans_to_tags(rev_arg_spans, rev_seq_len)
                temp_rr_dict[rep_arg_span] = tags
                sample_reply_dir_temp['rr_arg_dict']= temp_rr_dict


                sample_list_task2_for_reply_dir.append(sample_reply_dir_temp)



    return sample_list_task1_for_review,sample_list_task1_for_reply,sample_list_task2_for_review_dir,sample_list_task2_for_reply_dir
def get_bert_emb_for_task1_ids(para_tokens_list):
    question_tokens = '[q]'
    question_tokens_cls_sep = longtokenizer.cls_token + ' ' + question_tokens + ' ' + longtokenizer.sep_token
    question_ids = longtokenizer.convert_tokens_to_ids(question_tokens_cls_sep.split(' '))
    question_length = len(question_ids)

    sent_tokens_list = [sent for para in para_tokens_list for sent in para]
    sent_length_list = [len(sent.split(' ')) for para in para_tokens_list for sent in para]
    passage_tokens = ' '.join(sent_tokens_list)
    passage_tokens_cls_sep = longtokenizer.cls_token + ' ' + passage_tokens + ' ' + longtokenizer.sep_token
    sent_ids = longtokenizer.convert_tokens_to_ids(passage_tokens_cls_sep.split(' '))
    question_sents_ids = [question_ids + sent_ids]
    return question_sents_ids,question_length,sent_length_list

def get_bert_emb_for_task2_ids(para_tokens_list,argument_para_tokens_list):
    argument_tokens=' '.join(argument_para_tokens_list)
    argument_tokens_cls_sep=longtokenizer.cls_token +' '+argument_tokens+' '+longtokenizer.sep_token
    argument_ids=longtokenizer.convert_tokens_to_ids(argument_tokens_cls_sep.split(' '))
    argument_length=len(argument_ids)

    sent_tokens_list = [sent for para in para_tokens_list for sent in para]
    sent_length_list = [len(sent.split(' ')) for para in para_tokens_list for sent in para]
    passage_tokens=' '.join(sent_tokens_list)
    passage_tokens_cls_sep=longtokenizer.sep_token +' '+passage_tokens+' '+longtokenizer.sep_token
    sent_ids = longtokenizer.convert_tokens_to_ids(passage_tokens_cls_sep.split(' '))

    pair_ids=[argument_ids+sent_ids]

    return pair_ids,argument_length,sent_length_list
def get_ids_for_task1_write( para_tokens_list):
    q_s_ids,q_len,sent_len_list = get_bert_emb_for_task1_ids(para_tokens_list)
    return q_s_ids,q_len,sent_len_list
def get_ids_for_task2_write(rev_para_tokens_list,rep_para_tokens_list, arg_pair_sems_list):
    temp_arg_list = []
    for batch_i, pred_arguments_labeldict in enumerate(arg_pair_sems_list):
        for rev_arg_span, label_dict in pred_arguments_labeldict.items():

            temp_argu_o = rev_para_tokens_list[batch_i][rev_arg_span[0]:rev_arg_span[1] + 1]
            temp_arg_list.append(temp_argu_o)

    for arg in temp_arg_list:
        pair_ids,arg_len,sent_len_list = get_bert_emb_for_task2_ids(rep_para_tokens_list, arg)

    return pair_ids,arg_len,sent_len_list
def write_f( para_tokens_list, para_tokens_list_for_2, rr_arg_pair_list,tag_list_o):

    tag=tag_list_o[0]
    if tag == "task1_review" or tag == "task1_reply":
        question_sent_ids,question_len,sent_len_list = get_ids_for_task1_write(para_tokens_list)
        return question_sent_ids,question_len,sent_len_list
    elif tag == "task2_review":
        pair_ids,question_len,sent_len_list= get_ids_for_task2_write(para_tokens_list, para_tokens_list_for_2, rr_arg_pair_list)
        return pair_ids,question_len,sent_len_list
    elif tag == "task2_reply":
        pair_ids,question_len,sent_len_list= get_ids_for_task2_write(para_tokens_list_for_2, para_tokens_list, rr_arg_pair_list)
        return pair_ids,question_len,sent_len_list


train_list_task1_review,train_list_task1_reply, train_list_task2_review,train_list_task2_reply= load_data_new_sample(os.path.join(o_path,'data_v2/train.txt.bioes'))
train_list=train_list_task1_review+train_list_task1_reply+train_list_task2_review+train_list_task2_reply

train_len = len(train_list)
new_sample_list_for_train=[]


batch_id = 0
for batch_i in tqdm(range(train_len)):
    train_batch = train_list[batch_i:(batch_i + 1)]
    para_tokens_list= []
    match_labels, start_labels,end_labels = [], [],[]
    para_tokens_list_for_2 = []
    rr_arg_pair_list=[]
    sample_tags_list=[]
    pair_ids_list=[]
    q_len_list = []
    sent_len_list_list = []
    tt=[]
    for sample in train_batch:
        sample_tags_list.append(sample['tag'])
        if "task1" in sample['tag']:
            para_tokens_list.append(sample['sentences'])
            para_tokens_list_for_2.append([])
            rr_arg_pair_list.append({})
            match_labels.append(sample['match_labels'])
            start_labels.append(sample['start_labels'])
            end_labels.append(sample['end_labels'])

        elif "task2" in sample['tag']:
            para_tokens_list.append(sample['review_sentences'])
            para_tokens_list_for_2.append(sample['reply_sentences'])

            rr_arg_pair_list.append(sample['rr_arg_dict'])

            match_labels.append(sample['match_labels'])
            start_labels.append(sample['start_labels'])
            end_labels.append(sample['end_labels'])

    for sample in train_batch:
        qs_ids, q_len, sents_len_list = write_f(para_tokens_list, para_tokens_list_for_2, rr_arg_pair_list,tag_list_o=sample_tags_list)

        new_sample = {}
        new_sample['tag'] = sample['tag']
        new_sample['qs_ids'] = qs_ids
        new_sample['q_len'] = q_len
        new_sample['sents_len_list'] = sents_len_list
        new_sample['match_labels'] = sample['match_labels']
        new_sample['start_labels'] = sample['start_labels']
        new_sample['end_labels'] = sample['end_labels']

        pair_ids_list.append(new_sample['qs_ids'])
        q_len_list.append(new_sample['q_len'])
        sent_len_list_list.append(new_sample['sents_len_list'])

        new_sample_list_for_train.append(new_sample)


train_samples_path=os.path.join(o_path,'data_v2/all_train_samples.pkl')
with open(train_samples_path, 'wb') as fp:
    pickle.dump(new_sample_list_for_train,fp)




