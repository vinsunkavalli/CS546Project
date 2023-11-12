import os
import json
import glob
import tqdm
import traceback
from argparse import ArgumentParser
import time
import scorer

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig, BertTokenizerFast, BertModel, RobertaTokenizer, XLMRobertaTokenizer, DistilBertTokenizer

from model import OneIE
from config import Config
from util import save_result
from data import IEDatasetEval, IEDataset
from convert import json_to_cs

cur_dir = os.path.dirname(os.path.realpath(__file__))
format_ext_mapping = {'txt': 'txt', 'ltf': 'ltf.xml', 'json': 'json',
                      'json_single': 'json'}

def load_model(bert_model_name, model_path, device=0, gpu=False, beam_size=5):
    print('Loading the model from {}'.format(model_path))
    map_location = 'cuda:{}'.format(device) if gpu else 'cpu'
    state = torch.load(model_path, map_location=map_location)

    config = state['config']
    if type(config) is dict:
        config = Config.from_dict(config)
    config.bert_cache_dir = os.path.join(cur_dir, 'bert')
    vocabs = state['vocabs']
    valid_patterns = state['valid']

    # recover the model
    model = OneIE(config, vocabs, valid_patterns)
    model.load_state_dict(state['model'])
    model.beam_size = beam_size
    if gpu:
        model.cuda(device)
    #changing bert model
    #bert_model_name = 'bert-large-cased-whole-word-masking-finetuned-squad'
    #tokenizer = BertTokenizer.from_pretrained(new_bert_model, cache_dir=config.bert_cache_dir, do_lower_case=False)

    tokenizer = None

    if bert_model_name.startswith('bert-'):
        tokenizer = BertTokenizer.from_pretrained(bert_model_name, cache_dir=config.bert_cache_dir, do_lower_case=False)

    elif bert_model_name.startswith('roberta-'):
        tokenizer = RobertaTokenizer.from_pretrained(bert_model_name, cache_dir=config.bert_cache_dir, do_lower_case=False)
                                            
    elif bert_model_name.startswith('xlm-roberta-'):
        tokenizer = XLMRobertaTokenizer.from_pretrained(bert_model_name, cache_dir=config.bert_cache_dir, do_lower_case=False)

    elif bert_model_name.startswith('distilbert-'):
        tokeizer = DistilBertTokenizer.from_pretrained(bert_model_name, cache_dir=config.bert_cache_dir, do_lower_case=False)

    else:
        raise ValueError('Unknown model: {}'.format(bert_model_name))
    
    model.load_bert(bert_model_name, cache_dir=config.bert_cache_dir)
    #tokenizer = BertTokenizer.from_pretrained(config.bert_model_name,
    #                                          cache_dir=config.bert_cache_dir,
    #                                          do_lower_case=False)

    return model, tokenizer, config


def predict_document(path, model, tokenizer, config, batch_size=20, 
                     max_length=128, gpu=False, input_format='txt',
                     language='english'):

    test_set = IEDatasetEval(path, max_length=max_length, gpu=gpu,
                             input_format=input_format, language=language)
    #test_set = IEDataset(config.test_file, gpu=gpu,
    #                 relation_mask_self=config.relation_mask_self,
    #                 relation_directional=config.relation_directional,
    #                 symmetric_relations=config.symmetric_relations)
    test_set.numberize(tokenizer)
    # document info
    info = {
        'doc_id': test_set.doc_id,
        'ori_sent_num': test_set.ori_sent_num,
        'sent_num': len(test_set)
    }
    # prediction result
    result = []
    for batch in DataLoader(test_set, batch_size=batch_size, shuffle=False, 
                            collate_fn=test_set.collate_fn):
        graphs = model.predict(batch)
        for graph, tokens, sent_id, token_ids in zip(graphs, batch.tokens,
                                                     batch.sent_ids,
                                                     batch.token_ids):
            graph.clean(relation_directional=config.relation_directional,
                        symmetric_relations=config.symmetric_relations)
            result.append((sent_id, token_ids, tokens, graph))
        #scorer.score_graphs(batch,graphs)
    return result, info


def predict(new_bert_model, model_path, input_path, output_path, log_path=None, cs_path=None,
         batch_size=50, max_length=128, device=0, gpu=False,
         file_extension='txt', beam_size=5, input_format='txt',
         language='english'):

    # set gpu device
    if gpu:
        torch.cuda.set_device(device)
    # load the model from file
    model, tokenizer, config = load_model(new_bert_model, model_path, device=device, gpu=gpu,
                                          beam_size=beam_size)
    # get the list of documents
    file_list = glob.glob(os.path.join(input_path, '*.{}'.format(file_extension)))
    # log writer
    if log_path:
        log_writer = open(log_path, 'w', encoding='utf-8')
    # run the model; collect result and info
    start = time.time()
    doc_info_list = []
    progress = tqdm.tqdm(total=len(file_list), ncols=75)
    for f in file_list:
        progress.update(1)
        try:
            doc_result, doc_info = predict_document(
                f, model, tokenizer, config, batch_size=batch_size,
                max_length=max_length, gpu=gpu, input_format=input_format,
                language=language)
            # save json format result
            doc_id = doc_info['doc_id']
            with open(os.path.join(output_path, '{}.json'.format(doc_id)), 'w') as w:
                for sent_id, token_ids, tokens, graph in doc_result:
                    output = {
                        'doc_id': doc_id,
                        'sent_id': sent_id,
                        'token_ids': token_ids,
                        'tokens': tokens,
                        'graph': graph.to_dict()
                    }
                    w.write(json.dumps(output) + '\n')
            # write doc info
            if log_path:
                log_writer.write(json.dumps(doc_info) + '\n')
                log_writer.flush()
        except Exception as e:
            traceback.print_exc()
            if log_path:
                log_writer.write(json.dumps(
                    {'file': file, 'message': str(e)}) + '\n')
                log_writer.flush()
    progress.close()
    print("Time: "+str(time.time()-start))
    # convert to the cold-start format
    if cs_path:
        print('Converting to cs format')
        json_to_cs(output_path, cs_path)


parser = ArgumentParser()
parser.add_argument('-n', '--new_bert_model', help="name of alternative bert model")
parser.add_argument('-m', '--model_path', help='path to the trained model')
parser.add_argument('-i', '--input_dir', help='path to the input folder (ltf files)')
parser.add_argument('-o', '--output_dir', help='path to the output folder (json files)')
parser.add_argument('-l', '--log_path', default=None, help='path to the log file')
parser.add_argument('-c', '--cs_dir', default=None, help='path to the output folder (cs files)')
parser.add_argument('--gpu', action='store_true', help='use gpu')
parser.add_argument('-d', '--device', default=0, type=int, help='gpu device index')
parser.add_argument('-b', '--batch_size', default=10, type=int, help='batch size')
parser.add_argument('--max_len', default=128, type=int, help='max sentence length')
parser.add_argument('--beam_size', default=5, type=int, help='beam set size')
parser.add_argument('--lang', default='english', help='Model language')
parser.add_argument('--format', default='txt', help='Input format (txt, ltf, json)')

args = parser.parse_args()
extension = format_ext_mapping.get(args.format, 'ltf.xml')

predict(
    new_bert_model = args.new_bert_model,
    model_path=args.model_path,
    input_path=args.input_dir,
    output_path=args.output_dir,
    cs_path=args.cs_dir,
    log_path=args.log_path,
    batch_size=args.batch_size,
    max_length=args.max_len,
    device=args.device,
    gpu=args.gpu,
    beam_size=args.beam_size,
    file_extension=extension,
    input_format=args.format,
    language=args.lang,
)