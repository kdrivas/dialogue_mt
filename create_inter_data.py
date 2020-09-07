import os
import json
from pathlib import Path

DATASETS = ['wmtchat2020', 'openSub_ende', 'openSub_enes', 'openSub_enfr', 'openSub_enru']
PARTITIONS = ['train', 'dev', 'test']

def main(base_path='data', type_c='agnostic'):
    
    base_path = Path(base_path)
    raw_path = base_path / 'raw'
    inter_path = base_path / 'inter'
    for dataset in DATASETS:
        print(f'Processing {dataset} ...')
        if dataset == 'wmtchat2020':
            lang_src = 'en'
            lant_tgt = 'de'
            spk_b = 'agent'
            spk_a = 'customer'
        elif 'openSub' in dataset:
            pair_lang = dataset.split('_')[1]
            lang_src = pair_lang[:2]
            lant_tgt = pair_lang[2:]
            spk_b = '<en>'
            spk_a = '<2en>'
        else:
            raise NotImplementedError
            
        for partition in PARTITIONS:
            print(f'Reading {partition} ...')
            with open(raw_path / dataset / f'{partition}.json') as f:
                data = json.load(f)
                
            data_src = [] # Always english
            data_tgt = [] 
            for conversation in data:
                for block in data[conversation]:
                    if type_c == 'agnostic':
                        if block['speaker'] == spk_a:
                            data_src.append(block['source'])
                            data_tgt.append(block['target'])
                        elif block['speaker'] == spk_b:
                            data_src.append(block['source'])
                            data_tgt.append(block['target'])
                        else:
                            raise NotImplementedError
                    else:
                        raise NotImplementedError
                  
            os.makedirs(inter_path / type_c / dataset, exist_ok=True)
            
            with open(inter_path / type_c / dataset / f'{partition}.input', 'w') as f:
                for line in data_src:
                    if len(line):
                        print(line, file=f)
                    
            with open(inter_path / type_c / dataset / f'{partition}.output', 'w') as f:
                for line in data_tgt:
                    if len(line):
                        print(line, file=f)
                        
        with open(raw_path / dataset / f'{partition}.json') as f:
            data = json.load(f)
                    
main()
