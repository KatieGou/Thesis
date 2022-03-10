import json
from googletrans import Translator
from tqdm import tqdm
import os.path as op
import time

def translate(s: str) -> str:
    translator=Translator()
    translator.raise_Exception = True
    res=translator.translate(s, dest='sv')
    return res.text

def translate_objects(fin: str):
    """translate object tags into Swedish

    Args:
        fin (str): input file name
    """    
    with open(fin, 'r') as fp:
        classes=fp.readlines()
    # print('classes:', classes)
    fout=op.splitext(fin)[0]+'_sv'+op.splitext(fin)[1]
    with open(fout, 'w', encoding='utf-8') as fp:
        for c in tqdm(classes):
            fp.write(translate(c))
            fp.write('\n')

def translate_captions(fin: str):
    """read from json file and translate captions into Swedish

    Args:
        fin (str): input file name

    Raises:
        ValueError: in case caption id duplicated
    """    
    image_captions_sv=dict()
    with open(fin, 'r') as fp:
        image_captions = json.load(fp)
    for image_id in tqdm(image_captions.keys()):
        time.sleep(0.1)
        captions=image_captions[image_id]
        for caption in captions:
            cap=caption['caption']
            caption_id=caption['caption_id']
            if caption_id in image_captions_sv.keys():
                raise ValueError('caption_id already exists!')
            image_captions_sv[caption_id]=translate(cap)
    fout=op.splitext(fin)[0]+'_sv'+op.splitext(fin)[1]
    with open(fout, 'w', encoding='utf-8') as fp:
        json.dump(image_captions_sv, fp, ensure_ascii=False)

def main():
    # translate_objects('objects_vocab.txt')
    translate_captions(fin='image_captions_train2017.json')

if __name__=='__main__':
    main()


'''
To run in Colab

import json
from googletrans import Translator
from tqdm import tqdm
import os.path as op

def translate(s: str) -> str:
    translator=Translator()
    translator.raise_Exception = True
    res=translator.translate(s, dest='sv')
    return res.text

def translate_objects(fin: str):
    """translate object tags into Swedish

    Args:
        fin (str): input file name
    """    
    with open(fin, 'r') as fp:
        classes=fp.readlines()
    # print('classes:', classes)
    fout=op.splitext(fin)[0]+'_sv'+op.splitext(fin)[1]
    with open(fout, 'w', encoding='utf-8') as fp:
        for c in tqdm(classes):
            fp.write(translate(c))
            fp.write('\n')

def read_saved(fin: str):
    try:
        with open(fin, 'r') as f:
            d=json.load(f)
        return len(d), d
    except FileNotFoundError:
        return (0, {})

def translate_captions(fin: str):
    """read from json file and translate captions into Swedish

    Args:
        fin (str): input file name
    """
    image_captions_sv=dict()
    with open(fin, 'r') as fp:
        image_captions = json.load(fp)
    image_ids=list(image_captions.keys())
    image_ids.sort()

    fout=op.splitext(fin)[0]+'_sv'+op.splitext(fin)[1]
    
    i=0
    all_i, all_captions=read_saved(fin='sample_data/image_captions_test2017_sv.json')
    print('initial starting point:', all_i)
    for image_id in tqdm(image_ids[all_i:]):
        if i>=500:
            all_i+=i
            i=0
            all_captions.update(image_captions_sv)
            image_captions_sv=dict()
            with open(fout, 'w', encoding='utf-8') as fp:
                json.dump(all_captions, fp, ensure_ascii=False)
            print('all_i=', all_i)

        captions=image_captions[image_id]
        # for caption in captions:
        caption=captions[0]
        cap=caption['caption']
        caption_id=caption['caption_id']
        image_captions_sv[caption_id]=translate(cap)

        i+=1
    
    # the last 500 items
    all_captions.update(image_captions_sv)
    image_captions_sv=dict()
    with open(fout, 'w', encoding='utf-8') as fp:
        json.dump(all_captions, fp, ensure_ascii=False)  

def main():
    # translate_objects('objects_vocab.txt')
    translate_captions(fin='sample_data/image_captions_test2017.json')

if __name__=='__main__':
    main()

'''
