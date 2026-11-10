"""Extract noun compounds from generated phrases; vocabulary remains image-derived."""
import re
import sys
from .common import canonicalize


def normalize_phrases(phrases,config):
    sys.path.insert(0,config['runtime']['nlp_packages'])
    import nltk
    from nltk.stem import WordNetLemmatizer
    nltk.data.path.insert(0,config['runtime']['nlp_data'])
    lemma=WordNetLemmatizer();result={}
    meaningful={'fire','water','light','exit','air','electric','electrical','tactile'}
    for phrase in sorted(set(phrases)):
        simple=canonicalize(phrase)
        if not simple:result[phrase]=None;continue
        if simple=='person':result[phrase]='person';continue
        tagged=nltk.pos_tag(re.findall(r'[a-z]+',simple))
        nouns=[i for i,(_,pos) in enumerate(tagged) if pos.startswith('NN')]
        if not nouns:result[phrase]=None;continue
        last=nouns[-1];start=last
        while start>0 and (tagged[start-1][1].startswith('NN') or tagged[start-1][0] in meaningful):start-=1
        words=[w for w,_ in tagged[start:last+1]]
        if words[-1] not in {'glasses','sunglasses','pants','jeans','scissors'}:
            words[-1]=lemma.lemmatize(words[-1],'n')
        result[phrase]=canonicalize(' '.join(words))
    return result
