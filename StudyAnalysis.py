#!/usr/bin/env python
# coding: utf-8

import xml.etree.ElementTree as ET
import random
import glob
import os
import io

import nltk
nltk.download('punkt_tab')

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.tag import pos_tag
from collections import defaultdict

import stanza
from stanza.resources.common import load_resources_json
from stanza.pipeline.core import DownloadMethod


import jieba
import jieba.posseg as pseg
import re

# load Simaligner
from simalign import SentenceAligner


# A global variable initialize once in the beginning
myaligner = ''

def InitializeMyaligner() :
    global myaligner 
    myaligner = SentenceAligner(model="bert", token_type="bpe", matching_methods='a')


def TranslogSessionProps(fn, stanzaFeats=False, Verbose = 0) :


    ### Root of the SessionProps XML output file
    SessionProps = ET.Element("SessionProps")
    
    if(not os.path.exists(fn)) :
        if(verbose): print(f"'{fn}' does not exists")
        return SessionProps
        
    ### Read Translog-XML file
    translog = ET.parse(fn)
    # Root of the Translog XML input file
    translog_root = translog.getroot()
    
    # get Source and Target Languages
    e = translog_root.find('.//Languages')
    SL = e.get('source') 
    TL = e.get('target')
    
    ##################################################################
    ### ST segmentation and tokenization
    # <SourceToken language="en" >
    #    <Token cur="0" tokId="1" pos="NNP" sntId="1" tok="Killer" />
      
    # Source text and target Text 
    SText = getSourceText(translog_root)
    if(SText == '') :
        print(f"\t{fn}: empty source text")
        return SessionProps
    
    # segment and tokenize the source text
    (STsnt, SToken) = Tokenize(SText, SL, stanzaFeats=stanzaFeats, Verbose=Verbose)
    
    # convert token Dictionary to xml
    SToken_root = list_of_dicts_to_xml(SToken, root_tag='SourceToken', item_tag='Tok')
    
    # assign source language 
    SToken_root.set('language', str(SL))
    
    # append ST tokenization to SessionProps root 
    SessionProps.append(SToken_root)
    
    ##################################################################
    ### TT final target text (translation) segmentation and tokenization
    #  <FinalToken language="ar" >
    #    <Token  cur="0" tokId="1" sntId="1" tok="ﺎﻠﻤﻣﺮﺿ" />  
    
    # get final text from Translog file 
    FText = getFinalText(translog_root)
    if(FText == '') : print(f"\t{fn}: empty final text")
    
    # segment and tokenize the target text
    (FTsnt, FToken) = Tokenize(FText, TL, stanzaFeats=stanzaFeats)

    # convert token Dictionary to xml
    FToken_root = list_of_dicts_to_xml(FToken, root_tag='FinalToken', item_tag='Tok')
    
    # assign target language 
    FToken_root.set('language', str(TL))

    # append FT tokenization to SessionProps root 
    SessionProps.append(FToken_root)
    
    ##################################################################
    ### Segment-alignment
    #  <SntAlignment>
    #     <Snt STsnt="1" FTsnt="1" />
    
    # very preliminary Sentence Alignment
    SntAlignList = sntAlignment(STsnt, FTsnt)
    
    # convert SntAlign Dictionary to xml
    SntAln_root = list_of_dicts_to_xml(SntAlignList, root_tag='SntAlignment', item_tag='Snt')
    
    # append SntAlign Dictionary to SessionProps root 
    SessionProps.append(SntAln_root)
    
    ##################################################################
    ### Token-alignment
    #  <TokAlign>
    #     <Tok STid="1" FTid="1" />

    # merge sentences in segment alignment goups
    segAlign = snt2segAlign(STsnt, FTsnt, SntAlignList)
    
    # random token alignment: not very useful
#    tokAlign = rndAlignment(segAlign)

    # use SimAlign to align tokens in segment alignment goups
    tokAlign = simAlignment(segAlign)

    # add sentence Ids to token Alignments
    addAlignIDs(tokAlign, SToken, FToken)
    
    # convert tokAlign Dictionary to xml
    TokAln_root = list_of_dicts_to_xml(tokAlign, root_tag='TokAlignment', item_tag='Tok')
    
    # append TokAln_root Dictionary to SessionProps root 
    SessionProps.append(TokAln_root)

    ##################################################################
    ### Keystroke-Token mapping
    #  <Modifications>
    #    <Mod time="72953" type="Mins" cur="0" chr="ﺍ" X="0" Y="0" sntId="1" sid="2" tid="1"  />

    # returns a dictionary of Keystrokes with tokIds
    Keys = KeyMapping(translog_root, FText, FToken,  Verbose = Verbose)
    if(Keys == {}) : 
        print(f"\t{fn}: no keystrokes recorded")
    
    # converts the keystrokes-dictionary into list of Modification
    Mods = Modifications(Keys, Verbose = 0)
    
    # add sentence Ids to token Modifications
#    addModsIDs(Mods, tokAlign)

    # map Modification to xml
    Mods_root = list_of_dicts_to_xml(Mods, root_tag="Modifications", item_tag='Mod')

    # append Modifications to SessionProps root 
    SessionProps.append(Mods_root)
     
    ##################################################################
    ### Fixation-Token mapping
    #  <Fixations>
    #      <Fix time="30" win="1" cur="227" dur="175" X="502" Y="228" Fsnt="3" STid="41" FTid="39" />

    Gaze = FixMapping(translog_root, tokAlign, Mods, SToken, FToken, Verbose=0)
    
    # converts the keystrokes-dictionary into list of Modification
    Fix = Fixations(Gaze)

    # map Modification to xml
    Fix_root = list_of_dicts_to_xml(Fix, root_tag="Fixations", item_tag='Fix')

    # append Modifications to SessionProps root 
    SessionProps.append(Fix_root)

    
    ##################################################################
    ### Segment open-closing
    #  <Segments>
    #    <Seg sntId="1" open="72952" close="89436" />

    return SessionProps


# add snt Ids to Token Alignments
def addAlignIDs(tokAlign, SToken, FToken):

    # Dictionaries to map tokId to sntId
    S = {}
    for s in SToken : S[str(s['tokId'])] = s
    T = {}
    for t in FToken : T[str(t['tokId'])] = t

    # look over list of Token Alignments 
    for aln in tokAlign :
        # get ST sentence Id for ST token ID
        for s in aln['STid'].split('+') :
            # 
            if(('STsnt' in aln) and (str(S[s]['sntId']) not in str(aln['STsnt']))) : 
                aln['STsnt'] = f"{aln['STsnt']}+{S[s]['sntId']}"
            else: aln['STsnt'] = str(S[s]['sntId'])
            del S[s]

        # get FT sentence Id for FT token ID
        for t in aln['FTid'].split('+'):
            if(('FTsnt' in aln)  and (str(T[t]['sntId']) not in str(aln['FTsnt']))): 
                aln['FTsnt'] = f"{aln['FTsnt']}+{T[t]['sntId']}"
            else: aln['FTsnt'] = str(T[t]['sntId'])
            del T[t]
    '''
    # add non-aligned words with their sentence ID
    for s in S :
        aln = {}
        aln['STid'] = s
        aln['STsnt'] = str(S[s]['sntId'])
        aln['FTid'] = aln['FTsnt'] = '0'
        tokAlign.append(aln)
    
    for t in T :
        aln = {}
        aln['STid'] = aln['STsnt'] = '0'
        aln['FTid'] = t
        aln['FTsnt'] = str(T[t]['sntId'])
        tokAlign.append(aln)
    '''
    
    return tokAlign

############################
# add snt Ids to Modifications
def addModsIDs(Mods, tokAlign):

    A = {}
    for aln in tokAlign : 
        for a in aln['FTid'].split('+') :
            A[int(a)] = aln

    print(A)
    # look over list of Token Alignments 
    for mod in Mods :
        fid = int(mod['FTid'])
        # not all FT words are aligned
        if(fid not in A) : 
            continue
        mod['STid'] = A[fid]['STid']
        mod['STsnt'] = A[fid]['STsnt']
        mod['FTsnt'] = A[fid]['FTsnt']



#####################################################
# ## Linguistic processing
# - sentence segmentation (NLTK)
# - tokenization (NLTK)
# - lexical features (Stanza)
# - cursor offset of words in text
#                     

def Tokenize(text, lng, form=1, stanzaFeats=True, Verbose = 0):
    """
    Tokenize and annotate text with linguistic features.
    
    Parameters:
    -----------
    text : str
        Raw input text to process
    lng : str
        Language code (e.g., 'en', 'es', 'de')
    form : int
        form=1: Output Format: [[('token', 'POS'), ...], ...]
        
    Returns:
    --------
    tuple: (snt, toksFeats)
        - snt: List of tokenized sentences with POS tags
        - toksFeats: List of token dictionaries with all features
    """
    
    # segment and tokenize source text 
    # snt: is list of tokenized ,Tagged sentences: 
    #    [[(token, pos), ...], [(token, pos), ... ], ...]  
    snt = segmentText(text, lng=lng, form=form)

    # create list of tokens with sntId, tokId, cursor offset
    #    [{tok1features}, {tok2features}, ...]
    toksList = tokenCurOffset(text, snt)

    # get additional features from Stanza to list of tokens
    # add features to list of STokens 
    if(stanzaFeats) :
        tokens = stanzaFeatures(snt, lng, toksList, Verbose = Verbose)
    else: tokens = toksList
    
    return (snt, tokens)


#########################################################################
# segment text FT from the Translog file
def segmentText(text, lng='en', form = 1):

    '''
    form 0: Output Format: [['token', ...], ...] : Just tokenized sentences
    form 1: Output Format: [[('token', 'POS'), ...], ...]: With POS tags (default)
    form 2: Output Format: ['token token ...', ...]: Sentences as strings
    form 4: Uses Stanza for all languages: Use Case: High-quality annotations
    '''
    
    # replace multiple \n by one (no impact on NLTK segmentation)
    text1 = re.sub(r'\n+', '\n', text)
    
    if(form == 4) : return segmentStanza(text, lng)
    if(lng == 'ja') : return segmentStanza(text, lng)
    if(lng == 'zh') : return segmentChineseJieba(text, form)
        
    # Segment text into list of sentences
    snt0 = sent_tokenize(text1)

    # segment text at newline into segments (not covered by NLTK)
    snt1 = []
    for i in range(len(snt0)) :
        s = snt0[i]
        snt1.extend(s.split('\n'))
        
    # Tokenize each sentences
    snt1 = [word_tokenize(s) for s in snt1]
    
    # remove empty sentences (e.g. produced by \n)
    if([] in snt1): snt1.remove([])

    # Part-of-speech tagging each sentences: works only properly for English
    if(form == 1) : snt1 = [pos_tag(s) for s in snt1]

    # collapse back into list of sentences
    if(form == 2) : snt1 = [" ".join(s) for s in snt1]
        
    return snt1


def segmentStanza(text, lng) :
#    pattern = r'[。！？]'

    nlp = stanza.Pipeline(lang=lng, processors='tokenize,pos')

    doc = nlp(text)
    
    # stanza document to list of list of dictionaries
    stza_list = doc.to_dict()   
    
    L = []
    for snt in stza_list:
        T = []
        for tok in snt :
            T.append((tok['text'], tok['upos']))
        if(len(T) > 0):  L.append(T)
    return L

def segmentChineseJieba(text, form) :
    
    words = pseg.cut(text)
    return words2snt(list(words), form)

def words2snt(words, form) :

    pattern = r'[。！？.!?]'
    
    S = []
    L = []
    for tok, pos in words:
        # skip word that are whitespaces
        if(re.search(r'\s',  tok)) : 
            # new sentence with \n,but it's not a token
            if(re.search(r'\n',  tok)) : 
                if(len(L) > 0) : S.append(L)
                L = []
            continue
            
        # token and pos
        if(form == 1) : L.append((tok, pos))
        # only token if form != 1
        else : L.append(tok)

        # end of sentence
        match = re.search(pattern, tok)
        if(match) : 
            if(len(L) > 0) : S.append(L)
            L = []

    if(len(L) > 0) : S.append(L)

    return S


# additional features from Stanza
def stanzaFeatures(snt, lng, token, tokenize_no_ssplit=False, Verbose=False):

    # check which processors are available for language
    PocessorList=['tokenize','pos','lemma','mwt','ner', 'depparse']

    # 'zh-hans' for loading processors 
    if(lng == 'zh') : lng = 'zh-hans'

    resources = load_resources_json()
    processor = ''
    for p in PocessorList:
        # Check for supported processors in a specific language, e.g., Portuguese ('pt')
        if p in resources[lng]:
            if(processor == '') :  processor = p
            else: processor += f",{p}"
    
    if(Verbose) : print(f"\t{lng} processors:{processor}")
    
    # initialize stanza pipeline
    nlp = stanza.Pipeline(lang=lng, processors=processor, tokenize_pretokenized=True, 
                          tokenize_no_ssplit=False, download_method=DownloadMethod.REUSE_RESOURCES, verbose=False)

    # keep only token from the list of sentences
    # there can be empty tokens '' which are substituted by '.'
    sntList = [[w if w != '' else '.' for w, p in s] for s in snt]

    doc = nlp(sntList)

    # stanza document to list of list of dictionaries
    stza_list = doc.to_dict()
    
    # map list of NLTK tokens into dictinary for faster lookup 
    TD = {d['tokId']: d for d in token}

    sntId = 0
    tokId = 0
    Token = []
    off = 0

    for snt in stza_list:
        sntId +=1
        for tok in snt :
            tokId +=1
            tok['sntId'] = sntId
            tok['tokId'] = tokId
            
            # these features must be identical
            if(tok['text'] != TD[tokId]['tok']) :
                print(f"stanzaFeatures Warning: snt:{sntId} tokId:{tokId} stanzaWord:>{tok['text']}< NLTKWord:>{TD[tokId]['tok']}<")
                      
            # copy from tokList
            tok['cur'] = TD[tokId]['cur']
            tok['tok'] = TD[tokId]['tok']
            
            if('space' in TD[tokId]):  tok['space'] = TD[tokId]['space']
            else:  tok['space'] = ''

            # pos tag from NLTK
            if('pos' in TD[tokId]): tok['pos'] = TD[tokId]['pos']

            # delete Stanza features
            if('text' in tok): tok.pop('text')
            if('misc' in tok): tok.pop("misc")
            if('id' in tok): tok.pop("id")
            if('start_char' in tok): tok.pop("start_char")
            if('end_char' in tok): tok.pop("end_char")
            Token.append(tok)
    return Token

################################################################
# Find cursor offset for tokens in text
def tokenCurOffset(text, snt): 
    
    L = [] # list of dictionaries the contain Token information
    end = 0 # position of end of previous word in text
    tokId = 0  # word ID
    sntId = 0  # sentence ID

    for s in snt:
        sntId += 1
        #for tok, pos in s:
        for i in range(len(s)):
            tok, pos = s[i]
            start = text[end:].find(tok)
            space = text[end:end+start]
            cur = end+start 
            tokId += 1
            H = {'tokId': tokId, 
                 'sntId' : sntId, 
                 'cur': end+start,
                 'tok' : tok, 
                 'space' :space, 
                 'pos' : pos
                }
            # memorize (tok, tokId)
            s[i] = (tok, tokId)
            
            L.append(H)
#            print(f"id:{tokId} cur:{cur}\t{tok:<20}\tend0:{end} space:{start}>{space}< {pos}")
    
            end += start + len(tok) 
    return L

#########################################################################
# get ST from the Translog file
def getSourceText(root):
    
    # get text from UTF8 container in the xml file 
    ST = root.find('.//SourceTextUTF8')
    if ST is not None:
        if(ST.text): return ST.text
        return ''
        
    # in older versions there is no UTF8 version in the xml file 
    # else SourceTextChar must extist
    text2 = ''
    STchars = root.findall('.//SourceTextChar/CharPos')
    for chars in STchars:
        text2 += chars.get('Value')
    return text2

# get FT from the Translog file
def getFinalText(root):

    # FinalText in UTF8 should usually always be there
    FT = root.find('.//FinalText')
    if FT is not None:
        if(FT.text): return FT.text
        return ''
        
    # else FinalTextChar must extist
    text2 = ''
    FTchars = root.findall('.//FinalTextChar/CharPos')
    for chars in FTchars:
        text2 += chars.get('Value')
    return text2

# get initial translation from the Translog file 
def getTranslation(root):

    # FinalText in UTF8 should usually always be there
    FT = root.find('.//TranslationUTF8')
    if FT is not None:
        if(FT.text): return FT.text
        return ''
        
    # or TargetTextUTF8 
    FT = root.find('.//TargetTextUTF8')
    if FT is not None:
        if(FT.text): return FT.text
        return ''
        
    # else FinalTextChar must extist
    text2 = ''
    FTchars = root.findall('.//TranslationChar/CharPos')
    for chars in FTchars:
        text2 += chars.get('Value')
    if(len(text2) > 0) : return text2

    FTchars = root.findall('.//TargetTextChar/CharPos')
    for chars in FTchars:
        text2 += chars.get('Value')

    return text2

#####################################################
# ## Segment / Word Alignment
# - segment: sentence by sentence
# - word alignment
# - merged groups 


# build alignment segments with m *n sentence alignment groups 
def snt2segAlign(STnt, FTnt, SntAlign):

    """
    Convert sentence alignments to token-level aligned segments.
    
    Parameters:
    -----------
    STnt : list of lists of Source Text (tokens , tokIds)
        Source sentences: [[(token, id), (token, id), ...], ...]
        Example: [[('Killer', 1), ('nurse', 2), ...], ...]
    
    FTnt : list of lists of Target Text (tokens , tokIds)
        Target sentences: [[(token, id), (token, id), ...], ...]
        Example: [[('El', 1), ('enfermero', 2), ...], ...]
    
    SntAlign : list of aligned sentence ids per segment (list of dicts)
        Sentence alignments: [{'STsnt': '1', 'FTsnt': '1'}, {'STsnt': '2+3', 'FTsnt': '2+3'}, ...]
        where: 
        'STsnt': source sentence id
        'FTsnt': target sentence id
    
    Returns:
    --------
    SEGS : dict of alignment segments
        Alignment groups with tokens and IDs
    """
    
    s = len(STnt)
    t = len(FTnt)

    # list of tokens per sentence
    STok = [[t for t, i in s] for s in STnt]
    FTok = [[t for t, i in s] for s in FTnt]
    # list of token ids per sentence
    STid = [[i for t, i in s] for s in STnt]
    FTid = [[i for t, i in s] for s in FTnt]

    SEGS = {}
    seg = 0
    # loop over aligned sentence ids
    for aln in SntAlign:
        # ST sentences 
        sIds = [int(s)-1 for s in aln['STsnt'].split('+')]
        # aligned TT sentences
        tIds = [int(s)-1 for s in aln['FTsnt'].split('+')]
        
        SEGS.setdefault(seg, {})
        SEGS[seg]['STok'] = []
        SEGS[seg]['FTok'] = []
        SEGS[seg]['STid'] = []
        SEGS[seg]['FTid'] = []

        # join ST/TT tokens of a segment
        for i in sIds : SEGS[seg]['STok'].extend(STok[i])
        for i in tIds : SEGS[seg]['FTok'].extend(FTok[i])
        # join ST/TT token ids of a segment
        for i in sIds : SEGS[seg]['STid'].extend(STid[i])
        for i in tIds : SEGS[seg]['FTid'].extend(FTid[i])        
        seg += 1
        
    return SEGS

# random word alignmet per bilingual segment 
def rndAlignment(SEGS):

    # random word alignment
    L = []
    for seg in SEGS:
        SEGS[seg]['aln'] = []

        for i in range(int((len(SEGS[seg]['STtok']) / 1.5))) :
            # Get a random index from the list
            rs = random.randint(0, len(SEGS[seg]['STtok']) - 1)
            rt = random.randint(0, len(SEGS[seg]['FTtok']) - 1)

            L.append({'STid' : SEGS[seg]['STsnt'][rs], 'FTid':SEGS[seg]['tid'][rt]})

    M = merge_alignments_graph(L, 'STid', 'FTid')
                
    return M

# simAlign word alignmet per bilingual segment 
def simAlignment(SEGS):
    
    aln = []
    for seg in SEGS :
        # myaligner must be initialized globally
        # returns a dictionary of aligned indexes {key: [(s,t), (s,t), ...]}
        A = myaligner.get_word_aligns(SEGS[seg]['STok'], SEGS[seg]['FTok'])

        # map simalign segment-relative indexes into TPR-DB text-relative indexes
        for m in A:
            for s, t in A[m]:
                aln.append({'STid' : SEGS[seg]['STid'][s], 'FTid': SEGS[seg]['FTid'][t]})
                
    return merge_alignments_graph(aln, 'STid', 'FTid')
                

def sntAlignment(STnt, FTnt):
    s = len(STnt)
    t = len(FTnt)
    
    L = []
    for i in range(min(s, t)):
        L.append({'STsnt': i+1, 'FTsnt': i+1})
    
    if(s > t) :
        for i in range(t, s): 
            L.append({'STsnt': i+1, 'FTsnt': t})
        
    if(t > s) :
        for i in range(s, t): 
            L.append({'STsnt': s, 'FTsnt': i+1})

    # bring into a grouped format
    M = merge_alignments_graph(L, 'STsnt','FTsnt') 
    return M


def merge_alignments_graph(alignments, src='src', tgt='tgt'):
    """
    Use graph-based approach to find connected components.
    Alignments that share src or tgt indices are in the same group.
    """
    
    if not alignments:
        return []
    
    # Build graph of connections
    graph = defaultdict(set)
    
    for i, align in enumerate(alignments):
        graph[i].add(i)
    
    # Connect alignments that share indices
    for i in range(len(alignments)):
        for j in range(i + 1, len(alignments)):
            if (alignments[i][src] == alignments[j][src] or
                alignments[i][tgt] == alignments[j][tgt]):
                graph[i].add(j)
                graph[j].add(i)
    
    # Find connected components
    visited = set()
    components = []
    
    def dfs(node, component):
        if node in visited:
            return
        visited.add(node)
        component.add(node)
        for neighbor in graph[node]:
            dfs(neighbor, component)
    
    for i in range(len(alignments)):
        if i not in visited:
            component = set()
            dfs(i, component)
            components.append(component)
    
    # Build merged results
    merged = []
    for component in components:
        src_indices = set()
        tgt_indices = set()
        for idx in component:
            src_indices.add(alignments[idx][src])
            tgt_indices.add(alignments[idx][tgt])
        
        merged.append({
            src: sorted(src_indices),
            tgt: sorted(tgt_indices)
        })
    
    # Sort by first src index
    merged.sort(key=lambda x: int(x[src][0]))

    M = []
    for item in merged:
        src_str = '+'.join(map(str, item[src])) if len(item[src]) > 1 else str(item[src][0])
        tgt_str = '+'.join(map(str, item[tgt])) if len(item[tgt]) > 1 else str(item[tgt][0])
        M.append({src: src_str, tgt: tgt_str})
    
    return M


#####################################################
# ## Keystroke mapping
# - Keystrokes -> Modifications


def KeyMapping(translog_root, Text, Token, Verbose = 1):

    text = list(Text)   # list of characers for final text 
    index = [1] * len(text)     # initialize list of token indexed (FTid)

    pm = 15 # margin for character visualization, left and right

    # test whether tokens fit text, print warning
    for tok in Token:
        cur =  tok['cur']
        if(not Text.startswith(tok['tok'],  cur)):
            t1 = ''.join(text[max(cur-pm, 0): cur]).replace('\n', '\\n')
            t2 = ''.join(text[cur:cur+len(tok['tok'])]).replace('\n', '\\n')
            t3 = ''.join(text[cur+len(tok['tok'])+1: min(cur+pm,len(text))]).replace('\n', '\\n')
            print(f"\tWARNING cur:{tok['cur']} token:{tok['tokId']}:>{tok['tok']}< ~~ {t1}>{t2}<{t3}")      
  
    # character arrays of final text
    for tok in Token:
        cur = tok['cur']
        end = cur + len(tok['tok'])
        
        # space before is part of the following token 
        if('space' in tok) : cur -= len(tok['space'])
        index[cur:end] = [tok['tokId']]*(end-cur)
    
    # characters at the end of the text get tokId of last word
    index[end:] = [Token[-1]['tokId']] * len(index[end:])

    # get keystrokes from the xml file
    # map keystrokes into dictionary 
    Keys = {}
    time = 0
    e = translog_root.find('.//Events')
    for key in e.findall('Key'):
        d = dict(key.attrib)

        # there may be no keystrokes 
        if(d == {}) : return Keys
            
        d['Cursor'] = int(d['Cursor'])
        d['Time'] = int(d['Time'])
        
        time = d['Time']
        
        # two keystrokes at the same time
        while(time in Keys) :
            if(Verbose) : 
                print(f"\tKeystrokes same time {time}\t{d['Type']}:{d['Value']} -- {Keys[d["Time"]]['Type']}:{Keys[d["Time"]]['Value']}")
            time +=1
        Keys[time] = d

    # insert the 'cut' string at position 'cur' in text 
    # insert a tokId in index for inserted cut string
    def DeleteString(cur, cut, time=time):

        if((cur < len(index)) and (len(index) > 0)) : 
            tokId = index[cur]      
            Keys[time]['tokId'] = tokId
            text[cur:cur] = cut
            index[cur:cur] = [tokId] * len(cut)
        
        else :
            # cursor is > index
            if (len(index) == 0) :  tokId = 1
            else:  tokId = index[-1]
                
            text.extend(cut)
            index.extend([tokId] * len(cut))
            
        Keys[time]['tokId'] = tokId

    # delete the 'cut' string at position from text 
    # insert a tokId in index for inserted cut string
    def InsertString(cur, ins, time=time):
        w = ''
        if(cur >= len(index)) :
            w = f"\tWARNING1 InsertString:{time} cur:{cur} length text {len(text)}"
            return w
            
        Keys[time]['tokId'] = index[cur]

        t1 = ''.join(ins).replace('\n', '\\n')
        t2 = ''.join(text[cur:cur+len(ins)]).replace('\n', '\\n')
        if((t1 != t2) and ('#' not in t1) and ('#' not in t2) ) :
            txt = f"\t{''.join(text[cur-pm:cur]).replace('\n', '\\n')}>{t2}<{''.join(text[cur+len(ins)+1:cur+len(ins)+pm+1]).replace('\n', '\\n')}"
            w = f"\tWARNING2 InsertString:{time} cur:{cur} ins:>{t1}<\t{txt}"

        # delete the string from the arrays
        del text[cur : cur + len(ins)]
        del index[cur : cur + len(ins)]
        
        return w

    # count non-matching insertion warnings 
    Warn = 0

    # accumulate keystroke data for IME 
    IME = {'value' : '', 'text': '', 'count': 0, 'end': 0, 'tpe': ''}

    ############################################################
    # main loop over keystrokes in reversed time
    for time in  sorted(Keys.keys(), reverse=True) :

        tpe = Keys[time]['Type']    # one of 'insert', 'delete', 'edit'

        # navigation not interesting for modifications
        if(tpe == 'navi') : continue
            
        # study ATJA22 has no Value feature for deletions
        if('Value' not in Keys[time]) :
            key = Keys[time]['Value'] ='[Delete]'
        else :    
            # Ctrl+C does not change the text
            if('[Ctrl+C]' in Keys[time]['Value']) : continue        
            val = Keys[time]['Value']   # the value of the keystroke
            

        # collect warning string from 
        warning = ''

        cur = Keys[time]['Cursor']  # cursor offeset in text
        
        # initialize tokId
        if(len(index) > 0) :
            if(cur >= len(index)) : 
                Keys[time]['tokId'] = index[-1]
            else: Keys[time]['tokId'] = index[cur]                    
        else : Keys[time]['tokId'] = 1       

        # plot text changes every line before
        if(Verbose > 2) :
            # marked text 
            cut = ''
            if('Text' in Keys[time]) : cut = Keys[time]['Text']
            if('Paste' in Keys[time]) : cut = Keys[time]['Paste']
            print(f"KeyMapping loop: {time}\t{tpe} i:{val.replace('\n', '\\n')}< d:{cut.replace('\n', '\\n')}< \tc:{cur}\t{''.join(text[max(cur-pm, 0):min(cur+pm,len(text))]).replace('\n', '\\n')}", end='')


        ###############################################
        # Chinese / Japanese IME input
        if(tpe == 'IME') :
            val1 = val[1:-1]
            if(len(val) <= 0) : print("IME: invalid Value", time, Keys[time])

            # accummulate sequence of keystrokes
            IME['value'] = val1 + IME['value']
            # count no of keystrokes
            IME['count'] += 1

            # the sequence of keystrokes == the IMEtext
            if(IME['value'] == IME['text']) :
                end = IME['end']
                if(IME['tpe'] == 'delete') :
                    DeleteString(Keys[end]['Cursor'], list(Keys[end]['Text']), time = end)
                elif(IME['tpe'] == 'insert') :  
                    warning = InsertString(Keys[end]['Cursor'], list(Keys[end]['Value']) , time = end)
                elif(IME['tpe'] == 'return') :
                    warning = InsertString(Keys[end]['Cursor'], list('\n') , time = end)

                else :  print("Warning1 IME type :", end, IME['tpe'],  Keys[end])

                Keys[end]['Dur'] = end - time
                Keys[end]['Strokes'] = IME['count']
                
                IME['value'] = ''
                IME['count'] = 0
                   
        # insert the IME text in the editor
        elif('IMEtext' in Keys[time]) :          
            if((IME['value'] != '') and (IME['text'] != '')) : 
                end = IME['end']
                if(IME['tpe'] == 'delete') :  
                    if(len(IME['text']) != len(IME['value'])) :
                        print(f"\tWarning3 IMEtext {end} does not match Text:>{IME['text']}< IME>{IME['value']}<")
                    DeleteString(Keys[end]['Cursor'], list(Keys[end]['Text']), time = end)
                    
                elif(IME['tpe'] == 'insert') :
                    if(len(IME['text']) != len(IME['value'])) :
                        print(f"\tWarning4 IMEtext {end} does not match Text:>{IME['text']}< IME>{IME['value']}<")
                    warning = InsertString(Keys[end]['Cursor'], list(Keys[end]['Value']) , time = end)
                    
                elif(IME['tpe'] == 'return') :
                    if(len(IME['text']) != len(IME['value'])) :
                        print(f"\tWarning5 IMEtext {end} does not match Text:>{IME['text']}< IME>{IME['value']}<")
                    warning = InsertString(Keys[end]['Cursor'], list('\n') , time = end)
                    
                else :  print("Warning2 IME type:", end, IME)
                    
                Keys[end]['Dur'] = end - time
                Keys[end]['Strokes'] = IME['count']

            IME['text'] = Keys[time]['IMEtext'] 
            IME['end'] = time
            IME['tpe'] = tpe 
            IME['value'] = ''
            IME['count'] = 0                        

        #################################################################
        # [Ctrl+V] and [Ctrl+X]
        elif(tpe == 'edit') :

            # insertion: [Ctrl+V]
            if('Paste' in Keys[time]) : 
                ins = list(Keys[time]['Paste'])
                warning = InsertString(cur, ins, time=time)
                            
            # deletion: [Ctrl+X]
            if('Text' in Keys[time]) : 
                cut = (list(Keys[time]['Text']))
                DeleteString(cur+1, cut)        

        #################################################################
        elif(tpe == 'insert' or tpe == 'return') :
            cut = ''

            # marked text that is deleted with the insertion
            if('Text' in Keys[time]) : 
                cut = (list(Keys[time]['Text']))
                DeleteString(cur+1, cut, time=time)

            # insert keystroke
            if(tpe == 'return') : val ="\n"
                
            # insert a blank keystroke
            if(val == '' and cut == '') : 
                val = ' '
                print(f"\tInsert with no value: {cur}")
                
            # insert the string
            warning = InsertString(cur, list(val), time=time)
    
        #################################################################
        elif(tpe == 'delete') :
            if('Text' in Keys[time]) : cut = (list(Keys[time]['Text']))
            else: 
                cut = list('#')
                Keys[time]['Text'] = '#'
                if(Verbose) : print(f"\tKeyMapping:{time} cur:{cur} insert {val} text: #")

            # could be: [Back], [Ctrl+Back], [Shift+Back]
            if('Back' in val) :
                cutId = 1
                if(cur >= len(index)) :
                    if(len(index) > 0) : cutId = index[-1]                
                    Keys[time]['tokId'] = cutId
                    text.extend(cut)
                    index.extend([cutId] * len(cut))
                    if(Keys[time]['tokId'] == 0) : print("DDD1", cur)


                elif(cur > 0) : 
                    cutId = index[cur-1]                
                    Keys[time]['tokId'] = index[cur-1]
                    text[cur:cur] = cut
                    index[cur:cur] = [cutId] * len(cut)

                    if(Keys[time]['tokId'] == 0) : print("DDD2", cur)

                # cursor at first position. index / text might be empty
                else : 
                    Keys[time]['tokId'] = 1 
                    
                    # insert deletion in the first position
                    text = cut + text
                    index = [cutId] * len(cut) + index
                    
                    if(Keys[time]['tokId'] == 0) : print("DDD")

            elif('Delete' in val): DeleteString(cur, cut, time=time)
                
            # a different version in early Translog from 2013
            elif('delete' in val): 
                Keys[time]['Value']  = '[Delete]'
                DeleteString(cur, cut, time=time)
                
            else:  print(f"Delete key not covered: {val}")
                
        elif(tpe == 'speech') :
            warning = InsertString(cur, list(Keys[time]['Value']))
            
        else:  print(f"Type not covered: {Keys[time]}")

        ##############
        # plot text changes every line before

        if(Verbose > 2) :
            print(f"\t\t{''.join(text[max(cur-pm, 0): cur]).replace('\n', '\\n')}@{''.join(text[cur: min(cur+pm,len(text))]).replace('\n', '\\n')}\tToken:{Keys[time]['tokId']}\tlen:{len(text)}-{len(index)}")
       
        if(Verbose) :
            if(warning): print(warning)

        # count non-matching insertion warnings 
        if(len(warning) > 1) : Warn += 1
        
    if(Verbose) : print(f"\tRemaining Text length: {len(index)}")
    if(Warn > 0) : print(f"\tKeyMapping Warnings:{Warn}")

    ##################################################################
    # copy FT sentence Id into Keys
    T = {}
    for t in Token: 
        T[t['tokId']] = t['sntId']
    
    for k in Keys:
        # could be navigation keystrokes
        if('tokId' not in Keys[k]) : continue
        
        # this should not be the case
        if(Keys[k]['tokId'] not in T) : 
            print("No tokId in Token", Keys[k])
            Keys[k]['sntId'] = 0
            continue
        Keys[k]['sntId'] = T[Keys[k]['tokId']]
        
    return Keys

     
###########################################################################################

def Modifications(Keys, Verbose = 0) :
    
    Modifs = []
    for time in sorted(Keys):

        # navigation, and [Ctrl+C] not interesting for modifications
        if(Keys[time]['Type'] == 'navi') : continue 
        # IME keystrokes : already accumulated in I
        if(Keys[time]['Type'] == 'IME') : continue 
        if('[Ctrl+C]' in Keys[time]['Value']) : continue
            

        Mod = {}
        Mod['time'] = time
        Mod['cur'] = Keys[time]['Cursor']
        Mod['FTsnt'] = Keys[time]['sntId']
        if('Dur' in  Keys[time]): Mod['dur'] = Keys[time]['Dur']
        if('Strokes' in  Keys[time]): Mod['strokes'] = Keys[time]['Strokes']

        if('tokId' in Keys[time]) : Mod['FTid'] = Keys[time]['tokId']
        else: print(f"Modifications: tokId :{Keys[time]}")

        if(Keys[time]['Type'] == 'edit') :
            if('Ctrl+X' in Keys[time]['Value']) :
                Mod['type'] = 'Mdel'
                if('Text' not in Keys[time]) :
                    Mod['chr'] = '#'
                    if(Verbose) : print(f"\tModifications: edit added Text: '#'\t{Keys[time]}")
                else : Mod['chr'] = Keys[time]['Text'] 

            elif('Ctrl+V' in Keys[time]['Value']) :
                Mod['type'] = 'Mins'
                if('Paste' not in Keys[time]) :
                    Mod['chr'] = '#'
                    if(Verbose) : print(f"Modifications: edit {Keys[time]}")                    
                else : Mod['chr'] = Keys[time]['Paste'] 

            else :
                print(f"\tEdit uncovered:{Keys[time]}")

            
        elif(Keys[time]['Type'] == 'return') :
            Mod['type'] = 'Mins'
            Mod['chr'] = '\n'
            
        elif(Keys[time]['Type'] == 'insert') :

            # insert deletion in insertion
            if('Text' in Keys[time]) :
                # allocate new Mod
                Mod1 = {}
#                time1 = time - 1
                Mod1['time'] = time - 1
                Mod1['cur'] = Keys[time]['Cursor'] #+ len(Keys[time]['Text']) - 1
                Mod1['FTid'] = Keys[time]['tokId']
                Mod1['FTsnt'] = Keys[time]['sntId']
                Mod1['type'] = 'Mdel'
                Mod1['chr'] = Keys[time]['Text']
                Modifs.append(Mod1)
#                print(Mod1)

            if('Dur' in Keys[time]) : Mod['dur'] = Keys[time]['Dur']
            if('Strokes' in Keys[time]) : Mod['strokes'] = Keys[time]['Strokes']
                
            Mod['type'] = 'Mins'
            Mod['chr'] = Keys[time]['Value']
            
        elif(Keys[time]['Type'] == 'delete') :
            Mod['type'] = 'Mdel'
            Mod['chr'] = Keys[time]['Text'] 
            
            if('Back' in Keys[time]['Value']) :
                Mod['cur'] = Keys[time]['Cursor']
#                Mod['cur'] += len(Keys[time]['Text'])-1
#                print(f"Back subtract:{len(Keys[time]['Text'])-1}")

            elif('Delete' in Keys[time]['Value']) :                
                Mod['cur'] = Keys[time]['Cursor']
            else :
                print(f"delete: uncovered:{Keys[time]}")
                
        elif(Keys[time]['Type'] == 'speech') :
            Mod['type'] = 'Sins'
            Mod['chr'] = Keys[time]['Value'] 
                              
        else:
            print(f"\tModifications: uncovered:{Keys[time]}")
        Modifs.append(Mod)

    return Modifs


#####################################################
# ## Fixation mapping

def Fixations(Fixes, Verbose=0) :

    L = []
    for fix in Fixes :
        F = {}
        if(Verbose) : print(fix)
        F['time'] = fix['Time']
        F['win'] = fix['Win']
        F['cur'] = fix['Cursor']
        F['dur'] = fix['Dur']
        
        if('X' in fix) : F['X'] = fix['X']
        else:  
            F['X'] = 0
            print("\tFixation without X:", fix)
        if('Y' in fix) : F['Y'] = fix['Y']
        else: 
            F['Y'] = 0
            print("\tFixation without Y", fix['Time'])
            
        if('FTid' in fix) : F['FTid'] = fix['FTid']
        else : F['FTid'] = 0
        if('STid' in fix) : F['STid'] = fix['STid']
        else : F['STid'] = 0
#        F['STid'] = fix['STid']
#        F['FTsnt'] = fix['FTsnt']
#        F['STsnt'] = fix['STsnt']
        L.append(F)
        
    return L

def FixMapping(translog_root, tokAlign, Mods, SToken, FToken, Verbose=0) :


    #######################################################
    # get the TT fixation at a time and cur position
    # get fixations from the xml file
    Fixes = [] # store fixations in a list
    
    e = translog_root.find('.//Events')
    for fix in e.findall('Fix'):
        d = dict(fix.attrib)
        
        # there may be no fixations 
        if(d == {}) : continue 
        # skip the beginning of the fixations 
        if('Dur' not in d): continue
        # skip fixations not in ST or TT window             
        if(d['Win'] == '0') : continue 

        if('Cursor' in d) : d['Cursor'] = int(d['Cursor'])
        else : d['Cursor'] = 0
            
        d['Time'] = int(d['Time'])
        d['Win'] = int(d['Win'])
        Fixes.append(d)

    #######################################################
    # ST fixations: map ST cur offet to ST Token ID
    STcur2TokId = {}
    for tok in SToken:
        cur = tok['cur']
        end = cur + len(tok['tok'])
        
        # space before is part of the following token 
        if('space' in tok) : cur -= len(tok['space'])
        for i in range(cur, end):
            STcur2TokId[i] = tok['tokId']
    
    #######################################################
    # map list of alignments into dictionaries for faster retrieval
    STid2Aln = {}
    FTid2Aln = {}
    for aln in tokAlign :
        for a in aln['STid'].split('+') :
            STid2Aln[int(a)] = aln
        for a in aln['FTid'].split('+') :
            FTid2Aln[int(a)] = aln
        
    ##########################################################
    # getinitial Translation from Translog file 
    Text = list(getFinalText(translog_root))
    FTid = [1] * len(Text)
    
    # character arrays of final text
    for tok in FToken:
        cur = tok['cur']
        end = cur + len(tok['tok'])
        
        # space before is part of the following token 
        if('space' in tok) : cur -= len(tok['space'])
        FTid[cur:end] = [tok['tokId']]*(end-cur)
    
    # characters at the end of the text get FTid of last word
    FTid[end:] = [FToken[-1]['tokId']] * len(FTid[end:])
    
    P = {'T': Text, 'I' : FTid, 'Mods': Mods, 'Midx' : len(Mods)-1, 'Warn' : 0}

    ##########################################################
    # main loop over reversed fixation order
    for d in list(reversed(Fixes)):
        # fixation on ST
        if(d['Win'] == 1):
            if(d['Cursor'] in STcur2TokId) : tokId = STcur2TokId[d['Cursor']]
            else: tokId = SToken[-1]['tokId']
                
            d['STid'] = tokId
            if(tokId in STid2Aln) :
                d['FTid'] = STid2Aln[tokId]['FTid']
#                d['FTsnt'] = STid2Aln[tokId]['FTsnt']
#                d['STsnt'] = STid2Aln[tokId]['STsnt']
            else: d['FTid'] = 0
                
            if(Verbose > 2) : print("\tSTcur2TokId:", d['Cursor'], tokId)
            
        # fixation on TT
        if(d['Win'] == 2):
            tokId = ReverseGenerateMods(P, d, Verbose=Verbose)
            d['FTid'] = tokId
            if(tokId in FTid2Aln) :
                d['STid'] = FTid2Aln[tokId]['STid']
#                d['FTsnt'] = FTid2Aln[tokId]['FTsnt']
#                d['STsnt'] = FTid2Aln[tokId]['STsnt']
            else: d['STid'] = 0
                
            if(Verbose > 1): print(f"ReverseGenerateMods Fix:{d['Time']} dur:{d['Dur']}\tcur:{d['Cursor']} id:{tokId}")

    if(P['Warn'] > 0) : print(f"\tFixMapping Warnings:{P['Warn']}")

    return Fixes


# alternative Version
def ReverseGenerateMods(P, d, Verbose = 0) :

    fix_time = int(d['Time']) 
    fix_cur = int(d['Cursor'])
    fix_dur = int(d['Dur'])

    # no keystrokes or at the beginning of Keystrokes
    if(P['Midx'] < 0) :
        # there is a target text before keystrokes
        if(len(P['I']) > 0) :
            if(fix_cur >= len(P['I'])) : return P['I'][-1]
            else: return P['I'][fix_cur]
        # the string is empty
        return 1

    while (P['Midx'] >= 0) :
        
        d = P['Mods'][P['Midx']]
#        print(f"RGM Mod:{d['time']} Fix:{fix_time}-{fix_time + fix_dur} dur:{fix_dur} fix_cur:{fix_cur} diff:{fix_time - d['time']} {fix_time + fix_dur - d['time']}\tidx:{P['Midx']}")
        
        if(d['time'] <= fix_time + fix_dur):
#            print(f"Return: len:{len(P['I'])} fix_cur:{fix_cur}\t{P['I'][fix_cur-2:fix_cur+2]}\t{P['I'][-1:]}")  

            # string is 
            if(len(P['I']) > 0) :
                if(fix_cur >= len(P['I'])) : return P['I'][-1]
                else: return P['I'][fix_cur]
            # the string is empty
            return 1
            
        # go backwards through Modifications
        P['Midx'] -= 1
        
        mod_cur = int(d['cur'])
            
        # plot every line
        if(Verbose > 1) :
            print(f"{d['type']} {d['time']} cur:{d['cur']} chr:{d['chr']:<10}\t", end ='')  
            print(f"\t{''.join(P['T'][mod_cur-20:mod_cur]).replace('\n', '\\n')}", end = '')
            print(f">{''.join(P['T'][mod_cur:mod_cur+len(d['chr'])]).replace('\n', '\\n')}<", end='')
            print(f"{''.join(P['T'][mod_cur+len(d['chr']):mod_cur+len(d['chr'])+30]).replace('\n', '\\n')}", end = '')
            print(f"\tlen:{len(P['I'])}")
#            print(f"\tI:{mod_cur-3}-{mod_cur+3}\t{P['I'][mod_cur-3:mod_cur+3]}")  
            
        if(d['type'] == 'Mdel') :
            P['T'][mod_cur:mod_cur] = list(d['chr'])
            P['I'][mod_cur:mod_cur] = [d['FTid']] * len(d['chr'])
        if(d['type'] == 'Mins') :
            txt1 = ''.join(P['T'][mod_cur:mod_cur+len(d['chr'])]).replace('\n', '\\n')
            txt2 = ''.join(d['chr']).replace('\n', '\\n')
            
                # check whether deletions match text
            if(mod_cur + len(d['chr']) > len(P['T'])) : 
                P['Warn'] += 1
                if(Verbose) :
                    print(f"\tDeletion cur:{d['cur']} >{d['chr']}< beyond text len:{len(P['T'])}\t{''.join(P['T'][-20:]).replace('\n', '\\n')}")
    
            elif(('#' not in txt1) and ('#' not in txt2) and (txt1 != txt2)) : 
                P['Warn'] += 1
                if(Verbose) :
                    print(f"\tMismatch {d['time']} cur:{d['cur']} chr:{txt2}<\t", end ='')
                    print(f"\t{''.join(P['T'][mod_cur-10:mod_cur]).replace('\n', '\\n')}", end = '')
                    print(f">{txt1}<", end='')
#                    print(f"{''.join(P['T'][cur+len(d['chr']):cur+len(d['chr'])+10]).replace('\n', '\\n')}")
    
            # delete the substring
            del P['T'][mod_cur : mod_cur + len(d['chr'])]
            del P['I'][mod_cur : mod_cur + len(d['chr'])]

    return 0

    

##########################################################################
# ## XML 


# Convert list of dictionary into xml
def list_of_dicts_to_xml(data_list, root_tag='root', item_tag='item'):
    """
    Converts a list of dictionaries into an XML root,
    placing dictionary values into attributes of XML elements.

    Args:
        data_list (list): A list of dictionaries to convert.
        root_tag (str): The tag name for the root element of the XML.
        item_tag (str): The tag name for each item element in the XML.

    Returns:
        root: The XML root.
    """
    root = ET.Element(root_tag)
    for item_dict in data_list:
        item_element = ET.SubElement(root, item_tag)
        for key, value in item_dict.items():
            # Convert value to string as XML attributes are strings
            item_element.set(key, str(value))

    return root



#####################################################################################
# ### Events XML -> WKS XML

# Events Token container -> list of token dictionaries
def tokens2dict(xml_root):
    """Convert list token XML to dictionary."""
    
    Token = []    
    d = {}
    for tok in xml_root.findall('Token'):
        token_old = d
        d = dict(tok.attrib)

        # Convert numeric strings to numbers if needed
        if 'id' in d: d['tokId'] = int(d['id'])

        if ('cur' not in d):
            print(f"tokens2dict Warning: no cur snt:{d['segId']} tokId:{d['id']} >{d['tok']}<")
            if('cur' in o) : d['cur'] = int(d['cur'])
            else : d['cur'] = 0
        d['cur'] = int(d['cur'])
        
        Token.append(d)
    
    return Token

# Events Token container -> list of modifications dictionaries
def mod2dict(xml_root):
    """Convert list token XML to dictionary."""

    def to_int(s):
        """
        Checks if a given string can be safely converted to an integer.
    
        Args:
            s: The string to check.
    
        Returns:
            True if the string can be converted to an integer, False otherwise.
        """
        try:
            int(s)
            return True
        except ValueError:
            return False

    Mod = [] 
    for m in xml_root.findall('Mod'):
        # convert Mod entry to dictionary
        d = dict(m.attrib)

        # Convert numeric strings to numbers if needed
        if('time' in d and to_int(d['time'])): d['time'] = int(d['time'])
        if('cur' in d and to_int(d['cur'])): d['cur'] = int(d['cur'])
        if('tid' in d and to_int(d['cur'])): d['tid'] = int(d['tid'])

        Mod.append(d)

    return Mod


def tokens2snt(tokList):
    """
    Convert list of token dictionary to list of lists of sentence.
    Assumes tokens have 'segId'
    """
    
    # Group tokens by sentence
    S = []
    L = []
    segId = 0
    for token in tokList:
        if(token.get('segId') != segId) :
            if(segId != 0) :
                S.append(L)
                L=[]
            segId = token.get('segId')        
        L.append((token.get('tok'), token.get('id')))
#        print(token.get('segId'), token.get('tok'), token.get('id'))
    
    if(L) : S.append(L)

    return S
