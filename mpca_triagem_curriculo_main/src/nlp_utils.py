import re
import unidecode
import spacy


nlp = spacy.load('en_core_web_sm', disable=['ner', 'textcat'])


def preprocessing(resume_txt: str) -> str:
    new_resume_txt = re.sub(r'<.*?>', '', resume_txt)
    new_resume_txt = re.sub(r'\s+', ' ', new_resume_txt)
    new_resume_txt = re.sub(r'https?://\w+', '', new_resume_txt)
    new_resume_txt = new_resume_txt.replace('\n', ' ')
    new_resume_txt = unidecode.unidecode(new_resume_txt)
    doc = nlp(new_resume_txt)
    tokens = [t.lemma_.lower() for t in doc if t.pos_ != 'PUNCT' and not t.is_stop]
    new_resume_txt = ' '.join(tokens).strip()
    return new_resume_txt.lower()


def preprocessing_v2(resume_txt: str) -> str:
    new_resume_txt = re.sub(r'<.*?>', '', resume_txt)
    new_resume_txt = re.sub(r'\s+', ' ', new_resume_txt)
    new_resume_txt = new_resume_txt.replace('\n', ' ')
    new_resume_txt = unidecode.unidecode(new_resume_txt)
    return new_resume_txt.lower()


def no_spacing(resume_txt: str) -> str:
    new_resume_txt = re.sub(r'\s+', '', resume_txt)
    return new_resume_txt.lower()
    
