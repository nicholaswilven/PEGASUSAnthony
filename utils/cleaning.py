import re
import unicodedata
import os

from .sentencepiece_tokenizer import fetch_tokenizer

# Load Environment Variables from .env
from dotenv import load_dotenv
load_dotenv()
MODEL_MAX_LENGTH = int(os.getenv("MODEL_MAX_LENGTH"))

def remove_news_headline(text,delim):
    # Helper function to remove news headline (for example: JAKARTA, liputan6.com -- )
    x = text.split(delim)
    if len(x)>1: # buang yang bukan konten
        return " ".join(x[1:])
    else:
        return x[0]

def text_cleaning(input_string, is_news = True):
    # Main function to clean text, removes link, bullet point, non ASCII char, news headline, parantheses,
    # punctuation except "," and ".", numbers with dot (enumerating), extra whitespaces, too short sentences.
    lowercase = input_string.lower()
    # stripped_html = BeautifulSoup(lowercase, 'html.parser').get_text()
    remove_link = re.sub(r'(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w\.-]*)', '', lowercase).replace("&amp;","&")
    remove_bullet = "\n".join([T for T in remove_link.split('\n') if '•' not in T and "baca juga:" not in T])
    remove_accented = unicodedata.normalize('NFKD', remove_bullet).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    # remove news headline
    if is_news:
        y = remove_news_headline(remove_accented,'- ')
        y = remove_news_headline(y,'– ') 
    else:
        y = remove_accented
    remove_parentheses = re.sub("([\(\|]).*?([\)\|])", "\g<1>\g<2>", y) 
    remove_punc = re.sub(r"[^\w\d.\s]+",' ', remove_parentheses)
    remove_num_dot = re.sub(r"(?<=\d)\.|\.(?=\d)|(?<=#)\.","",remove_punc)
    remove_extra_whitespace =  re.sub(r'^\s*|\s\s*', ' ', remove_num_dot).strip()
    return ".".join([s for s in remove_extra_whitespace.strip().split('.') if len(s.strip())>10]).replace("_","")

def process_input_eval(text : str, tokenizer = fetch_tokenizer()):
    # Prepare text for generation
    t = text_cleaning(text)
    return tokenizer(t,
              return_tensors = "tf",
              padding = 'max_length',
              max_length = MODEL_MAX_LENGTH,
              truncation = True
              )

def cleaning_oscar(examples):
    # hf datasets mapping function to clean OSCAR corpus dataset
    # 1. Takes only confident indonesian sentences (language label is "id")
    # 2. Applies above text_cleaning function
    # 3. Truncate articles to 500 words
    clean_article = []
    L_article = [doc.split('\n') for doc in examples['text']]
    L_lang_iden = [doc['line_identifications'] for doc in examples['meta']]
    for k in range(len(L_article)):
        article = L_article[k]
        lang_iden = L_lang_iden[k]
        clean_text = []
        for i in range(len(article)):
            if lang_iden[i] != None:
                if lang_iden[i]['label'] == 'id':
                    clean_text.append(article[i])
        text = text_cleaning("\n".join(clean_text))
        tokens = len(text.split())
        if tokens >= 500:
            text = " ".join(text.split()[:500])
        #else:
        #    text = "SKIP" # can also skip this line is want to include the "short" articles
        clean_article.append(text)
    return {'text':clean_article}