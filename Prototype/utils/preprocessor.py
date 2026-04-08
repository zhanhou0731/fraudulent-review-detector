import re
import emoji
import emot

emot_obj = emot.core.emot()

def get_first_feature_tag(meaning):
    if not meaning:
        return ""

    first_part = meaning.split(',')[0]
    first_part = first_part.split(' or ')[0]
    
    clean_tag = first_part.strip().replace(' ', '_')
    return f" :{clean_tag}: "

def convert_emoticons(text):
    res = emot_obj.emoticons(text)
    
    if not res['flag']:
        return text

    for i in range(len(res['value'])):
        emoticon = res['value'][i]
        meaning = res['mean'][i]
        
        clean_meaning = get_first_feature_tag(meaning)

        escaped_emoticon = re.escape(emoticon)

        text = re.sub(escaped_emoticon, clean_meaning, text)
        
    return text

def clean_text(text):
    if text is None:
        return ""
    text = str(text)
    
    text = emoji.demojize(text, delimiters=(" :", ": ")) 

    text = convert_emoticons(text)

    text = re.sub(r'<.*?>', '', text)

    text = re.sub(r'\s+', ' ', text).strip()
    
    return text