import nltk
nltk.download('punkt')

from transformers import AutoTokenizer, BertTokenizer
import pyarabic.araby as araby
from emoji.unicode_codes import UNICODE_EMOJI
import string

import re 
import qalsadi.lemmatizer
from nltk.stem.isri import ISRIStemmer
from emoji.unicode_codes import UNICODE_EMOJI
#from emosent import get_emoji_sentiment_rank
import regex

class preProcess:
      """ Class for pre-processing text for analyzing
      gets a text and creating a preProcess object with the following Attributes:
      1. text :  the input text
      2. Emojie : the text with no emojies , emojies sentiment
      3. text satistic: word counts , sentences
      4. cleaner : the text after Removing punctuation, stop words and emojies, and Merging duplicates ( ignoring the letter "L" - for allah) 
      5.Tokenize : list of the words after cleaner and tokenization 
      6. Lemmatization : list of the words after cleaner and tokenization and Lemmatization

      pipline of the process : text input(string) :
      *Emoji sentiment and remove emojies
      *text satistic:word counts,sentences
      *cleaner: 
          1. Removing punctuation
          2. Removing stop words and emojies
          3. Merging duplicates ( ignoring the letter "L" - for allah)
      *Tokenizer the text
      *Lemmatization
        Parameters
        :text: string for processing

        Examples
        text ="شركة شراء اثاث مستعمل بالرياض بافضل الاسعار "
        print (test.lemmatization) --> ['شركة', 'شراء', 'اثاث', 'مستعمل', 'رياض', 'بافضل', 'الاسعار']

        """
  def __init__(self, tokenizer_model='gigabert', remove_consecutive_chars=True, pure_arabic=True,punc_remove=True,remove_emoji=True, extract_emoji_sentiment=False):
    """   
    creating a preProcess object with the following Attributes:
      1. text :  the input text
      2. Emojie : the text with no emojies , emojies sentiment
      3. text satistic: word counts , sentences
      4. cleaner : text after Removing punctuation, stop words and emojies, and Merging duplicates ( ignoring the letter "L" - for allah) 
      5. Tokenizer : list of the words after cleaner and tokenization 
      6. Lemmatization : list of the words after cleaner and tokenization and Lemmatization
  
    Parameters: 
    :text (string): string for processing
   
    """
    self.remove_consecutive_chars = remove_consecutive_chars
    self.punc_remove = punc_remove
    self.extract_emoji_sentiment = extract_emoji_sentiment
    self.remove_emoji = remove_emoji
    self.pure_arabic = pure_arabic
    self.tokenizer = arabic_tokenizer(model_name=tokenizer_model)
    self.satistics = None
    self.emojies_sentiment = None
    self.tokenized_text = None
    self.lemmatized_text = None
     
  def transform(self, text):
    #checking if the text is string  
    if (not isinstance(text, str)) : 
      raise Exception("The input is not a string")
    new_text = text
    if self.pure_arabic:
      new_text = self.remove_non_arabic_char(new_text)
    if self.remove_consecutive_chars:
      new_text = self.consecutive_chars_removal(new_text)
    if self.punc_remove:
      new_text = self.punc_removal(new_text)
    if self.extract_emoji_sentiment:
      self.emojies_sentiment =  self.emoji_sentiment()    #to implemete tne sentimente
    if self.remove_emoji:
      new_text = self.deEmojify(new_text)
    
    self.satistics = self.words_counter(new_text) # int: num of words
    self.tokenized_text = self.tokenizer.tokenize(new_text, remove_punckt=self.punc_remove)
    self.lemmatized_text = self.lemmatizer(new_text)
    return new_text
  
  
  """ to add more satistic parameters
  def calc_stats(text):
    #checking if the text is string  
    if (not isinstance(text, str)) : 
      raise Exception("The input is not a string")
    self.satistics = {"word_count" : self.words_counter() } # te comlete sentence counter
    return self.satistics
  """

  # emoji func
  def emoji_sentiment(self):
    # lack of implementation!!! neet to check if needed with avrahami and if to do it before or after cleanning
    return "in process"
  
  # satistic func
  def words_counter(self, text):
    """
    gets string and returns th number of the word (int) 
  
    """  
    return len(re.findall(r'\w+', text))
  
  # Non-Arabic characters removal- keep arabic and numeric chars and remove the rest
  def remove_non_arabic_char(self, text):
  # Gets String and remove non arabic letters
    t = re.sub(r'[^0-9\u0600-\u06ff\u0750-\u077f\ufb50-\ufbc1\ufbd3-\ufd3f\ufd50-\ufd8f\ufd50-\ufd8f\ufe70-\ufefc\uFDF0-\uFDFD-\~@#$^*()_+=[\]{}|\\,.?: -@#$% <>` "]+', ' ', text)
    return t


  #cleaner
  def punc_removal(self, text):
    """
  gets string and returns the string  after Removing punctuation, stop words and emojies,
  and Merging duplicates ( ignoring the letter "L" - for allah)
  
    """  
    known_punct = string.punctuation
    clean_text =''.join([t for t in text if not all(l in string.punctuation for l in t)])
    return clean_text

  #delete emoji
  def deEmojify(self, text):
    """
  gets string and returns the string  after Removing emojies
  
  :parameters: string
  :returns: string  
    """  
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return str(regrex_pattern.sub(r'', text))
  
  #Merging duplicates letters
  def consecutive_chars_removal(self, text):
    """
  Merging duplicates letters( ignoring the letter "L" - for allah and "h" for loughing)
  
  :parameters: text (of type string)
    the text to run the function over
  :returns: string  
    """
    list = []
    prev = ""
    count = 0 
    for ch in text:
        if ch != prev:
            list.append(ch)
        #need to be fixxed
        """
        # a special case of two specifc letters in arabic
        elif ch == "ل" or ch == "ه":
          if count < 2 :
             count +=1
             list.append(ch)
          else:
            count = 0 
        """
        prev = ch  
    return ''.join(list)
  
  #lemmatinazer
  def lemmatizer(self, text):
    """
   lemmatize a text to list

   module using: "qalsadi.lemmatizer"
  
  :parameters: string
  
  :returns: list  
    """
    lemmer = qalsadi.lemmatizer.Lemmatizer()
    return lemmer.lemmatize_text(text)

class arabic_tokenizer(object):
    """ Class for tokenize arabic text in different methods

        Parameters
        ----------

        Attributes
        ----------

        Examples
        --------

        """
    def __init__(self, model_name):
      self.model_name = model_name
      if self.model_name == 'arabert':
        model_str = 'aubmindlab/bert-base-arabert'
        arabert_tokenizer = AutoTokenizer.from_pretrained(model_str)
        arabert_tokenizer = self.add_emojies_to_tokenizer(arabert_tokenizer)
        self.tokenizer = arabert_tokenizer.tokenize
      elif self.model_name == 'gigabert':
        model_str = 'lanwuwei/GigaBERT-v4-Arabic-and-English'
        gigabert_tokenizer = BertTokenizer.from_pretrained(model_str, do_lower_case=True)
        gigabert_tokenizer = self.add_emojies_to_tokenizer(gigabert_tokenizer)
        self.tokenizer = gigabert_tokenizer.tokenize
      elif self.model_name == 'nltk':
        self.tokenizer = nltk.word_tokenize
      elif self.model_name == 'pyarabic':
        self.tokenizer = araby.tokenize
    def get_model(self):
      return self.model

    def set_model(self, model_name):
      self.model = model_name
      
    def tokenize(self, text, remove_punckt=True):
      tokenized_text = self.tokenizer(text)
      if remove_punckt:
        known_punct = string.punctuation
        tokenized_text_filtered = [t for t in tokenized_text if not all(l in string.punctuation for l in t)]
        return tokenized_text_filtered
      else:
        return tokenized_text


    @staticmethod
    def add_emojies_to_tokenizer(tokenizer):
      for cur_emoji in UNICODE_EMOJI.keys():
        tokenizer.add_tokens(cur_emoji)
      return tokenizer