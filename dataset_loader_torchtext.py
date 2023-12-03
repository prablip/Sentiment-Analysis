import torch
from torchtext.data import Field, LabelField, BucketIterator, TabularDataset
import spacy
import re

#dictionary for abbreviations
abbr = { 
"aren't": "are not",
"ain't": "am not",
"could've": "could have",
"couldn't": "could not",
"can't": "cannot",
"'cause": "because",
"doesn't": "does not",
"didn't": "did not",
"don't": "do not",
"hadn't": "had not",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'll": "he will",
"he's": "he is",
"how'd": "how did",
"how'll": "how will",
"how's": "how is",
"here's": "here is",
"i'd": "i would",
"i'll": "i will",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'll": "it will",
"it's": "it is",
"let's": "let us",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"must've": "must have",
"mustn't": "must not",
"needn't": "need not",
"she'd": "she would",
"she'll": "she will",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"that'd": "that would",
"that's": "that is",
"there'd": "there had",
"there's": "there is",
"they'd": "they would",
"they'll": "they will",
"they're": "they are",
"they've": "they have",
"wasn't": "was not",
"we'd": "we would",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"where'd": "where did",
"where's": "where is",
"who'll": "who will",
"we'll": "we will",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"who's": "who is",
"won't": "will not",
"you're": "you are",
"you've":"you have",
"y'all":"you all",
"wouldn't": "would not",
"you'd": "you would",
"you'll": "you will",
"let's": "let us"
}

def multiple_replace(adict, text):
  # Create a regular expression from all of the dictionary keys
  regex = re.compile("|".join(map(re.escape, adict.keys(  ))))

  # For each match, look up the corresponding value in the dictionary
  return regex.sub(lambda match: adict[match.group(0)], text)


def decontract(sentence):
  sentence=sentence.split()
  sentence=' '.join(sentence)
  sentence = multiple_replace(abbr, sentence)
  return sentence


def cleanEng(x):
  x=str(x)
  x=x.lower()
  x=re.sub(r'[^a-z0-9]+',' ',x)
  x=re.sub(' +', ' ',x) #removing extra spaces 
  if x and x[-1]==' ':
    x=x[:-1]
  x=x.strip()
  return x
  
# Load the spacy English tokenizer
spacy_english = spacy.load("en_core_web_sm")
def tokenize_english(text):
    text = decontract(text)
    text = cleanEng(text)
    return [token.text for token in spacy_english.tokenizer(text)]

class CustomDatasetLoader:
    def __init__(self, path, split_ratio=0.9):
        # Define the fields
        self.TEXT = Field(sequential=True, tokenize=tokenize_english, lower=True, init_token="<sos>", eos_token="<eos>", include_lengths=True)
        self.LABEL = LabelField(sequential=False, batch_first=True)
        self.ID = Field(use_vocab=False, batch_first=True)

        # load the CSV data into a TabularDataset
        data = TabularDataset(
            path=path,
            format='csv', 
            fields=[('id', None),("text", self.TEXT), ("label", self.LABEL)], 
            skip_header=True
        )

        # split the dataset into train and validation sets
        self.train_data, self.test_data = data.split(split_ratio=split_ratio, stratified=False)

        # Build the vocabulary
        self.TEXT.build_vocab(self.train_data, max_size=25000, min_freq=3, vectors = "glove.6B.100d", unk_init = torch.Tensor.normal_)
        self.LABEL.build_vocab(self.train_data)

    def get_data(self):
        return self.train_data, self.test_data

    def get_iterators(self, batch_size=32):
        # Create the iterators
        train_iterator, test_iterator = BucketIterator.splits(
            (self.train_data, self.test_data), 
            batch_size=batch_size, 
            sort_key=lambda x: len(x.text), 
            sort_within_batch=True, 
            shuffle=True,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        return train_iterator, test_iterator
    
    def get_vocab(self):
        return self.TEXT.vocab, self.LABEL.vocab

    def get_fields(self):
        return self.TEXT, self.LABEL
    
