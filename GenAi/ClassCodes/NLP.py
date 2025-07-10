# Importing necessary modules from NLTK (Natural Language Toolkit)
import nltk

# Downloading required NLTK resources
nltk.download('punkt')  # Tokenizer models for sentence and word tokenization
nltk.download('stopwords')  # Stopword corpus

# Sample document for sentence tokenization
document = """It was a very very pleasant day. The weather was cool and there were light showers. 
I went to the market to buy some groceries."""

# Sentence to process for word tokenization and further NLP steps
sentence = "Send all the 50 documents related to chapters 1,2,3 at Omega@ewc.com"

# Importing sentence and word tokenizers
from nltk.tokenize import sent_tokenize, word_tokenize

# Sentence Tokenization
sents = sent_tokenize(document)
print("Sentences from document:")
print(sents)
print()

# Word Tokenization
words = word_tokenize(sentence)
print("Word Tokens from sentence:")
print(words)
print()

# Importing regular expression library
import re

# Splitting string by commas and whitespace
split_tokens = re.split(r'[,\s]+', sentence)
print("Split tokens using regex:")
print(split_tokens)
print()

# Removing all non-alphabetic characters (excluding @ and . for email)
cleaned_sentence = re.sub(r'[^a-zA-Z\s@.]+', '', sentence)

# Importing stopwords
from nltk.corpus import stopwords
sw = set(stopwords.words('english'))  # Load stopwords into a set for faster lookup

# Tokenizing cleaned sentence again using regex
tokens = re.split(r'[,\s]+', cleaned_sentence)

# Importing different stemmers
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer

# Initializing Snowball Stemmer
sst = SnowballStemmer('english')

# Stemming each token (excluding stopwords)
stemmed_tokens = [sst.stem(w) for w in tokens if w.lower() not in sw]
print("Stemmed tokens (after removing stopwords):")
print(stemmed_tokens)
print()

# Demonstrating how different stemmers behave
print("Snowball Stemmer on 'curled':", sst.stem('curled'))

# Initializing and using Porter Stemmer
ps = PorterStemmer()
print("Porter Stemmer on 'curl':", ps.stem('curl'))
print("Porter Stemmer on 'curly':", ps.stem('curly'))
