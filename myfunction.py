from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
import string
from nltk.corpus import words
import nltk
import re
import os
from nltk.corpus import cmudict
import tqdm
import wordninja
import pandas as pd



#read all positive and negative words from the folder
positive_words = []
negative_words = []

with open('Master_Directory/positive-words.txt', 'r') as positive_file:
    positive_words = positive_file.read().splitlines()

with open('Master_Directory/negative-words.txt', 'r') as negative_file:
    negative_words = negative_file.read().splitlines()

    
positive_words = set(positive_words)
negative_words = set(negative_words)



# read all stopwords from folder
folder_path = "stopword"

# Initialize an empty list to store the stop words
stopwords_list = []

# Function to clean and normalize a word
def clean_word(word):
    # Convert to lowercase and remove symbols
    cleaned_word = word.lower()
    return cleaned_word

# Loop through all the text files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r") as file:
            # Read the content of the file, split it into words, clean them, and extend the stopwords_list
            words = file.read().split()
            cleaned_words = [clean_word(word) for word in words]
            stopwords_list.extend(cleaned_words)


# Remove duplicates by converting the list to a set and then back to a list
stopwords_list = list(set(stopwords_list))

# Define a regular expression pattern to match non-word characters (symbols)
pattern = re.compile(r'[^\w\s]')

# Remove non-word characters from each word in the list
stopwords_list = [pattern.sub('', word) for word in stopwords_list]

punctuation = set(string.punctuation)






# Define a function to clean and extract words
def clean_and_extract_words(tokens):
    cleaned_words = []
    for token in tokens:
        # Remove hyphens and replace them with a space
        cleaned_token = token.replace('-', ' ')
        # Remove non-alphanumeric characters and convert to lowercase
        cleaned_token = ''.join(e for e in cleaned_token if e.isalnum() or e.isspace())
        cleaned_token = cleaned_token.lower()
        # Split the token into words
        
        words = wordninja.split(cleaned_token)
        # Remove stopwords and punctuation
        clean_words = [word for word in words if word not in stopwords_list and word not in punctuation]
        clean_words = [word for word in clean_words if word != '']
        clean_words = [word for word in clean_words if not word.isdigit()]
        # Add words to the cleaned_words list
        cleaned_words.extend(clean_words)
    
    return cleaned_words



# Define a function to count_syllables
def count_syllables(word):
    d = cmudict.dict()
    if word.lower() in d:
        return max([len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]])
    else:
        # If the word is not found in the dictionary, you can use a simple rule-based approach
        vowels = "AEIOUaeiou"
        count = 0
        prev_char_was_vowel = False
        for char in word:
            if char in vowels:
                if not prev_char_was_vowel:
                    count += 1
                prev_char_was_vowel = True
            else:
                prev_char_was_vowel = False
        return count




# Define a function to count_personal_pronouns
def count_personal_pronouns(txt):
    # Define a regex pattern to match personal pronouns
    pattern = r'\b(I|we|my|ours|us)\b'
    
    # Use the regex pattern to find matches in the text (case-insensitive)
    matches = re.findall(pattern, txt, flags=re.IGNORECASE)

    # Filter out the word "US" as a country name
    matches = [word for word in matches if word.lower() != "us"]

    # Count the number of personal pronoun mentions
    pronoun_count = len(matches)

    return pronoun_count




# Define a function to calculate_average_word_length
def calculate_average_word_length(text):
    # Tokenize the input text by splitting it into words using whitespace
    words = text.split()

    # Calculate the total number of characters in all words
    total_characters = sum(len(word) for word in words)

    # Calculate the total number of words
    total_words = len(words)

    # Calculate the average word length
    if total_words > 0:
        average_word_length = total_characters / total_words
    else:
        average_word_length = 0

    return average_word_length





import nltk
from nltk.corpus import words

# Download the NLTK words corpus if you haven't already
nltk.download('words')

# Custom function to count syllables in a word with exceptions
def count_syllables_with_exceptions(word):
    vowels = 'aeiouAEIOU'
    word = word.lower()
    
    # Check if the word is in the NLTK English words corpus
    if word.endswith(('es', 'ed')) and word not in words.words():
        return 0  # Consider it as one syllable
    else:
        # Count the vowels in the word
        syllable_count = sum(1 for char in word if char in vowels)
        return syllable_count



    
# Custom function to txt_analysis
def txt_analysis(txt):
    
    tokens = word_tokenize(txt)
    
    # Use the function to clean and extract words
    clean_words = clean_and_extract_words(tokens)

    # Convert the list to a set to remove duplicates
    clean_words = list(set(clean_words))
    
    # Initialize the scores
    positive_score = 0
    negative_score = 0

    # Calculate the scores
    for word in clean_words:
        if word.lower() in positive_words:
            positive_score += 1
        elif word.lower() in negative_words:
            negative_score -= 1*-1

    Polarity_Score =(positive_score-negative_score)/ ((positive_score + negative_score) + 0.000001)
    Subjectivity_Score = (positive_score + negative_score)/ ((len(clean_words)) + 0.000001)
    
    # Tokenize the input text into sentences
    sentences = sent_tokenize(txt)

    # Count the number of words and sentences
    num_words = len(clean_words)
    num_sentences = len(sentences)

    # Calculate the average sentence length
    if num_sentences > 0:
        average_sentence_length = num_words / num_sentences
    else:
        average_sentence_length = 0
        
        
    # Calculate the average number of words per sentence
    if len(sentences) > 0:
        average_words_per_sentence = len(clean_words) / len(sentences)
    else:
        average_words_per_sentence = 0
        
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    clean_words_nltk = len([word for word in tokens if word.lower() not in stop_words and word not in punctuation])
    
    # Count the number of complex words (words with three or more syllables)
    num_complex_words = sum(1 for word in tqdm.tqdm(clean_words) if count_syllables(word) > 2)

    # Count the total number of words
    total_word_count = len(clean_words)

    # Calculate the percentage of complex words
    if total_word_count > 0:
        percentage_complex_words = (num_complex_words / total_word_count) * 100
    else:
        percentage_complex_words = 0
        
    Fog_Index = 0.4 * (average_sentence_length + percentage_complex_words)

    pronoun_count = count_personal_pronouns(txt)
    
    # Count the number of syllables in each word
    syllables_per_word = [count_syllables_with_exceptions(word) for word in clean_words]

    # # Create a dictionary to map each word to its syllable count
    word_syllable_counts = sum(syllables_per_word)/len(syllables_per_word)
    
    average_length = calculate_average_word_length(txt)
    
    
    return positive_score,negative_score,Polarity_Score,Subjectivity_Score,\
           average_sentence_length,percentage_complex_words,Fog_Index,average_words_per_sentence,\
            num_complex_words,clean_words_nltk,word_syllable_counts,pronoun_count,average_length




