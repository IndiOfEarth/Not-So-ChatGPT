# import all necessary dependencies
import json
import re
import random
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
chat_memory = []

# Download necessary NLTK resources 
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger_eng')

# loads the json file 
# converts to a pandas dataframe
def load_json(file_path):
    with open(file_path) as file:
        data = json.load(file)
        intents_list = data["intents"]
        df = pd.DataFrame(intents_list)
        return df
    
# Loop through patterns list and tags.
# For each pattern in list, add an "r" symbol
# Create a dictionary with pattern and tags
# will return as rgx2int
def storePatterns(df):
    rgx2int = {}
    for _, row in df.iterrows():
        # Join all patterns in the list with a | character to create a regex string
        joined_pattern = "|".join(row["patterns"])
        # use raw string format to preserve regex characters
        regex_pattern = rf"{joined_pattern}"
        # map the regex pattern to the corresponding tag
        rgx2int[regex_pattern] = row["tag"]
    return rgx2int


# Loop through the list of JSON objects:
# For each object in list, get responses and tag
# Add to the dictionary: tags as keys and responses as values
# will return as int2res
def storeResponses(df):
    return dict(zip(df['tag'], df['responses']))

# print(storePatterns(load_json('intents.json'))) # rgx2int (for debugging)

def load_files():
    df = load_json('./intents.json')
    rgx2int = storePatterns(df)
    int2res = storeResponses(df)
    return rgx2int, int2res

# Adding POS (Part of Speech) Tagging/Recognition
# Wanting to identify the elements from the user's speech/text
# Will use this to create different responses
def get_pos_tags(tokens):
    # tokens = nltk.word_tokenize(text) # splits into tokens list
    tagged = nltk.pos_tag(tokens) # uses nltk pos tag function to create tags for each

    # will lemmatise for each noun and/or verb in the (word, tag) pair from 'tagged'
    lemmatizer = WordNetLemmatizer() # instance of the lemmatizer
    lemmatized = [(lemmatizer.lemmatize(word, pos='n') if tag.startswith('NN')
                   else lemmatizer.lemmatize(word, pos='v') if tag.startswith('VB')
                   else word, tag)
                   for word, tag in tagged
                  ]
    
    return lemmatized # returns lemmatized 


# Just adding the necessary preprocessing function for user_input
# preprocess user input
def preprocess_text(text):
    # make all the text lower case (removes any need for case-sensitivity in user_input)
    text = text.lower()

    # remove punctuation
    # uses String.translate() method and .maketrans() method (mapping table)
    text = text.translate(str.maketrans('','', string.punctuation))

    # tokenise - NLTK library
    tokens = nltk.word_tokenize(text)

    # remove stopwords - create a set from nltk stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # stemming
    # chosen the PorterStemmer() for this - slight variations in different stemmers
    # stemmer = PorterStemmer()
    # tokens = [stemmer.stem(word) for word in tokens]

    # lemmatization (applied after stemming just to show both)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # return the tokens all joined together to create preprocessed text
    return tokens
    # return " ".join(tokens) 


# TEST - Preprocessing and tag patterns in intents.json
def prepare_pattern_data(rgx2int):
    pattern_data = []

    for pattern, intent in rgx2int.items():
        individual_phrases = pattern.split('|')

        all_tokens = []
        all_nouns = []
        all_verbs = []
    
        for phrase in individual_phrases:
            tokens = preprocess_text(phrase)

            tagged = get_pos_tags(tokens)
    
            # extracts the nouns and verbs
            nouns = [word for word, tag in tagged if tag.startswith('NN')]
            verbs = [word for word, tag in tagged if tag.startswith('VB')]
    
            all_tokens.extend(tokens)
            all_nouns.extend(nouns)
            all_verbs.extend(verbs)

        # add to the pattern data
        pattern_data.append({
            "regex": pattern,
            "intent": intent,
            "pattern_tokens": list(set(all_tokens)),
            "pattern_nouns": list(set(all_nouns)),
            "pattern_verbs": list(set(all_verbs))
        })
    print("DEBUG: prepare_pattern_data(): ", pattern_data)
    print("\n")

    return pattern_data

# # TEST - Preprocessing and tag patterns in intents.json
# def prepare_pattern_data(rgx2int):
#     pattern_data = []

#     for pattern, intent in rgx2int.items():
#         tokens = preprocess_text(pattern) # preprocess the pattern (remove stopwords, etc.)

#         # gets the pos tags from the cleaned text
#         # extracts the nouns and verbs
#         tagged = get_pos_tags(tokens)
#         nouns = [word for word, tag in tagged if tag.startswith('NN')]
#         verbs = [word for word, tag in tagged if tag.startswith('VB')]

#         # add to the pattern data
#         pattern_data.append({
#             "regex": pattern,
#             "intent": intent,
#             "pattern_tokens": tokens,
#             "pattern_nouns": nouns,
#             "pattern_verbs": verbs
#         })
#         print("DEBUG: prepare_pattern_data(): ", pattern_data)
#         print("\n")

#     return pattern_data

# one thing that does need to be done is to amend the generate_response()
# we will now preprocess the user_input
# then continue with regex pattern matching to get rgx2int
# and int2res

def generate_response(user_input):
    rgx2int, int2res = load_files()
    # cleaned_input = preprocess_text(user_input)
    # print(preprocess_text(user_input))

    # Part-Of-Speech Tagging to add depth to the responses and query based on what the user says
    tokens = nltk.word_tokenize(user_input)  # <-- Fix added
    pos_tags = get_pos_tags(tokens)
    
    # pos_tags = get_pos_tags(user_input)

    # From the user's response, then get all the nouns and verbs from their speech
    # Will use this in the response back to the user.
    nouns = [word for word, tag in pos_tags if tag.startswith('NN')]
    verbs = [word for word, tag in pos_tags if tag.startswith('VB')]
    
    # all regex patterns (from rgx2int) with preprocessing and pos tagging
    pattern_data = prepare_pattern_data(rgx2int)
        
    best_intent = "unknown"
    best_score = 0
    
    # loop over all the patterns to identify parts of speech and return a response based on this
    for pattern_info in pattern_data:
        # compare the number of nouns and verbs from patterns and the user_input
        noun_comparison = len(set(nouns) & set(pattern_info["pattern_nouns"]))
        verb_comparison = len(set(nouns) & set(pattern_info["pattern_verbs"]))
        total_score = noun_comparison + verb_comparison

        # FOR DEBUGGING
        print("user nouns: ", set(nouns))
        print("pattern nouns: ", set(pattern_info["pattern_nouns"]))
        # print("total score ", total_score)
        # print("best score ", best_score)
        

        if total_score > best_score:
            best_score = total_score
            best_intent = pattern_info["intent"]
    
    # check if a matching intent/tag was found and return a response from this
    if best_intent != "unknown":
        response_list = int2res[best_intent]
        response = random.choice(response_list)
        print(f"Chatbot: {response}")
        return True

    # does the basic regex match
    for pattern, tag in rgx2int.items():
        if re.search(pattern, user_input, re.IGNORECASE):
            response_list = int2res[tag]
            response = random.choice(response_list)

            # Including the POS Tagging
            # Acknowledge the noun and highlight the action/verb
            if nouns:
                response += f" I noticed you mentioned {', '.join(nouns)}."
            if verbs:
                response += f" Are you looking to {', '.join(verbs)} it?"
            
            print(f"Chatbot: {response}")
            return True
        
            
    fallback = "Sorry, I don't understand."
    print(f"Chatbot: {fallback}")
    return False

# # one thing that does need to be done is to amend the generate_response()
# # we will now preprocess the user_input
# # then continue with regex pattern matching to get rgx2int
# # and int2res

# def generate_response(user_input):
#     rgx2int, int2res = load_files()
#     # cleaned_input = preprocess_text(user_input)
#     print(preprocess_text(user_input))

#     # Part-Of-Speech Tagging to add depth to the responses and query based on what the user says
#     pos_tags = get_pos_tags(user_input)
#     print(pos_tags)

#     # From the user's response, then get all the nouns and verbs from their speech
#     # Will use this in the response back to the user.
#     nouns = [word for word, tag in pos_tags if tag.startswith('NN')]
#     verbs = [word for word, tag in pos_tags if tag.startswith('VB')]
#     print(nouns, verbs)

#     for pattern, tag in rgx2int.items():
#         if re.search(pattern, user_input, re.IGNORECASE):
#             response_list = int2res[tag]
#             response = random.choice(response_list)

#             # Including the POS Tagging
#             # Acknowledge the noun and highlight the action/verb
#             if nouns:
#                 response += f" I noticed you mentioned {', '.join(nouns)}."
#             if verbs:
#                 response += f" Are you looking to {', '.join(verbs)} it?"
            
#             print(f"Chatbot: {response}")
#             return True
            
#     fallback = "Sorry, I don't understand."
#     print(f"Chatbot: {fallback}")
#     return False

# Just rewriting the main chat loop
def chat():
    print("Chatbot: Hello! How can I help you today?")
    print("Chatbot: Say 'bye' anytime to exit!")

    while True:
        user_input = input("You: ")
        bot_response = generate_response(user_input)
        if user_input == "bye":
            print("Chatbot: Thank you for your interaction")
            break
        
        if bot_response == False:            
            print("Chatbot: I'm sorry, could you rephrase that again")
            
chat()