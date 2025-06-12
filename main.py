# import all necessary dependencies
import json
import re
import random
import pandas as pd
import nltk
import string
import math
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
chat_memory = []

# Handles loading and organising data from the intents.json file
class IntentLoader:
    # Constructor
    def __init__(self, path):
        self.path = path # stores the path to the json file.
        self.df = self.load_json() # stores the dataframe
        self.rgx2int = self.storePatterns() # rgx2int (precepts with matching intents)
        self.int2res = self.storeResponses() # int2res (intents with matching responses)
        self.tag2memory = self.storeMemoryCaptures() # (tags with matching memory fields)
        self.tag2followup = self.storeFollowUps() # stores tags with followups 

    # Loads the json file as a pandas dataframe
    def load_json(self):
        # opens the json file
        with open(self.path) as file:
            data = json.load(file)
            return pd.DataFrame(data["intents"])
    
    # Creates the rgx2int - dictionary full of patterns and matching intents
    def storePatterns(self):
        rgx2int = {}
        for _, row in self.df.iterrows():
            pattern = "|".join(row["patterns"]) # Join all patterns in the list with a | character to create a regex string
            rgx2int[rf"{pattern}"] = row["tag"] # Map the regex pattern to the corresponding tag 
        return rgx2int

    # Creates the int2res - dictionary full of intents and matching responses
    # This is the zip() function
    def storeResponses(self):
        return dict(zip(self.df['tag'], self.df['responses']))
    
    # Creates the tags2memory dictionary
    # For each tag, if there is a memory item, then add this pair to the dictionary
    def storeMemoryCaptures(self):
        tag2memory = {}
        for _, row in self.df.iterrows():
            if "memory" in row:
                tag2memory[row["tag"]] = row["memory"]
        return tag2memory
    
    # Needed to create the dictionary for follow up responses
    # for each tag, if there is a potential list of follow ups, add this pair to the dictionary
    def storeFollowUps(self):
        tag2followup = {}
        for _, row in self.df.iterrows():
            if "followups" in row:
                tag2followup[row["tag"]] = row["followups"]
        return tag2followup


# Handles tokenisation, stopword removal, lemmatization, and POS tagging
class Preprocessor:
    # Constructor
    def __init__(self):
        # Download necessary NLTK resources 
        nltk.download('punkt_tab')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        nltk.download('averaged_perceptron_tagger_eng')
        self.stop_words = set(stopwords.words('english')) # stores stop words
        self.lemmatizer = WordNetLemmatizer() # using the nltk lemmatizer

    # Cleans the text
    # Removes punctuation, tokenises the input, removes stopwords
    def clean_text(self, text):
        # make all the text lower case
        # remove punctuation and use String.translate() method and .maketrans() method (mapping table)
        text = text.lower().translate(str.maketrans('', '', string.punctuation))
        tokens = nltk.word_tokenize(text) # tokenises the text
        tokens = [word for word in tokens if word not in self.stop_words] # removes all stopwords that match from the set
        return tokens

    # map the NLTK tags (NN, VB, etc.) to WordNet POS tag
    def get_wordnet_pos(self, tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    # Gets the pos tags from the tokens provided (by user input)
    def get_pos_tags(self, tokens):
        tagged = nltk.pos_tag(tokens) # performs pos tagging
        lemmatized = [] # stores all lemmatized words
        for word, tag in tagged:
            wn_tag = self.get_wordnet_pos(tag)
            if wn_tag:
                lemma = self.lemmatizer.lemmatize(word, pos=wn_tag)
            else:
                lemma = word
            lemmatized.append((lemma, tag))
        return lemmatized
        # return [(self.lemmatize_by_pos(word, tag), tag) for word,tag in tagged]
    
    # Lemmatize the pos tagged words (nouns and verbs)
    def lemmatize_by_pos(self, word, tag):
        # checks if tag is a noun tag
        if tag.startswith('NN'):
            return self.lemmatizer.lemmatize(word, pos='n')
        # checks if tag is a verb tag
        elif tag.startswith('VB'):
            return self.lemmatizer.lemmatize(word, pos='v')
        return word
    

# Handles the intent matching via regex and POS comparison
class PatternMatcher:
    # Constructor
    def __init__(self, rgx2int, preprocessor):
        self.rgx2int = rgx2int
        self.preprocessor = preprocessor
        self.pattern_data = self.prepare_patterns()
    
    # Processing the tag patterns
    # for each regex pattern, distilling down to tokens and looking at pattern nouns and verbs
    # pattern data is much more detailed than standard regex matching
    def prepare_patterns(self):
        pattern_data = []
        for pattern, intent in self.rgx2int.items():
            phrases = pattern.split('|')
            all_tokens, all_nouns, all_verbs, all_adjs = [], [], [], []
            for phrase in phrases:
                tokens = self.preprocessor.clean_text(phrase)
                tagged = self.preprocessor.get_pos_tags(tokens)
                all_tokens.extend(tokens)
                all_nouns.extend([w for w, t in tagged if t.startswith('NN')])
                all_verbs.extend([w for w, t in tagged if t.startswith('VB')])
                all_adjs.extend([w for w, t in tagged if t.startswith('JJ')])
            pattern_data.append({
                "regex": pattern,
                "intent": intent,
                "pattern_tokens": list(set(all_tokens)),
                "pattern_nouns": list(set(all_nouns)),
                "pattern_verbs": list(set(all_verbs)),
                "pattern_adjs": list(set(all_adjs))
            })
        return pattern_data 
    
    # Performs matching logic for pattern data and user input
    def match(self, user_input):
        tokens = self.preprocessor.clean_text(user_input)
        tagged = self.preprocessor.get_pos_tags(tokens)

        # gets all the nouns, verbs andd adjectives from user input
        user_nouns = [w for w, t in tagged if t.startswith('NN')]
        user_verbs = [w for w, t in tagged if t.startswith('VB')]
        user_adjs = [w for w, t in tagged if t.startswith('JJ')]

        best_intent, best_score = "unknown", 0

        for p in self.pattern_data:
            noun_match = len(set(user_nouns) & set(p["pattern_nouns"]))
            verb_match = len(set(user_verbs) & set(p["pattern_verbs"]))
            adj_match = len(set(user_adjs) & set(p.get("pattern_adjs", [])))

            score = noun_match + (verb_match * 3) + (adj_match * 2)

            if score > best_score:
                best_intent, best_score = p["intent"], score
        
        if best_score != "unknown":
            return best_intent, user_nouns, user_verbs, user_adjs

        # Fallback to regex
        for pattern, intent in self.rgx2int.items():
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                return intent, user_nouns, user_verbs, user_adjs


        return "unknown", user_nouns, user_verbs, user_adjs

# Main class that ties everything together and runs the loop
class Chatbot:
    # Constructor
    def __init__(self, intents_path="./intents.json"):
        self.loader = IntentLoader(intents_path) # creates an instance of the intent loader
        self.preprocessor = Preprocessor() # creates the preprocessor object
        self.matcher = PatternMatcher(self.loader.rgx2int, self.preprocessor) # creates the pattern matcher
        self.responses = self.loader.int2res # creates the responses
        self.followupresponses = self.loader.tag2followup # creates the follow up responses
        self.memory = {} # creates a memory store (of key-value pairs)

    # Generates a response that will be returned based on user_input
    # Will look through tags, nouns, verbs
    def generate_response(self, user_input):

        # Memory Capture Functionality called before self.matcher.match() below
         # First check for memory-capturing patterns
        for pattern, tag in self.loader.rgx2int.items():
            if tag in self.loader.tag2memory:
                match = re.search(pattern, user_input, re.IGNORECASE)
                if match:
                    memory_fields = self.loader.tag2memory[tag]
                    if not isinstance(memory_fields, float):
                        for key, group_num in memory_fields.items():
                            self.memory[key] = match.group(group_num).capitalize()
                        response = random.choice(self.responses.get(tag, ["Got it!"]))
                        for key in self.memory:
                            response = response.replace(f"{{{key}}}", self.memory[key])
                        return response
        
        tag, nouns, verbs, adjs = self.matcher.match(user_input) # gets the intent, nouns, and verbs            
        
        if tag in self.responses:
            response = random.choice(self.responses[tag])


            # FOLLOW UP RESPONSES
            # Adding a follow up to the response based on the tag
            # access self.int2followupres
            # get a random response and append to already had response
            if tag in self.followupresponses:
                # print(self.followupresponses[tag])
                no_of_responses = int(len(self.followupresponses[tag]))
                x = random.randint(0, no_of_responses-1)
                follow_up = self.followupresponses[tag][x]
                response += f"\nChatbot: {follow_up}"
            
            # Substitute placeholders in response
            response = response.replace("{name}", self.memory.get("name", "friend"))
            response = response.replace("{capture_project_type}", self.memory.get("project_type","project"))
            response = response.replace("{capture_location}", self.memory.get("location","you"))
            response = response.replace("{capture_technology}", self.memory.get("technology","different technologies"))


           
            if nouns:
                response += f" I noticed you mentioned: {', '.join(nouns)}."
            if verbs:
                response += f" Are you looking to: {', '.join(verbs)}?"
            if adjs:
                response += f" It's great you described things as: {', '.join(adjs)}."
        else:
            response = "I'm not quite sure I understood. Could you rephrase that?"
        return response
    
    # Performs the main chat loop
    def chat_loop(self):
        print("Chatbot: Hello! How can I help you today?")
        print("Chatbot: Say 'exit' or 'quit anytime to exit!")
        while True:
            user_input = input("You: ")
            # termination logic
            if user_input.lower() == "exit" or user_input.lower() == "quit":
                print("Chatbot: Thank you for your interaction.")
                break
            print("Chatbot:", self.generate_response(user_input))
    

bot = Chatbot()
bot.chat_loop()
