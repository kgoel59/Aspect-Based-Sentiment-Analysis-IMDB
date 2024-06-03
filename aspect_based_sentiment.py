import os
import re
import ast
import string
import tarfile
import requests
import zipfile
import pandas as pd
import numpy as np
import collections
import logging
import warnings
import tensorflow as tf
import tensorflow_datasets as tfds
import gensim
import nltk
from tempfile import NamedTemporaryFile
from bs4 import BeautifulSoup
from keras import layers
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
from gensim import corpora
from transformers import logging as transformers_logging
from transformers import TFDistilBertModel, AutoTokenizer
import spacy

# Configure warnings and logging
warnings.filterwarnings("ignore")
transformers_logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
tf.get_logger().setLevel(logging.ERROR)

# Global variables
batch_size = 32
random_state = 1
epoch = 1

class Setup:
    """Class to Setup"""
    def __init__(self):
        self.setup_nltk()
        self.setup_gpu()

    @staticmethod
    def setup_nltk():
        """Download necessary NLTK datasets"""
        nltk.download('averaged_perceptron_tagger')
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('brown')

    @staticmethod
    def setup_gpu():
        """Set up GPU configuration."""
        print("Setting Tensorflow for GPU")
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)

    @staticmethod
    def section(text,n=20):
        """section text"""
        print()
        print("-"*n)
        print(text)
        print("-"*n)
        print()

class DatasetManager:
    """Class to Download and Load Data"""
    def __init__(self, url, extract_to):
        self.url = url
        self.extract_to = extract_to

    def download_dataset(self):
        """Download and extract the dataset from a given URL."""
        print(f"Downloading and extracting dataset from {self.url}")

        # Extract the file name from the URL
        tar_gz_path = self.url.split("/")[-1]

        # Download the dataset
        response = requests.get(self.url, stream=True, timeout=50000)
        with open(tar_gz_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)

        # Create directory if it does not exist
        os.makedirs(self.extract_to, exist_ok=True)

        # Extract the downloaded file
        with tarfile.open(tar_gz_path, "r:gz") as tar:
            tar.extractall(path=self.extract_to)

        # Optionally, remove the tar.gz file after extraction
        os.remove(tar_gz_path)

        print("Dataset downloaded and extracted successfully")

    def read_reviews(self, directory, label):
        """Read reviews from specified directory and return as DataFrame."""
        reviews = []
        file_paths = [os.path.join(directory, f) for f in os.listdir(directory)]
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as file:
                review_text = file.read()
                reviews.append((review_text, label))
        return pd.DataFrame(reviews, columns=['review', 'sentiment'])

    def load_dataset(self):
        """Load the dataset into pandas DataFrame."""
        if not os.path.exists(self.extract_to):
            self.download_dataset()

        pos_train = self.read_reviews(os.path.join(self.extract_to, 'aclImdb', 'train', 'pos'), 1)
        neg_train = self.read_reviews(os.path.join(self.extract_to, 'aclImdb', 'train', 'neg'), 0)
        pos_test = self.read_reviews(os.path.join(self.extract_to, 'aclImdb', 'test', 'pos'), 1)
        neg_test = self.read_reviews(os.path.join(self.extract_to, 'aclImdb', 'test', 'neg'), 0)

        # Combine into single DataFrame
        dataset = pd.concat([pos_train, neg_train, pos_test, neg_test], ignore_index=True)

        # Optionally shuffle the data
        dataset = dataset.sample(frac=1, random_state=random_state).reset_index(drop=True)

        print("Datasets loaded and combined successfully")
        return dataset

class TextPreprocessor:
    """Class to Preprocess text"""
    def __init__(self):
        # Define combined set of custom and English stopwords
        self.remove_words = set([
            's', 'no', 'in', 'there', 'he', 'more', 'can', 'end', 'life', 'i', 'out', 'as',
            'get', 'go', 'an', 'take', 'turn', 'be', 'so', 'can', 'more', 'many', 'film',
            'nt', 'who', 'thing', 'little', 'not', 'said', 'say', 'could', 'think', 'know',
            'give', 'one', 'way', 'come', 'first', 'point', 'back', 'br', 'movie', 'good',
            'like', 'find', 'black', 'sex', 'time', 'day', 'm', 'ca', 'us', 'make', 'will',
            'even', 'try', 'lot', 'year', 'old', 'though', 'away', 'me', 'man', 'true',
            'stand', 'would', 'ever', 'never', 'down', 'here', 'best', 'see', 'seen',
            'done', 'called', 'being', 'something', 'guy', 'every', 'really', 'going',
            'movie', 'film', 'flick', 'one', 'in', 'it', 'have', 'who', 'as', 'he'
        ])
        self.stopwords_set = set(stopwords.words('english')).union(self.remove_words)
        self.lemmatizer = WordNetLemmatizer()
        self.nlp = spacy.load("en_core_web_sm")
        self.add_custom_boundaries(self.nlp)

    def add_custom_boundaries(self, nlp):
        """Add a custom component to the spaCy pipeline to adjust sentence boundaries."""
        @nlp.component("set_custom_boundaries")
        def set_custom_boundaries(doc):
            for token in doc[:-1]:  # Avoid accessing out of range
                if token.text == ',':
                    doc[token.i + 1].is_sent_start = False
            return doc
        nlp.add_pipe("set_custom_boundaries", before='parser')

    def denoise_text(self, text):
        """Remove unwanted characters and format text."""
        text = self.remove_whitespace(text)
        text = self.remove_html(text)
        text = self.remove_between_square_brackets(text)
        text = self.remove_url(text)
        text = self.remove_special_characters(text)
        text = self.remove_punctuation(text)
        return text

    def standardize_text(self, text):
        """Convert text to lowercase."""
        return text.lower()

    def denoise_and_standardize_text(self, text):
        """Apply denoise and standardize functions"""
        return self.standardize_text(self.denoise_text(text))

    def lemmatize(self, text):
        """Lemmatize text, excluding all stopwords."""
        tokens = word_tokenize(text)
        return [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stopwords_set]

    def load_clauses(self, url, data, labels):
        """
        Attempts to load a CSV file from a URL to extract preprocessed clauses and sentiment.
        If loading fails, extracts clauses manually from the provided data using spaCy's dependency parser.
        """
        try:
            print("Loading Clauses")
            # Attempt to fetch the data from the URL
            response = requests.get(url, timeout=50)
            response.raise_for_status()  # Raise an exception for bad responses
    
            # Use a temporary file to save the CSV content
            with NamedTemporaryFile(mode='w+', delete=False) as tmp_file:
                tmp_file.write(response.text)
                tmp_file.seek(0)  # Move the file pointer to the beginning of the file
    
                # Read the CSV data from the temporary file
                df = pd.read_csv(tmp_file.name)
            
            # Convert the string representation of list back to a list
            df['clauses'] = df['clauses'].apply(ast.literal_eval)
    
            return df['clauses'], df['sentiment']

        except (requests.RequestException, pd.errors.ParserError) as e:
          print("Loading Failed Extracting it")  # Print or log the error
          return [self.extract_clauses(review) for review in data], labels

    def extract_clauses(self, text):
        """
        Extracts clauses from a sentence using spaCy's dependency parser and preprocess.
        """
        text_clauses = []
        doc = self.nlp(text)

        for sent in doc.sents:
          sentence_clauses = []

          for token in sent:
            if 'VERB' in token.pos_:
                clause = ' '.join([tok.text for tok in token.subtree])
                processed_clause = ' '.join(self.lemmatize
                                            (self.standardize_text
                                             (self.denoise_text(clause))))
                if len(processed_clause) > 0:
                  sentence_clauses.append(processed_clause)

          if len(sentence_clauses) > 0:
            text_clauses.append(sentence_clauses)

        return text_clauses

    @staticmethod
    def remove_whitespace(text):
        """Remove Whitespace"""
        return re.sub(r'\s+', ' ', text).strip()

    @staticmethod
    def remove_html(text):
        """Remove HTML"""
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()

    @staticmethod
    def remove_between_square_brackets(text):
        """Remove Brackets"""
        return re.sub('\[[^]]*\]', '', text)

    @staticmethod
    def remove_url(text):
        """Remove Url"""
        return re.sub(r"https?://\S+|www\.\S+", '', text)

    @staticmethod
    def remove_special_characters(text):
        """Remove Special Characters"""
        pattern = r'[^a-zA-Z0-9\s]'
        return re.sub(pattern, '', text)

    @staticmethod
    def remove_punctuation(text):
        """Remove Punctuation"""
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)

class Tokenizer:
    """Class to handle Tokenization"""
    def __init__(self,data,subword_url,bert_model='distilbert/distilbert-base-uncased'):
        print("Loading Tokenizer")
        self.data = data
        self.subword_url = subword_url
        self.subword_tokenizer = self.get_subword_tokenizer(data=data)
        self.bert_tokenizer, self.bert_model = self.get_bert_tokenizer(model_name=bert_model)

    def get_subword_tokenizer(self,data=None):
        """load or build subword Tokenizer"""
        print("Loading Vocab for SubWord Tokenizer")

        # URL of the tokenizer file
        url = self.subword_url

        try:
            # Attempt to download the file
            response = requests.get(url, timeout=50000)
            if response.status_code == 200:
                with open('tokenizer_vocab.subwords', 'wb') as f:
                    f.write(response.content)
                print("File downloaded successfully!")
                return tfds.deprecated.text.SubwordTextEncoder.load_from_file('tokenizer_vocab')
            else:
                raise Exception("Download failed")
        except Exception as e:
            # If download fails or any other error occurs, build from corpus
            if data is None:
                raise ValueError("Data must be provided to build tokenizer",e)
            print("Failed to download file. Building tokenizer from corpus")
            return tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(self.data, target_vocab_size=2**16)


    def get_bert_tokenizer(self,model_name):
        """download bert tokenizer and return"""
        print("Downloading and Loading Bert Tokenizer")
        bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
        bert_model = TFDistilBertModel.from_pretrained(model_name)
        return bert_tokenizer, bert_model

    def get_subword_tokens(self,data, maxlen=512):
        """get subword tokens"""
        print("Generating SubWord Tokens")
        subwords_tokens = [self.subword_tokenizer.encode(sentence) for sentence in data]
        padded_tokens = tf.keras.preprocessing.sequence.pad_sequences(subwords_tokens,
                                                            value=0,
                                                            padding="post",
                                                            maxlen=maxlen)
        return padded_tokens

    def get_bert_tokens(self, data, maxlen=512):
        """get bert tokens"""
        print("Generating BERT Tokens")
        bert_tokens = []
        attention_mask = []

        def bert_tokenize(sentence, bert_tokenizer):
            # Tokenize the sentence, and ensure outputs are tensorflow tensors
            tokens = bert_tokenizer.encode_plus(
                sentence,
                add_special_tokens=True,
                max_length=maxlen,
                truncation=True,
                padding='max_length',
                return_tensors='tf'  # Ensures output is tf.Tensor
            )

            input_ids = tokens['input_ids'][0]
            attention_mask = tokens['attention_mask'][0]

            return [input_ids, attention_mask]

        for review in data:
            tokens = bert_tokenize(review,self.bert_tokenizer)
            bert_tokens.append(tokens[0])
            attention_mask.append(tokens[1])

        return [bert_tokens,attention_mask]

class DCNNWithSubWord(tf.keras.Model):
    """CNN Model"""
    def __init__(self,
                 vocab_size,
                 emb_dim=128,
                 nb_filters=50,
                 FFN_units=512,
                 dropout_rate=0.1,
                 name="dcnn_subword"):
        super(DCNNWithSubWord, self).__init__(name=name)
        print("Setting up CNN Model")
        self.embedding = layers.Embedding(vocab_size, emb_dim, name="embedding_layer")
        self.bigram = layers.Conv1D(filters=nb_filters,
                                    kernel_size=2,
                                    padding="valid",
                                    activation="relu",
                                    name="bigram_conv")
        self.pool_1 = layers.GlobalMaxPool1D(name="global_max_pool_1")
        self.trigram = layers.Conv1D(filters=nb_filters,
                                     kernel_size=3,
                                     padding="valid",
                                     activation="relu",
                                     name="trigram_conv")
        self.pool_2 = layers.GlobalMaxPool1D(name="global_max_pool_2")
        self.fourgram = layers.Conv1D(filters=nb_filters,
                                      kernel_size=4,
                                      padding="valid",
                                      activation="relu",
                                      name="fourgram_conv")
        self.pool_3 = layers.GlobalMaxPool1D(name="global_max_pool_3")
        self.dense_1 = layers.Dense(units=FFN_units, activation="relu", name="dense_layer_1")
        self.dropout = layers.Dropout(rate=dropout_rate, name="dropout_layer")
        self.last_dense = layers.Dense(units=1, activation="sigmoid", name="output_layer")

    def call(self, inputs, training=False):
        """Model layers"""
        x = self.embedding(inputs)
        x_1 = self.bigram(x)
        x_1 = self.pool_1(x_1)
        x_2 = self.trigram(x)
        x_2 = self.pool_2(x_2)
        x_3 = self.fourgram(x)
        x_3 = self.pool_3(x_3)

        merged = tf.concat([x_1, x_2, x_3], axis=-1) # (batch_size, 3 * nb_filters)
        merged = self.dense_1(merged)
        merged = self.dropout(merged, training)
        output = self.last_dense(merged)

        return output

class BILSTMWithSubWord(tf.keras.Model):
    """LSTM Model"""
    def __init__(self,
                 vocab_size,
                 emb_dim=256,
                 lstm_units=64,
                 FFN_units=512,
                 dropout_rate=0.1,
                 name="bi_subword"):
        super(BILSTMWithSubWord, self).__init__(name=name)
        print("Setting up LSTM Model")
        self.embedding = layers.Embedding(vocab_size,
                                          emb_dim)
        # BiLSTM Layer
        self.bilstm = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=False))

        # Dense and Output Layers
        self.dense_1 = layers.Dense(units=FFN_units, activation="relu")
        self.dropout = layers.Dropout(rate=dropout_rate)
        self.last_dense = layers.Dense(units=1, activation="sigmoid")

    def call(self, inputs, training=False):
        """Model layers"""
        x = self.embedding(inputs)

        # Apply BiLSTM layer
        x = self.bilstm(x)

        # Dense layer and dropout for regularization
        x = self.dense_1(x)
        x = self.dropout(x, training=training)

        # Output layer
        output = self.last_dense(x)

        return output

class BILSTMModelWithBert(tf.keras.Model):
    """LSTM with BERT"""
    def __init__(self,
                 embeddings,
                 lstm_units=64,
                 FFN_units=512,
                 dropout_rate=0.2,
                 name="bilstm_model"):
        super(BILSTMModelWithBert, self).__init__(name=name)

        # Load pre-trained embeddings (could be BERT or any other)
        self.embeddings = embeddings

        # BiLSTM Layer
        self.bilstm = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=False))
        self.bilstm1 = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=False))
        self.bilstm2 = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=False))

        # Dense and Output Layers
        self.dense_1 = layers.Dense(units=FFN_units, activation="relu")
        self.dropout = layers.Dropout(rate=dropout_rate)
        self.last_dense = layers.Dense(units=1, activation="sigmoid")

    def call(self, inputs, training=False):
        # Obtain embeddings from the pre-trained model
        input_ids , attention_mask = inputs
        outputs = self.embeddings(input_ids, attention_mask=attention_mask)
        bert_output = outputs.last_hidden_state

        # Apply BiLSTM layer
        x = self.bilstm(bert_output)
        x = self.bilstm1(x)
        x = self.bilstm2(x)

        # Dense layer and dropout for regularization
        x = self.dense_1(x)
        x = self.dropout(x, training=training)

        # Output layer
        output = self.last_dense(x)

        return output

class ModelManager:
    def __init__(self, model):
        self.model = model

    def train_test_split(self,inputs,labels):
        """70-30 split"""
        print("Spliting the Dataset")
        dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))

        # Split the dataset into training and testing
        train_size = int(0.7 * len(labels))  # 70% for training
        train_dataset = dataset.take(train_size)
        test_dataset = dataset.skip(train_size)

        train_dataset = train_dataset.batch(batch_size)
        test_dataset = test_dataset.batch(batch_size)
        return train_dataset, test_dataset

    def setup(self,tokens,labels,file_url,save, name, train=False):
        """setup subword bilstm"""
        print("Compliling Model")
        self.model.compile(
          loss="binary_crossentropy",
          optimizer="adam",
          metrics=["accuracy", tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
        )

        train_dataset, test_dataset = self.train_test_split(tokens,labels)

        if not train and self.download_file(file_url,save):
            self.initialize_model(train_dataset)
            self.load_weights(name)
            self.evaluate(test_dataset)
            return None
        else:
            history = self.train(train_dataset, test_dataset)
            self.evaluate(test_dataset)

            return history



    def train(self, train_dataset, test_dataset):
        """Train the model"""
        print("Training the Model")
        history = self.model.fit(
            train_dataset,
            epochs = epoch,
            validation_data=test_dataset
        )
        return history

    def evaluate(self , test_dataset):
        """Evaluate"""
        print("Evalaute the model")
        self.model.evaluate(test_dataset, batch_size=batch_size)

    def download_file(self, url, save_path):
        """Download a file from a URL and unzip it if it is a zip file."""

        if os.path.exists(save_path):
          print(f"File already exists at {save_path}. Skipping download.")
          return True

        response = requests.get(url, stream=True, timeout=50)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("File downloaded successfully")

            # Check if the file is a zip file
            if save_path.endswith('.zip'):
                # Unzip the file
                with zipfile.ZipFile(save_path, 'r') as zip_ref:
                    # Extract all the contents into directory same as save_path
                    extract_path = os.path.dirname(save_path)
                    zip_ref.extractall(extract_path)
                    print("File unzipped successfully in directory:", extract_path)

            return True
        else:
            print("Failed to download file, status code:", response.status_code)
            return False

    def initialize_model(self, dataset):
        """Initialize the model with dummy data to build its layers."""
        print("Initializing the Model")
        for sample_batch in dataset.take(1):
          self.model(sample_batch[0])
          break

    def load_weights(self, file_path):
        """Load weights into the model."""
        try:
            self.model.load_weights(file_path)
            print("Weights loaded successfully")
        except Exception as e:
            print("Failed to load weights:", e)

class TopicModel:
    """Class for extracting topics using LDA with a focus on specific categories."""

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.category_keywords = {
            'plot': ['plot', 'storyline', 'narrative', 'story', 'subplot', 'twist'],
            'actor': ['actor', 'actress', 'performance', 'cast', 'star', 'role', 'character', 'performer', 'leading', 'supporting'],
            'music': ['score', 'soundtrack', 'composer', 'theme', 'song', 'music', 'musical'],
            'director': ['director', 'directed', 'filmmaker', 'visionary', 'producer', 'screenwriter', 'auteur', 'helmer', 'directorial']
        }

    def setup_and_train(self, url, words=5, num_topics=10, passes=10):
        """Prepare data, enrich with keywords, and run the LDA model."""
        self.prepare_data()
        self.enrich_with_keywords()
        if self.download_and_extract_lda(url):
            self.model = gensim.models.LdaModel.load('lda/lda.model')
            self.dictionary = corpora.Dictionary(self.flat_clauses)
            self.corpus = [self.dictionary.doc2bow(text) for text in self.flat_clauses]
        else:
            self.model, self.dictionary, self.corpus = self.run(num_topics, passes)
        self.categorized_topics = self.categorize_topics(self.model.print_topics(num_words=words))
        self.topic_mapping = self.topic_mappings()
        self.doc_topics = [self.model.get_document_topics(bow) for bow in self.corpus]
        return self.categorized_topics

    def topic_mappings(self):
        """Create a mapping from topic index to category based on LDA results."""
        topic_mapping = {}
        for topic in self.categorized_topics:
            topic_mapping[topic[0][0]] = topic[1]
        return topic_mapping

    def prepare_data(self):
        """Tokenize each clause and flatten the data structure."""
        self.flat_clauses, self.flat_labels, self.clause_count = [], [], []
        for i, sentences in enumerate(self.data):
            for clauses in sentences:
                for clause in clauses:
                    self.flat_clauses.append(word_tokenize(clause))
                    self.flat_labels.append(self.labels[i])
            self.clause_count.append(len(clauses))

    def enrich_with_keywords(self):
        """Increase frequency of keywords in the text to emphasize them in the LDA analysis."""
        enriched_text = []
        for clause in self.flat_clauses:
            enriched_clause = []
            for token in clause:
                token_added = False
                for keywords in self.category_keywords.values():
                    if token in keywords:
                        enriched_clause.extend([token] * min(10, len(keywords)))  # Limiting keyword repetition
                        token_added = True
                        break
                if not token_added:
                    enriched_clause.append(token)
            enriched_text.append(enriched_clause)
        self.review_clauses = self.flat_clauses
        self.flat_clauses = enriched_text  # Update flat_clauses with enriched text

    def download_and_extract_lda(self, url, extract_to='.'):
        """Download an LDA model zip file from a URL and extract it."""
        # Make the HTTP request to download the file
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            zip_path = os.path.join(extract_to, 'lda.zip')
            # Write the downloaded file to a new file locally
            with open(zip_path, 'wb') as file:
                file.write(response.content)

            # Extract the zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)

            print(f"Model extracted to {extract_to}")
            return True
        else:
            print(f"Failed to download the file. Status code: {response.status_code}")
            return False

    def categorize_topics(self, topics):
        """Categorize LDA topics based on predefined category keywords."""
        topic_categories = []
        for topic in topics:
            words = [word.split('*')[1].replace('"', '').strip() for word in topic[1].split('+')]
            category_count = {category: sum(word in words for word in keywords) for category, keywords in self.category_keywords.items()}
            best_category = max(category_count, key=category_count.get)
            topic_categories.append((topic, best_category, category_count[best_category]))
        return topic_categories

    def run(self, num_topics, passes):
        """Run the LDA model to extract topics."""
        dictionary = corpora.Dictionary(self.flat_clauses)
        corpus = [dictionary.doc2bow(text) for text in self.flat_clauses]
        model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=passes)
        return model, dictionary, corpus

    def category_counter(self):
        """Count topics by Category."""
        category_count = collections.Counter(topic[1] for topic in self.categorized_topics)
        return category_count

    def aspects(self):
        """Extract aspect categories with the highest probability for each document."""
        aspects = []
        for doc in self.doc_topics:
            highest_prob = {}
            for topic in doc:
                category = self.topic_mapping[topic[0]]
                prob = topic[1]
                if category not in highest_prob or prob > highest_prob[category]:
                    highest_prob[category] = prob
            aspects.append([(cat, highest_prob[cat]) for cat in sorted(highest_prob)])
        return aspects

    def max_aspect(self, doc_topics):
        """Find the maximum aspect category from document topics based on the probability."""
        highest_prob = {}
        for topic, prob in doc_topics:
            category = topic
            if category not in highest_prob or prob > highest_prob[category]:
                highest_prob[category] = prob
        max_category = max(highest_prob, key=highest_prob.get)
        return max_category, highest_prob[max_category]

def main():
    """main function"""
    dataset_url="https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    subword_url="https://github.com/kgoel59/models/raw/main/subword_vocab.subwords" 
    bilstm_weights ='https://github.com/kgoel59/models/raw/main/bilstm.h5'
    dcnn_weights = 'https://github.com/kgoel59/models/raw/main/dcnn.h5'
    clauses = 'https://media.githubusercontent.com/media/kgoel59/models/main/clauses.csv'
    lda_url = 'https://github.com/kgoel59/models/raw/main/lda.zip'

    Setup.section("Assignment 2 - Group AT")
    Setup()

    Setup.section("Loading IMDB Dataset")
    manager = DatasetManager(url=dataset_url,extract_to= "imdb_dataset")
    Xy = manager.load_dataset()

    print("Using first 1000 rows")
    Xy = Xy.head(1000)

    Setup.section("Processing IMDB Dataset")
    text_processor = TextPreprocessor()
    print("Cleaning Reviews for Sentiment Analysis")
    clean_reviews = Xy['review'].apply(text_processor.denoise_and_standardize_text)
    sentiment_labels = Xy['sentiment']

    Setup.section("Tokenization")
    tokenizer = Tokenizer(clean_reviews, subword_url)
    subword_tokens = tokenizer.get_subword_tokens(clean_reviews)
    bert_tokens = tokenizer.get_bert_tokens(clean_reviews)

    Setup.section("DCNN Model Creation")
    dcnn = DCNNWithSubWord(tokenizer.subword_tokenizer.vocab_size)
    manager1 = ModelManager(dcnn)
    inputs = tf.constant(np.array(subword_tokens))
    labels = tf.constant(np.array(sentiment_labels))
    manager1.setup(inputs,labels, file_url=dcnn_weights, save='dcnn.h5', name='dcnn.h5', train=False)

    manager1.model.summary()

    Setup.section("BILSTM Model Creation")
    bilstm = BILSTMWithSubWord(tokenizer.subword_tokenizer.vocab_size)
    manager2 = ModelManager(bilstm)
    inputs = tf.constant(np.array(subword_tokens))
    labels = tf.constant(np.array(sentiment_labels))
    manager2.setup(inputs,labels, file_url=bilstm_weights, save='bilstm.h5', name='bilstm.h5', train=False)

    manager2.model.summary()

    Setup.section("Topic Modeling For Aspects")
    clauses_text,clauses_labels  = text_processor.load_clauses(url=clauses,data=Xy['review'].tolist(),labels=Xy['sentiment'].tolist())
    
    lda_model = TopicModel(data=clauses_text, labels=clauses_labels)
    lda_model.setup_and_train(lda_url,words=5, num_topics=10, passes=10)    
    aspects = [lda_model.max_aspect(aspect)[0] for aspect in lda_model.aspects()]  

    print(lda_model.category_counter())
    # Print lengths of various data structures and results
    print(len(lda_model.flat_clauses), len(lda_model.flat_labels), len(aspects),  len(lda_model.clause_count))

    Setup.section("Aspect Based Sentiment Analysis")
    subword_tokens_lda = tokenizer.get_subword_tokens([' '.join(clause) for clause in lda_model.review_clauses])
    inputs = tf.constant(np.array(subword_tokens_lda))
    y_pred = [0 if pred < 0.5 else 1 for pred in manager2.model.predict(inputs)]
    print(f"ABSA FINAL SCORE f{accuracy_score(y_pred, lda_model.flat_labels)}")



if __name__ == "__main__":
    main()
