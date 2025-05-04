import collections
from concurrent.futures import ProcessPoolExecutor
import math
import os
import re
import spacy
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser


class JeopardyIR:
    # spaCy's natural language processing model for english
    nlp = spacy.load('en_core_web_sm')
    # english stopwords built into spaCy
    stopwords = nlp.Defaults.stop_words
    num = 0

    def __init__(self, folder):
        # index made by whoosh
        schema = Schema(id=ID, title=TEXT(stored=True), body=TEXT)

        if not os.path.exists("index"):
            os.mkdir("index")

        self.index = create_in("index", schema)

        all_docs = []
        # loop through all wiki files
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            print("reading file", filename)
            with open(filepath, "r", encoding='utf-8') as f:
                all_docs += f.read().split("[[")
            break #TODO: remove this later

        print(len(all_docs))

        with self.index.writer() as writer:
            for doc in all_docs:
                title, body = self.clean_article(doc)
                if title and body:
                    writer.add_document(id=str(self.num), title=title, body=body)
                    self.num += 1
                 
        print("all done :)", self.num, "articles processed")
        

    """
    Process the given plaintext of a document. Extract the title, clean the
    rest of the document, then add the cleaned information to the index. 
    """
    def clean_article(self, document):
        # extract title (with no preprocessing)
        original_title = re.match(r'(.*?)\]\]', document)
        body = None
        if original_title:
            original_title = original_title.group().strip("[]")
            
            # add cleaned information to index
            body = self.clean(document)
        return original_title, body


    """
    Clean the given plaintext of a document. Lowercase, remove punctuation,
    remove stop words, etc.
    """
    def clean(self, document):
        # lowercase
        document = document.strip().lower()

        # ignore redirect articles
        if "#redirect" not in document:
            # remove puctuation
            document = re.sub(r'[^\w\s]', ' ', document)
            # remove web addresses
            document = re.sub(r'http\S+|www.\S+', '', document)
            # remove markdown '=='
            document = re.sub(r'==', '', document)
            # remove special whitespace characters
            document = re.sub(r'[\n\t\r\f\v]+', '', document)
            # remove extra spaces
            document = re.sub(r"\s+", " ", document)
            document = document.strip()

            # gets the lemma of each word in the string
            with self.nlp.select_pipes(disable=["ner", "parser"]):
                doc = self.nlp(document)

            # remove stop words and create list of tokens
            tokens = " ".join([w.lemma_ for w in doc if len(w.lemma_.strip()) and w.lemma_ not in self.stopwords])
            return tokens
        return None

def main():
    ir = JeopardyIR("articles")


main()