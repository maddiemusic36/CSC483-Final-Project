import collections
import math
import os
import re
import spacy
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT
from whoosh.qparser import QueryParser


class JeopardyIR:
    # spaCy's natural language processing model for english
    nlp = spacy.load('en_core_web_sm')
    # english stopwords built into spaCy
    stopwords = nlp.Defaults.stop_words

    def __init__(self, folder):
        # index made by whoosh
        schema = Schema(title=TEXT(stored=True), body=TEXT(stored=True))
        self.index = create_in("index", schema)
        self.writer = self.index.writer()

        # loop through all wiki files
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            with open(filepath, "r", encoding='utf-8') as f:
                content = f.read().split("\n\n\n")
                # loop through each wiki article in the current file
                for doc in content:
                    self.clean_doc(doc)
                    break #TODO: remove later, this is for developing
            break #TODO: remove later, this is for developing

    """
    Process the given plaintext of a document. Lowercase, extract the title,
    remove punctuation/characters, remove stop words, etc. Then add the cleaned
    information to the index. 
    """
    def clean_doc(self, document):
        # extract title (no preprocessing yet)
        original_title = re.match(r'\[\[(.*?)\]\]', document)
        if original_title:
            # lowercase
            document = document.lower()
            # remove puctuation
            document = re.sub(r'[^\w\s]', '', document)
            ########


        # add cleaned information to index
        # self.writer.add_document(title=title, body=body)
        # self.writer.commit()  # not sure if this is needed????



def main():
    ir = JeopardyIR("articles")


main()