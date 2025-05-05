import os
import json
import spacy


"""
Go through all of the wiki files and process them into JSON. For each article,
extract the title and convert the rest into a cleaned list of tokens.
Args: None
Returns: None
"""
def wiki_files_to_json():
    # english natural language processor made by spaCy
    nlp = spacy.load('en_core_web_sm', disable=["ner", "parser"])
    
    # A list to store all article data as dictionaries
    articles = []
    # current document ID
    doc_id = 0
    # loop through all wiki files
    for filename in os.listdir("articles"):
        filepath = os.path.join("articles", filename)
        print("Reading file " + filename + "...")
        with open(filepath, "r", encoding='utf-8') as f:
            title = None
            body_lines = []

            # loop through each file line by line
            for line in f:
                line = line.strip()

                if line:
                    # if an article title is encountered
                    if line.startswith("[[") and line.endswith("]]"):
                        # we already have a title
                        if title:
                            body = " ".join(body_lines)
                            
                            # gets the lemma of each word
                            body = nlp(body)
                            # handles removing punctuation, extra spaces and stop words
                            body = [
                                token.lemma_.lower()
                                for token in body
                                if not token.is_punct and not token.is_space and not token.is_stop
                            ]

                            # add current article to list
                            articles.append({"id": doc_id, "title": title, "body": body})
                            doc_id += 1
                            # reset title
                            title = None
                        body_lines = []
                        title = line[2:-2]
                    # ignore redirect articles
                    elif "#redirect" in line.lower():
                        title = None
                        continue
                    # ignore section headers
                    elif not line.startswith('=='):
                        body_lines.append(line)
            # checks for edge case (last article in the file)
            if title:
                body = " ".join(body_lines)

                # gets the lemma of each word
                body = nlp(body)
                # handles removing punctuation, extra spaces and stop words
                body = [
                    token.lemma_.lower()
                    for token in body
                    if not token.is_punct and not token.is_space and not token.is_stop
                ]

                articles.append({"id": doc_id, "title": title, "body": body})
                doc_id += 1

    # write all articles to an output JSON file
    with open("cleaned_articles.json", "w", encoding='utf-8') as out_f:
        json.dump(articles, out_f, ensure_ascii=False, indent=4)

    print("All done :)", len(articles), "articles processed")


"""
Build a TF-IDF index from all of the article data. This uses the LNC LTN
formulas from Homework 3.
"""
def build_index(json_file):
    # extract article information from JSON file (if it exists)
    if os.path.exists(json_file):
        with open(json_file, "r", encoding='utf-8') as f:
            data = json.load(f)
            print(len(data)) # should be 137,471
    else:
        print("ERROR: file", json_file, "does not exist. Did you unzip it?")


def answer_questions(q_file):
    pass


def main():
    #### DO NOT RUN THIS ####
    #### This was used to create the cleaned_articles file
    #### It only needed to be ran once
    #wiki_files_to_json()

    # Construct the tf-idf index
    index = build_index("cleaned_articles.json")

    # Process the Jeopardy questions and evaluate the results
    answer_questions("questions.txt")


if __name__ == "__main__":
    main()
