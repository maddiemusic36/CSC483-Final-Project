import os
import json
import spacy

def main():
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

# if __name__ == "__main__":
#     main()
