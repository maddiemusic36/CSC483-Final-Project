# CSC483-Final-Project

### CSC483 Text Retrieval and Web Search final project submission

### Authors: Madeline DeLeon and Benjamin Curtis

Jeopardy! - Given a question and its topic, can our machine successfully return
the related Wikipedia article with at least 0.4 precision and 0.5 MRR?


\* The questions were extracted from j-archive.com from previous shows between
2013-01-01 and 2013-01-07


# Running the project

1. Create a virtual environment with `python -m venv env`
2. Source the environment. For Windows: `./env/Scripts/activate`
3. Install the dependencies `pip install -r requirements.txt`
4. Download the nlp `python -m spacy download en_core_web_sm`
5. Download the processed JSON files into a folder named `cleaned_articles`. The link for these will be in the project report.

Now you should be good to run the `jeopardy_engine.py` file!