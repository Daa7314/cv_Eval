#Import packages
import streamlit as st
#Statistical packages
import numpy as np
import pandas as pd
import requests
import statistics
#import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#Importing word document
import docx2txt
import docx
import pdfplumber


#word cloud
from pathlib import Path
#import matplotlib.pyplot as plt
from wordcloud import STOPWORDS, WordCloud



#Natural language processing packages
import nltk
from nltk.probability import FreqDist
from rake_nltk import Rake
nltk.download('stopwords')
nltk.download('punkt')
import spacy
from textblob import TextBlob
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer



st.header('Compare your Resume with a Job Listing')

st.title("Rename your resume as A and the Job description as B")
def main():
    menu = ["Image","Dataset","DocumentFiles","About"]
    choice = st.sidebar.selectbox("Menu",menu)

    if choice == "Image":
        st.subheader("Image")
    elif choice == "Dataset":
        st.subheader("Dataset")
    elif choice == "DocumentFiles":
        st.subheader("DocumentFiles")
        docx_files = st.file_uploader("Upload Document", type=["pdf","docx","txt"], accept_multiple_files=True)
        
        if st.button("Process"):
            data = []
            for docx_file in docx_files:
                if docx_file is not None:

                    file_details = {"filename":docx_file.name, "filetype":docx_file.type,
                                    "filesize":docx_file.size}
                    st.write(file_details)

                    if docx_file.type == "text/plain":
                        # Read as string (decode bytes to string)
                        raw_text = str(docx_file.read(),"utf-8")
                        data.append(raw_text)
                        

                    elif docx_file.type == "application/pdf":
                        try:
                            with pdfplumber.open(docx_file) as pdf:
                                pages = pdf.pages[0]
                                data.append(pages)
                        except:
                            st.write("None")

                    else:
                        raw_text = docx2txt.process(docx_file)
                        data.append(raw_text)
            
            count_vectorizer = CountVectorizer()
            vector_matrix = count_vectorizer.fit_transform(data)
            cosine_similarity_matrix = cosine_similarity(vector_matrix)
            vx = pd.DataFrame(cosine_similarity_matrix,['resume','jd_text'])
            st.write("Your resume and Job description match at: ", vx.loc["resume",1])

#JD keyword extraction
            tok_count= []
            tok_count.append(len(data[0]))
            tok_count.append(len(data[1]))
            if len(data[0]) > len(data[1]):
                text = data[1]
            else:
                text = data[0]

            rake_nltk_var = Rake()
            rake_nltk_var = Rake()
            rake_nltk_var.extract_keywords_from_text(text)
            keyword_extracted = rake_nltk_var.get_ranked_phrases()
            vectr = TfidfVectorizer()
            text3 = keyword_extracted
            keywrds = vectr.fit(text3)
            jd_keywords = list(keywrds.get_feature_names_out())
            st.write("The following is a list of keywords in the job description: ", jd_keywords)
                
            if len(data[0]) > len(data[1]):
                text = data[0]
            else:
                text = data[1]

            rake_nltk_var = Rake()
            rake_nltk_var = Rake()
            rake_nltk_var.extract_keywords_from_text(text)
            keyword_extracted = rake_nltk_var.get_ranked_phrases()
            vectr = TfidfVectorizer()
            text3 = keyword_extracted
            keywrds = vectr.fit(text3)
            x2 = list(keywrds.get_feature_names_out())
            missing_keywords = list(set(jd_keywords) - set(x2))
            st.write("These keywords dont appear in your resume: ", missing_keywords)
            final_file = [{"Missing Keywords From Resume":missing_keywords,  "Keywords found in Job description":jd_keywords}]
            #adding a download button to download csv file
            import json
            json_string = json.dumps(final_file)

            st.download_button( 

                label="Download data as CSV",

                data=json_string,

                file_name='Resume_exvaluation.txt')



if __name__ == '__main__':
	main()

st.subheader("About")
st.info("Built with Streamlit")
st.info("Jesus Saves @JCharisTech")
st.text("Jesse E.Agbe(JCharis)")
