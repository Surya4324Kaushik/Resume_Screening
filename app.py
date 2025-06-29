import streamlit as st
import pickle
import re
import nltk


nltk.download('punkt')
nltk.download('stopwords')


clf=pickle.load(open('clf.pkl','rb'))
tfidfd=pickle.load(open('tfidf.pkl','rb'))


def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)  
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText) 
    cleanText = re.sub('\s+', ' ', cleanText)

    return cleanText



def main():
    st.title("Resume Screening App")
    upload_file=st.file_uploader('Upload Resume',type=['text','pdf'])

    if upload_file is not None:
        try:
            resume_bytes=upload_file.read()
            resume_text=resume_bytes.decode('utf-8')

        except UnicodeDecodeError:
            
            resume_text=resume_bytes.decode('latin-1')

        cleaned_resume=cleanResume(resume_text)

        input_features = tfidfd.transform([cleaned_resume]).toarray()


        prediction_id=clf.predict(input_features)[0]

        st.write(prediction_id)

        #Category Mapping
        category_mapping={

        15:"Java Developer",
        23:"Testing",
        8:"Devops Engineer",
        20:"Python Developer",
        24:"Web Designing",
        13:"Hadoop",
        3:"Blockchain",
        10:"Etl Developer",
        18:"Operations Manager",
        0:"Data Science",
        22:"Sales",
        16:"Mechanical Engineer",
        1:"Arts",
        7:"Database",
        11:"Electrical Engineering",
        12:"Health and Fitness",
        19:"PMO",
        4:"Business Analyst",
        9:"Dotnet Developer",
        2:"Automation Testing",
        17:"Network Security Engineer",
        21:"SAP Developer",
        5:"Civil Engineer",
        6:"Advocate",
                                }
    
        category_name = category_mapping.get(prediction_id,"Unknown")
        st.write("Predicted Category :",category_name)

if __name__=="__main__":
    main()