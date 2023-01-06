from flask import Flask, render_template,request
import pandas as pd
import PyPDF2
from pprint import pprint
from Questgen import main
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods = ['POST'])
def predict():
    json_result = []
    if request.method == 'POST':
        myfile = request.form['myfile']
        print("****************ARTICLE*********************")
        # creating a pdf file object
        pdfFileObj = open(myfile, 'rb')

        # creating a pdf reader object
        pdfReader = PyPDF2.PdfReader(pdfFileObj)

        # printing number of pages in pdf file
        print(len(pdfReader.pages))

        # creating a page object
        pageObj = pdfReader.pages[0]

        # extracting text from page
        ARTICLE = pageObj.extract_text()
        print(ARTICLE)

        # closing the pdf file object
        pdfFileObj.close()
        
        print("*****************summary_text********************")
        # payload = {"input_text": ARTICLE}
        # summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        summ = summarizer(ARTICLE, min_length=100, do_sample=False)
        
        print(summ[0]["summary_text"])
        print("*****************MCQ********************")
        payload = {"input_text": summ[0]["summary_text"]}
        qg = main.QGen()
        output = qg.predict_mcq(payload)
        colname = ['question_statement', 'MCQ']
        for count, i in enumerate(output['questions']):
            json_result.append({'question_statement': i['question_statement'], 'MCQ': i['options']})
        print (json_result)
    return render_template('result.html', prediction = json_result, colnames = colname, summary= summ[0]["summary_text"])


if __name__ == '__main__':
    app.run()
