from flask import Flask,render_template,url_for,request
from google_play_scraper import app as gs
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import json
from nltk.util import print_string


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    scoreList = []
    
    # print(request.form)
    dict = request.form
    for key, value in dict.items():
        print(f'{key} {value}')
        score=0
    # if request.method == 'POST':
        url = value 
        #request.form['urla']
        file = open("cc.txt", "w",encoding='utf-8')
        link=url
        findId=link.find('id=')

        url=link[findId+3:]
        result = gs(
            url,
            lang='en', # defaults to 'en'
            country='us' # defaults to 'us'
        )
        file.write(str(result))
        file.close()
        
        myfile=[]
        with open("cc.txt",encoding='utf8') as mydata:
            for data in mydata:
                myfile.append(data)
        
        filename = f'{key}.json'
        with open(filename, 'w') as fp:
            json.dump(result, fp)

        start=myfile[0].find('description')
        end=myfile[0].find('editorsChoice')
        appName=result.get('title')
        c=data[start:end]

        cleandata = re.sub('[^A-Za-z0-9]+',' ',c)
        low=cleandata.lower()

        stop=set(stopwords.words('english'))
        wordstoken=word_tokenize(low)

        sentences=[w for w in wordstoken if not w in stop]
        sentences=[]


        for w in wordstoken:
            if w not in stop:
                sentences.append(w)

        total=0
        tot=0
        positive = open("positive.txt", "r",encoding='utf-8')
        negative = open("negative.txt", "r",encoding='utf-8')
        pos=positive.read().split()
        neg=negative.read().split()
        for word in sentences:
        #     print(word)
            tot=tot+1
            if word in pos:
                total=total+1
        #         print("good: "+word)
            if word in neg:
                total=total-1
        #         print("bad: "+word)

        print(f'{total} {tot}')   
        score=total/tot
        scoreList.append([score, appName])
        print(appName)
        print(score)
    scoreList.sort(reverse=True, key=lambda x:x[0])
    for each in scoreList:
        print(each)
    

    return render_template('result.html',scoreList = scoreList)



if __name__ == '__main__':
	app.run(debug=True, port=4000)