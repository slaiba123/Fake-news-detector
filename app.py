from flask import Flask, render_template, request
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))
vectorizer = pickle.load(open('vector.pkl','rb'))

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(content):
    processed_content = re.sub('[^a-zA-Z]', ' ', content)
    processed_content = processed_content.lower()
    words = processed_content.split() 
    stemmed_words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(stemmed_words) 

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news_content = request.form['content']
        # Preprocess the input article
        processed_content = preprocess_text(news_content)
        transformed_content = vectorizer.transform([processed_content]) # Transform the processed content using the TF-IDF vectorizer
        prediction = model.predict(transformed_content) # Make a prediction using the trained model
        
        if prediction[0] == 0:
            result = 'The News seems Real'
        else:
            result = 'The News seems Fake'
        return render_template('index.html', prediction=result) # Display the result on the webpage

if __name__ == '__main__':
    app.run(debug=True)
