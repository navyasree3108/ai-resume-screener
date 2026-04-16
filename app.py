from pyngrok import ngrok
ngrok.set_auth_token("3COtgrJgW6ziDE3YjlNeasTY41k_4HhNgKaijZNjNWRbaohau")
from flask import Flask, request
from pyngrok import ngrok
import nltk
nltk.download('stopwords')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

app = Flask(__name__)

# -------------------------------
# Preprocess
# -------------------------------
def preprocess(text):
    stop_words = set(stopwords.words('english'))
    words = text.lower().split()
    filtered = [word for word in words if word not in stop_words]
    return " ".join(filtered)

# -------------------------------
# Home Page UI
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        job = preprocess(request.form["job"])
        res1 = preprocess(request.form["res1"])
        res2 = preprocess(request.form["res2"])

        docs = [job, res1, res2]

        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform(docs)

        scores = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()

        return f"""
        <h2>📊 Results</h2>
        <p>Candidate 1 Match: {scores[0]*100:.2f}%</p>
        <p>Candidate 2 Match: {scores[1]*100:.2f}%</p>
        <br><a href="/">Go Back</a>
        """

    return """
    <h1>🤖 AI Resume Screener</h1>
    <form method="post">
        <h3>Job Description</h3>
        <textarea name="job" rows="4" cols="50"></textarea>

        <h3>Candidate 1 Resume</h3>
        <textarea name="res1" rows="4" cols="50"></textarea>

        <h3>Candidate 2 Resume</h3>
        <textarea name="res2" rows="4" cols="50"></textarea>

        <br><br>
        <input type="submit" value="Analyze">
    </form>
    """

# -------------------------------
# Run App with ngrok
# -------------------------------
public_url = ngrok.connect(5000)
print("🌐 Open this link:", public_url)

app.run(port=5000)
