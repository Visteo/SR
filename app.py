import pandas as pd
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import pickle

# Load Dataset
data = pd.read_csv('D:/movie_recommendation_project/processed_filmtv_movies.csv')

# Handle missing values
data = data.dropna(subset=['genre', 'title'])

# Encode target (genre)
label_encoder = LabelEncoder()
data['genre_encoded'] = label_encoder.fit_transform(data['genre'])

# Train-Test Split
X = data['title']  # Use 'description' as input feature
y = data['genre_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Pipeline (TF-IDF + Random Forest)
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
pipeline.fit(X_train, y_train)

# Flask App
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    genre = request.form['genre']
    # Decode genre untuk mencocokkan dengan dataset
    genre_decoded = label_encoder.inverse_transform([int(genre)])[0]
    # Filter dataset berdasarkan genre dan urutkan berdasarkan avg_vote (tertinggi ke terendah)
    recommendations = data[data['genre'] == genre_decoded].sort_values(by='avg_vote', ascending=False)
    # Ambil 10 film dengan avg_vote tertinggi
    recommendations = recommendations[['title', 'year', 'avg_vote']].head(10)
    # Render hasil rekomendasi ke halaman recommend.html
    return render_template('recommend.html', genre=genre_decoded, recommendations=recommendations.to_dict(orient='records'))


@app.route('/predict', methods=['POST'])
def search_actor():
    actor_name = request.form['title']  # Input nama aktor dari form
    # Filter dataset berdasarkan aktor
    actor_movies = data[data['actors'].str.contains(actor_name, case=False, na=False)]
    # Urutkan hasil berdasarkan kolom 'avg_vote' dari tertinggi ke terendah
    actor_movies = actor_movies.sort_values(by='avg_vote', ascending=False)
    # Ambil 10 film teratas
    actor_movies = actor_movies[['title', 'year', 'genre', 'avg_vote']].head(10)
    # Jika tidak ada hasil
    if actor_movies.empty:
        return render_template('predict.html', message=f"Tidak ada film ditemukan untuk aktor '{actor_name}'.")
    # Kirim hasil ke halaman HTML
    return render_template('predict.html', actor=actor_name, movies=actor_movies.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
