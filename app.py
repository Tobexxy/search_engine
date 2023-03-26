import openai
import pandas as pd
import numpy as np
from config import OPENAI_API_KEY
from openai.embeddings_utils import get_embedding, cosine_similarity

openai.api_key = OPENAI_API_KEY

from flask import Flask, request, render_template

app = Flask(__name__)
app.debug = True

# items = ['apple', 'banana', 'orange']


@app.route('/')
def search_form():
  return render_template('search_form.html')

@app.route('/search')
def search():
    # Get the search query from the URL query string
    query = request.args.get('query')

    search_term_vector = get_embedding(query, engine='text-embedding-ada-002')

    df = pd.read_csv('C:/Users/Dell 2023/Downloads/embeddin.csv')
    df['researchinterest_embedding'] = df['researchinterest_embedding'].apply(eval).apply(np.array)
    df['publication_embedding'] = df['publication_embedding'].apply(eval).apply(np.array)
    df['description_embedding'] = df['description_embedding'].apply(eval).apply(np.array)
    df['lecturer_embedding'] = df['lecturer_embedding'].apply(eval).apply(np.array)
    df['similarities_researchinterest'] = df['researchinterest_embedding'].apply(lambda x: cosine_similarity(x, search_term_vector))
    df['similarities_publication'] = df['publication_embedding'].apply(lambda x: cosine_similarity(x, search_term_vector))
    df['similarities_description'] = df['description_embedding'].apply(lambda x: cosine_similarity(x, search_term_vector))
    df['similarities_lecturer'] = df['lecturer_embedding'].apply(lambda x: cosine_similarity(x, search_term_vector))

    # Calculate weighted average of similarities
    df["similarity"] = (df["similarities_researchinterest"] * 0.5) + (df["similarities_description"] * 0.3) + (df["similarities_publication"] * 0.2)
    sorted_by_similarity = df.sort_values("similarity", ascending=False).head(3)

    #sorted_by_similarity = df[['researchinterest', 'similarities_researchinterest', 'publication', 'similarities_publication', 'description', 'similarities_description', 'lecturer', 'similarities_lecturer']].melt(id_vars=['researchinterest', 'publication', 'description', 'lecturer'], value_vars=['similarities_researchinterest', 'similarities_publication', 'similarities_description', 'similarities_lecturer'], var_name='similarity_type', value_name='similarities').sort_values('similarities', ascending=False).head(3)
   
    results = sorted_by_similarity[['lecturer', 'description','researchinterest']].values.tolist()

    # Render the search results template, passing in the search query and results
    return render_template('search_results.html', query=query, results=results)
   
@app.route('/static/<path:filename>')
def serve_static(filename):
  return app.send_static_file(filename)

if __name__ == '__main__':
  app.run()