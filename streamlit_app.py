import streamlit as st
from text_exploration import *


                         # set parameters for tf-idf for unigrams and bigrams
tfidf_matrix = tv.fit_transform(norm_corpus)                                      # extract tfidf features from norm_corpus
tfidf_matrix.shape

doc_sim = cosine_similarity(tfidf_matrix)    # compute document similarity by examining the cosine similairty b/w documents in matrix
doc_sim_df = pd.DataFrame(doc_sim)                                                  # take doc_sim, convert to dataframe
doc_sim_df.head()

# Create a Streamlit app
st.title('Repository Recommender System')

# Input field for search query
search_query = st.text_input('Enter a search query:')

# Function to find similar repositories based on the search query
def query_repository_recommender(search_query, repository_list, tfidf_matrix, tv):
    try:
        # Transform the search query into its vector form
        query_vector = tv.transform([search_query])

        cosine_similarities = cosine_similarity(query_vector, tfidf_matrix)

        similar_repository_idxs = cosine_similarities[0].argsort()[-5:][::-1]

        similar_repositories = repository_list[similar_repository_idxs]

        return similar_repositories
    except Exception as e:
        return ["Error: " + str(e)]

# Button to trigger the search and display recommendations
if st.button('Find Similar Repositories'):
    query_recommendations = query_repository_recommender(search_query, repository_list, tfidf_matrix, tv)
    
    # Debugging print statements to check if variables are correct
    print("Search Query:", search_query)
    print("Recommendations:", query_recommendations)
    
    if "Error" in query_recommendations[0]:
        st.write("An error occurred:", query_recommendations[0])
    else:
        st.write("Based on your search query, I'd recommend checking out:")
        for repo in query_recommendations:
            st.write(repo)



