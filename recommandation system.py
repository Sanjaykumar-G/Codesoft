import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Sample user-item matrix (rows are users, columns are items)
# In a real-world scenario, you'd likely have a larger matrix from a dataset
user_item_matrix = np.array([
    [1, 0, 3, 4, 0],
    [2, 0, 4, 5, 1],
    [0, 2, 0, 3, 4],
    [4, 1, 5, 0, 2],
    [0, 5, 0, 2, 0],
])

# Function to perform collaborative filtering
def collaborative_filtering(user_item_matrix):
    # Compute cosine similarity between users
    user_similarity = cosine_similarity(user_item_matrix)

    # Set diagonal elements to zero (self-similarity)
    np.fill_diagonal(user_similarity, 0)

    # Find the most similar user for each user
    most_similar_users = np.argmax(user_similarity, axis=1)

    # Recommend items based on the most similar user's preferences
    recommendations = np.zeros_like(user_item_matrix)
    for user in range(user_item_matrix.shape[0]):
        non_zero_indices = np.where(user_item_matrix[user] == 0)[0]
        for item in non_zero_indices:
            most_similar_user = most_similar_users[user]
            recommendations[user, item] = user_item_matrix[most_similar_user, item]

    return recommendations

# Get recommendations for each user
recommendations = collaborative_filtering(user_item_matrix)

# Display the original user-item matrix and recommendations
print("Original User-Item Matrix:")
print(user_item_matrix)
print("\nRecommendations:")
print(recommendations)
