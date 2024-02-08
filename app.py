import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
df = pd.read_csv('train.csv')

# Encode the 'target' column into numerical labels
label_encoder = LabelEncoder()
df['target_label'] = label_encoder.fit_transform(df['target'])

# Select only the feature columns (T1 to T18)
features = df.drop(['target', 'target_label'], axis=1)

# Apply PCA to reduce dimensionality to 2 components
pca = PCA(n_components=2)
components = pca.fit_transform(features)

# Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(features)

# Assign clusters to each data point
df['cluster'] = kmeans.labels_

# Function to identify the cluster for a given data point
def identify_cluster(data_point):
    # Predict the cluster for the given data point
    cluster_index = kmeans.predict([data_point])[0]
    return cluster_index

# Streamlit app
st.title('# Resolute AI Intern Assignment Outputs')

# Task 1: Clustering
st.subheader('## Task 1: Clustering')

# User input for all 18 values
st.subheader('### Enter values for T1 to T18:')
values = []
for i in range(18):
    values.append(st.number_input(f'T{i+1}', step=1))

# Convert user input to a data point (list)
data_point = values

# Identify the cluster for the data point
cluster_index = identify_cluster(data_point)
st.write(f'The data point belongs to Cluster {cluster_index}')

# Plot clusters
plt.figure(figsize=(10, 6))

# Scatter plot each cluster
for cluster_number in range(3):
    cluster_indices = df[df['cluster'] == cluster_number].index
    plt.scatter(components[cluster_indices, 0], components[cluster_indices, 1], label=f'Cluster {cluster_number}')

# Plot user input
plt.scatter(components[-1, 0], components[-1, 1], color='red', label='User Input', marker='*', s=200)
plt.annotate('User Input', xy=(components[-1, 0], components[-1, 1]), xytext=(-20, 20),
             textcoords='offset points', arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

plt.title('Clustering Visualization')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.tight_layout()

# Show plot
st.pyplot(plt)

# Load predictions DataFrame
predictions_df = pd.read_csv('predicted target values for test set.csv')

# Task 2: Classification
st.subheader('## Task 2: Classification')

# Display predictions DataFrame
st.write("### Predictions of target values for the test set:")
st.write(predictions_df)

# Load the output.csv file
output_df = pd.read_csv('output.csv')

# Add a subheader for Task 3
st.subheader("## Task 3: Python")

# Display the DataFrame
st.write("### Results derived from rawdata:")
st.write(output_df)
