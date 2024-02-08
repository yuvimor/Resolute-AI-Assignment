import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

# Load the dataset
df = pd.read_csv('your_dataset.csv')

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

# Define a function to identify the cluster for a given data point
def identify_cluster(data_point):
    # Predict the cluster for the given data point
    cluster_index = kmeans.predict([data_point])[0]
    return cluster_index

# Streamlit app
st.title('Cluster Identification App')

# User input for all 18 values
st.subheader('Enter values for T1 to T18:')
values = []
for i in range(18):
    values.append(st.number_input(f'T{i+1}', step=1))

# Convert user input to a data point (list)
data_point = values

# Button to trigger cluster identification
if st.button('Identify Cluster'):
    # Identify the cluster for the data point
    cluster_index = identify_cluster(data_point)
    st.write(f'The data point belongs to Cluster {cluster_index}')
