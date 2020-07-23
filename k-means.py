import pandas as pd

import requests
import io

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

plt.style.use('seaborn')

# load in dataset
url = "https://raw.githubusercontent.com/hip-hipJorge/K-Means_Clustering/master/seeds_dataset.txt"
read_data = requests.get(url).content
seed_df = pd.read_csv(io.StringIO(read_data.decode('utf-8')),
                      delimiter='\t', names=['area', 'perimeter', 'compactness', 'length_of_kernel',
                                             'width_of_kernel', 'asymmetry_coefficient', 'length_of_kernel_groove',
                                             'target'])


# helper functions
def preprocess(df_):
    # Drop of classifier
    df_ = df_.iloc[:, 0:7]
    # scale the data to better cluster and plot
    scaler = MinMaxScaler()
    scaler.fit(df_)
    new_df = pd.DataFrame(scaler.transform(df_), columns=df_.columns)
    return new_df


def report(k):
    print("\n\t\tFinal Report")
    print("-----------------------------------------")
    print("| Experiment # | Value of k | SSE Value |")
    print("-----------------------------------------")
    for i in range(len(k)):
        print("|\t{}          |\t{}       |\t{}     |".format(i, i+1, round(k[i], 1)))


def sse(df, centroid):
    # define k
    k = centroid.shape[0]
    # create a list of error
    error_list = []
    for i in range(df.shape[0]):
        # calculate error for each feature
        instance_error = 0
        for j in range(df.shape[1]):
            min_error = 1
            # select the closest centroid to compare for m feature
            for m in range(k-1):
                min_error = min(min_error, abs(df.iloc[i, j]-centroid[m, j]))
            instance_error = instance_error + min_error
        # add the calculated error for i instance
        error_list.append(instance_error)
    # calculate error mean
    mean = sum(error_list) / df.shape[0]
    return pow(mean, 2)


def k_selection(df, epsilon=0, max_iteration=10):
    # look for best k value until reached max_iterations or error (sse) = epsilon
    error = 1
    iterations = 0
    # insert sse into k_error at each cluster
    k_error = []
    while error > epsilon and iterations < max_iteration:
        iterations += 1
        kmeans = KMeans(n_clusters=iterations)
        kmeans.fit_predict(df)
        centroids = kmeans.cluster_centers_
        error = sse(df, centroids)
        k_error.append(error)
    return k_error


# main execution
def main():
    # prepare dataset, remove classification and scale data
    x = preprocess(seed_df)
    # gather clusters error
    k_results = k_selection(x)
    # plot k results
    plt.plot(list(range(1, 11)), k_results)
    plt.title('Elbow Method')
    plt.ylabel('sum of squared errors')
    plt.xlabel('num of clusters (k)')
    plt.show()
    # results
    report(k_results)


if __name__ == '__main__':
    main()
