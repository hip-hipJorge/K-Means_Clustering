
import pandas as pd
import requests
import io
import matplotlib.pyplot as plt
from matplotlib import dates as mpl_dates


plt.style.use('seaborn')


# helper functions
def preprocess(df_):
    df_['status_published'] = pd.to_datetime(df_['status_published'])
    df_ = df_.drop(df_.columns[[12, 13, 14, 15]], axis=1)
    return df_


# load in dataset
url="https://raw.githubusercontent.com/hip-hipJorge/K-Means_Clustering/master/FB-Live.csv"
read_data=requests.get(url).content
fb_live = pd.read_csv(io.StringIO(read_data.decode('utf-8')))

# ready dataset
fb_live = preprocess(fb_live)

x = fb_live.iloc[0:15, 2]
y_all = fb_live.iloc[0:15, 3]
y_cmnts = fb_live.iloc[0:15, 4]
y_shares = fb_live.iloc[0:15, 5]
y_likes = fb_live.iloc[0:15, 6]
y_loves = fb_live.iloc[0:15, 7]
y_wows = fb_live.iloc[0:15, 8]
y_hahas = fb_live.iloc[0:15, 9]
y_sads = fb_live.iloc[0:15, 10]
y_angrys = fb_live.iloc[0:15, 11]

plt.plot_date(x, y_all, c='b', label='reactions')
plt.plot_date(x, y_cmnts, c='g', label='comments')
plt.plot_date(x, y_shares, c='r', label='shares')
plt.plot_date(x, y_likes, c='c', label='likes')
plt.plot_date(x, y_loves, c='m', label='loves')
plt.plot_date(x, y_wows,c='y', label='wows')
plt.plot_date(x, y_hahas, c='k', label='hahas')
plt.plot_date(x, y_sads, c='burlywood', label='sads')
plt.plot_date(x, y_angrys, c='chartreuse', label='angrys')


plt.legend(loc=2)
plt.gcf().autofmt_xdate()
date_format = mpl_dates.DateFormatter('%b %d, %a')
plt.gca().xaxis.set_major_formatter(date_format)
plt.title('Date vs. # of Reactions')
plt.xlabel('Date (2018)')
plt.ylabel('# of FB reactions')
plt.show()

'''

# load in iris data
iris = datasets.load_iris()
X = scale(iris.data)
# target
y = pd.DataFrame(iris.target)

variable_names = iris.feature_names

# Building and running the model
clustering = KMeans(n_clusters=3, random_state=5)
clustering.fit(X)

# Plotting your model outputs
iris_df = pd.DataFrame(iris.data)
iris_df.columns = ['Sepal Length', 'Sepal_width', 'Petal_Length', 'Petal_Width']
y.columns = ['Targets']

color_theme = np.array(['darkgray', 'lightsalmon', 'powderblue'])

plt.subplot(1,2,1)
plt.scatter(x=iris_df.Petal_Length, y=iris_df.Petal_Width, c=color_theme[iris.target], s=50)
plt.title('Ground Truth classification')


plt.subplot(1,2,2)
plt.scatter(x=iris_df.Petal_Length, y=iris_df.Petal_Width, c=color_theme[clustering.labels_], s=50)

plt.title('K-Means Classification')
plt.show()

relabel = np.choose(clustering.labels_,[2,0,1]).astype(np.int64)

plt.subplot(1,2,1)
plt.scatter(x=iris_df.Petal_Length, y=iris_df.Petal_Width, c=color_theme[iris.target], s=50)
plt.title('Relabeld Ground Truth classification')


plt.subplot(1,2,2)
plt.scatter(x=iris_df.Petal_Length, y=iris_df.Petal_Width, c=color_theme[clustering.labels_], s=50)

plt.title('Relabeled K-Means Classification')
plt.show()

# Evaluate the cluster results
print(classification_report(y, relabel))
print(mean_squared_error(y, relabel))
'''