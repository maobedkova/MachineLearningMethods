import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import  optimize
import matplotlib.patches as mpatches
from wordcloud import WordCloud


df = pd.read_csv("train.txt")
df.drop(["Unnamed: 0", "Unnamed: 0.1"], axis=1, inplace=True)

# First graph

plt.figure()
df.groupby(["Annotation"]).count().Tweet.plot(kind='bar', title="Distribution of classes")
plt.xlabel("Polarity")
# plt.show()
plt.savefig('classes.png')

# Second graph

link = df[df["Link"] == True].groupby("Annotation", as_index=False).count().Tweet

retweet = df[df["retweet"] == True].groupby("Annotation", as_index=False).count().Tweet

sumup = pd.concat([link, retweet], axis=1, keys=["link", "retweet"])
sumup = sumup.rename(index={0: "neg", 1: "obj", 2: "pos"})

fig = plt.figure() # Create matplotlib figure
plt.title("Retweets vs. Citing")

ax = fig.add_subplot(111) # Create matplotlib axes
ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.

sumup.link.plot(kind='bar', color='#4AB57C', ax=ax, width=0.4, position=1)
sumup.retweet.plot(kind='bar', color='#4A67B5', ax=ax2, width=0.4, position=0)

l = mpatches.Patch(color='#4AB57C', label='link')
r = mpatches.Patch(color='#4A67B5', label='retweet')

plt.legend(handles=[l, r])

# plt.show()
plt.savefig('link_retweet.png')

# Third graph

text = ""
for sth, line in df.Tweet.iteritems():
    text += line + " "

wc = WordCloud(background_color="white").generate(text)
plt.imshow(wc, cmap=plt.cm.gray, interpolation='bilinear')
plt.axis("off")
# plt.show()
wc.to_file("wordcloud.png")



#
#corr = {"obj": 0., "pos": 1., "neg": 1.}
#df.replace({"Annotation": corr},  inplace=True)
#
#print(df.groupby(["No_Exc",  "Annotation"]).count().Tweet)
#
#exc = np.array(df.No_Exc)
#que = np.array(df.No_Que)
#ann = np.array(df.Annotation)
#
#plt.figure()
#plt.scatter(exc,  ann,  marker='^')
#plt.show()
#
#fitfunc = lambda p, ex: p[0]+ p[1]*ex
#error = lambda  p, ex, an: fitfunc(p, ex) - an
#
#p1, success = optimize.leastsq(error, args=(exc, ann))

