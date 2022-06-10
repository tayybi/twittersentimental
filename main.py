import tweepy
from matplotlib import pyplot as plt
from textblob import TextBlob
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import re

# plt.style.use(fivethirtyeight)
# load the data

# twitter api credential
consumer_key = "2i1UshnWggD6GQ3EWDSlKFJPO"
consumer_secret = "L1NTjohjFWbLsmTW7sfR2a9Oap4qMex1uK6wQLdysMewnHyXn1"
access_token = "1492836101689593864-UsJxsn86Dva7p0zRehjvNkxe7lcWzP"
access_token_secret = "JEfPdTLdVx6fc8GPpyRnDk8CbNbTCRBeh4WoiCkpsU60q"

# create authenticate object
authenticate = tweepy.OAuthHandler(consumer_key, consumer_secret)
# set and access token
authenticate.set_access_token(access_token, access_token_secret)
# create api object
api = tweepy.API(authenticate, wait_on_rate_limit=True)

# extrect tweets from twitter
posts = api.user_timeline(screen_name="BillGates", count=100, tweet_mode="extended")

#  last five tweet
# print("Show last five tweet")
# i=1
# for tweet in posts[0:5]:
#     print(str(i) + ')' +tweet.full_text+ '\n')
#     i=i+1

# create datafram with a colum called
df = pd.DataFrame([tweet.full_text for tweet in posts], columns=['Tweets'])
print(df)


# clean texts /tweets

def cleanTxt(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)  # remove @mention
    text = re.sub(r'#', '', text)  # remove # symbol
    text = re.sub(r'@RT[\s]+', '', text)  # remove RT
    text = re.sub(r'https?:\/\/\S+', '', text)  # remove hyperlinks
    return text


# cleaning the text
df['Tweets'] = df['Tweets'].apply(cleanTxt)
print(df)


# create function to get subjectivity
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity


# create a function to get polarity
def getPolarity(text):
    return TextBlob(text).sentiment.polarity


# create two new colum
df['Subjectivity'] = df['Tweets'].apply(getSubjectivity)
df['Polarity'] = df['Tweets'].apply(getPolarity)
print(df)

# plot the world cloud
allWords = ''.join([twts for twts in df["Tweets"]])
wordCloud = WordCloud(width=500, height=300, random_state=21, max_font_size=119).generate(allWords)
plt.imshow(wordCloud, interpolation="bilinear")
plt.axis('off')
plt.show()

#  create a function to build positive negativeand nuteral tweets

def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'


df['Analysis'] = df['Polarity'].apply(getAnalysis)

#  show the data frame
print(df)

# plot the ploarity and subjectivity
plt.figure(figsize=(8, 6))
for i in range(0, df.shape[0]):
    plt.scatter(df['Polarity'][i], df['Subjectivity'][i], color='blue')

plt.title('Sentimental Analysis')
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
plt.show()

# get the positive tweets
j = 1
sorteddf = df.sort_values(by=['Polarity'])
for i in range(0, sorteddf.shape[0]):
    if sorteddf['Analysis'][i] == 'Positive':
        print(str(j) + ') ' + sorteddf['Tweets'][i])
        print()
        j = j + 1

# get percentage of positive tweets
ptweets=df[df.Analysis == 'Positive']
ptweets=ptweets['Tweets']
print(round((ptweets.shape[0]/df.shape[0])*100,1))

# get percentage of Negative tweets
ptweets=df[df.Analysis == 'Negative']
ptweets=ptweets['Tweets']
print(round((ptweets.shape[0]/df.shape[0])*100,1))

# Show the value counts
df['Analysis'].value_counts()

# plot and visulize the counts
plt.title("Sentiment Analysis")
plt.xlabel('Sentiment')
plt.ylabel('Count')
df['Analysis'].value_counts().plot(kind='bar')
plt.show()

