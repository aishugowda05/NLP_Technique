import pandas as pd

# Load the new dataset
file_path = "C:/brainwave_matrix_task/twitter_validation.csv"
df = pd.read_csv(file_path, header=None)  # No headers in the dataset

# Display first few rows
df.head()

# Rename columns
df.columns = ["ID", "Topic", "Sentiment", "Tweet"]

# Remove "Irrelevant" sentiment tweets since they don't contribute to sentiment analysis
df = df[df["Sentiment"] != "Irrelevant"]

# Check updated sentiment distribution
df["Sentiment"].value_counts()


import re
# Define a basic set of stopwords manually
custom_stopwords = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
    "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have",
    "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
    "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "only", "own", "same", "so", "than", "too", "very", "s", "t",
    "can", "will", "just", "don", "should", "now"
])

# Function to clean text
def clean_text_alternative(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+|#", "", text)  # Remove mentions and hashtags
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove punctuation and numbers
    text = " ".join([word for word in text.split() if word not in custom_stopwords])  # Remove stopwords
    return text

# Apply cleaning function to the tweets
df["Cleaned_Tweet"] = df["Tweet"].astype(str).apply(clean_text_alternative)

# Display cleaned tweets
df[["Tweet", "Cleaned_Tweet"]].head()


from textblob import TextBlob

# Function to get sentiment polarity (-1 to 1 scale)
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

# Apply sentiment scoring
df["Sentiment_Score"] = df["Cleaned_Tweet"].apply(get_sentiment)

# Categorize sentiment based on polarity score
df["Predicted_Sentiment"] = df["Sentiment_Score"].apply(
    lambda x: "Positive" if x > 0 else ("Negative" if x < 0 else "Neutral")
)

# Compare predicted sentiment with original labels
df[["Cleaned_Tweet", "Sentiment", "Predicted_Sentiment", "Sentiment_Score"]].head()


# Countplot for sentiment distribution by topic
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x="Topic", hue="Predicted_Sentiment", palette="coolwarm")
plt.title("Sentiment Distribution by Topic", fontsize=14)
plt.xlabel("Topic")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.legend(title="Sentiment")
plt.show()

# Sort data by ID to simulate time-based sentiment trend
df_sorted = df.sort_values(by="ID")

# Plot sentiment trend over time
plt.figure(figsize=(10, 5))
sns.lineplot(data=df_sorted, x=range(len(df_sorted)), y="Sentiment_Score", marker="o", color="blue", alpha=0.7)
plt.axhline(0, color="black", linestyle="--", alpha=0.6)
plt.title("Sentiment Trend Over Time (Simulated)", fontsize=14)
plt.xlabel("Tweet Index (Sorted by ID)")
plt.ylabel("Sentiment Score")
plt.show()


from wordcloud import WordCloud

# Combine all cleaned tweets into one large string
all_words = " ".join(df["Cleaned_Tweet"])

# Generate word cloud
wordcloud = WordCloud(width=800, height=400, background_color="white", colormap="coolwarm").generate(all_words)

# Display word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Most Common Words in Tweets", fontsize=14)
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set(style="whitegrid")

# Countplot for sentiment distribution (Fixed version)
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Predicted_Sentiment", hue="Predicted_Sentiment", 
              order=["Positive", "Neutral", "Negative"], palette="coolwarm", legend=False)

plt.title("Sentiment Distribution", fontsize=14)
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()


