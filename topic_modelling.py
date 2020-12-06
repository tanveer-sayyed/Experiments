import re
import numpy
import pandas
import string
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


class TopicModelling():
    def __init__(self, text, no_of_topics):
        """
        Parameters:
        ----------
            text: pandas.Series(str)
                the text column of signals_df
            no_of_topics: int
                divide the text in these number of topics

        Returns:
        -------
            Dominant topic: pandas.Series(int)
                The topic number to which each item in text belongs
            Topic Keywords: DataFrame(str)
                The top 20 workds in each topic
        """
        stop = set(stopwords.words('english'))
        exclude = set(string.punctuation) # incase numbers also have to 
                                          # excluded do -> exclude.add(str(1))
        lemma = WordNetLemmatizer()

        def clean(doc):
            # Remove stop words after converring whole doc to lower case
            stop_free = " ".join(
                    [i for i in doc.lower().split() if i not in stop])
            # Remove punctuations
            punc_free = ''.join(
                    ch for ch in stop_free if ch not in exclude)
            # lemmatize words to remove multiple forms of the same word
            normalized = " ".join(
                    lemma.lemmatize(word) for word in punc_free.split())
            return normalized

        doc_clean = [clean(item) for item in text]

        """ Following code from:
            https://www.machinelearningplus.com/nlp/topic-modeling
                                       -python-sklearn-examples/"""
        # Remove emails    
        doc_clean = [re.sub('\S*@\S*\s?', '', sent) for sent in doc_clean]
        # Remove new line characters
        doc_clean = [re.sub('\s+', ' ', sent) for sent in doc_clean]
        # Remove single quotes
        doc_clean = [re.sub("\'", "", sent) for sent in doc_clean]

        vectorizer = CountVectorizer(analyzer='word',
                                     lowercase= False, # already done by 
                                                       # clean(doc) function
                                     token_pattern='[a-zA-Z0-9]{3,}'
                                     # only accept words with num chars > 3
                                     )
        data_vectorized = vectorizer.fit_transform(doc_clean)
        lda_model = LatentDirichletAllocation(n_components= 5,
                                              learning_method='online',
                                              random_state= 42,
                                              evaluate_every= 0,
                                              n_jobs = -1
                                              )
        # To increase performance set param 'evaluate_every' to an integer > 0.
        # Evaluating perplexity can help you check convergence in training
        # process, but it will also increase total training time.
        # Param 'evaluate_every' is checked and balanced by param 'perp_tol'.

        lda_output = lda_model.fit_transform(data_vectorized)

        # see the dominant topic in each document
        lda_output = lda_model.transform(data_vectorized)
        topic_names = ["Topic_" + str(i) for i in range(lda_model.n_topics)]
        row_names = ["Row_" + str(i) for i in range(len(doc_clean))]
        df_row_with_topic = pandas.DataFrame(numpy.round(lda_output, 2),
                                             columns=topic_names,
                                             index=row_names)
        dominant_topic = numpy.argmax(df_row_with_topic.values, axis=1)
        df_row_with_topic['Dominant_Topic'] = dominant_topic

        # Show top n keywords for each topic
        def show_topics(vectorizer=vectorizer,
                        lda_model=lda_model,
                        n_words=20):
            keywords = numpy.array(vectorizer.get_feature_names())
            topic_keywords = []
            for topic_weights in lda_model.components_:
                top_keyword_locs = (-topic_weights).argsort()[:n_words]
                topic_keywords.append(keywords.take(top_keyword_locs))
            return topic_keywords
        topic_keywords = pandas.DataFrame(show_topics(vectorizer=vectorizer,
                                                      lda_model=lda_model,
                                                      n_words=20),
                                          index= topic_names).T
        return{
                'text': df_row_with_topic['Dominant_Topic']
                'topic_keywords': topic_keywords
                }
