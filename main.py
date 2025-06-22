import pandas as pd 
import os 


from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder

from sklearn.naive_bayes import MultinomialNB

import apprendimentoSupervisionato
import warnings
import reteBayesana
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

fileName = os.path.join(os.path.dirname(__file__), "./Dataset/personality_dataset.csv")
dataSet = pd.read_csv(fileName)
dataSet['Stage_fear'] = pd.Categorical(dataSet['Stage_fear']).codes
dataSet['Drained_after_socializing'] = pd.Categorical(dataSet['Drained_after_socializing']).codes
dataSet['Personality'] = pd.Categorical(dataSet['Personality']).codes

differentialColumn = 'Personality'

#classificazione per Universo

# models = [
#     ('Naive Bayes', MultinomialNB())
# ]

model = apprendimentoSupervisionato.trainModelKFold(dataSet, differentialColumn)
