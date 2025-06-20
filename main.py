import pandas as pd 
import os 


from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import apprendimentoSupervisionato
import warnings
import reteBayesana
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

warnings.filterwarnings("ignore", category=UserWarning)

## apprendimento supervisionato + probabilistico (naive bayes)
fileName = os.path.join(os.path.dirname(__file__), "./Dataset/superhero_abilities_dataset.csv")

dataSet = pd.read_csv(fileName)
dataSet = dataSet.drop(columns=['Name'])
dataSet.dropna(inplace=True)
dataSet['Universe'] = pd.Categorical(dataSet['Universe']).codes
dataSet['Weapon'] = pd.Categorical(dataSet['Weapon']).codes
dataSet['Alignment'] = pd.Categorical(dataSet['Alignment']).codes


differentialColumn = 'Universe'

#classificazione per Universo

# models = [
#     ('Naive Bayes', MultinomialNB())
# ]

model = apprendimentoSupervisionato.trainModelKFold(dataSet, differentialColumn)
