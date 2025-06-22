import pandas as pd 
import os
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

# model = apprendimentoSupervisionato.trainModelKFold(dataSet, differentialColumn)
bayesianNetwork = reteBayesana.bNetCreation(dataSet)

# Generazione esempio randomico e predizione
esempioRandom = reteBayesana.generateRandomExample(bayesianNetwork)
print("Esempio random:\n ", esempioRandom.head())
print("Predizione associata:")
reteBayesana.predici(bayesianNetwork, esempioRandom.to_dict('records')[0], differentialColumn)

# Provo a predirre un esempio a cui manca anche una feature di input (Friends_circle_size)
del(esempioRandom['Friends_circle_size'])
print("Esempio random senza la feature 'Friends_circle_size'\n", esempioRandom)
print("Predizione dell'esempio random senza Personality e Friends_circle_size")
reteBayesana.predici(bayesianNetwork, esempioRandom.to_dict('records')[0], differentialColumn)