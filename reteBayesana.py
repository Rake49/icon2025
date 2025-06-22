import pickle
import networkx as nx
from matplotlib import pyplot as plt
from pgmpy.estimators import MaximumLikelihoodEstimator, HillClimbSearch
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork


# Funzione che visualizza il grafo del Bayesian Network
def visualizeBayesianNetwork(bayesianNetwork):
    G = nx.DiGraph(bayesianNetwork.edges())
    pos = nx.kamada_kawai_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightgreen", 
            font_size=10, font_weight="bold", arrows=True, arrowsize=20, edge_color="gray")
    plt.title("Bayesian Network Graph (Tree-like Variation)")
    plt.show()


def visualizeInfo(bayesianNetwork):
    # Ottengo le distribuzioni di probabilità condizionata (CPD)
    cpd_list = bayesianNetwork.get_cpds()
    for cpd in cpd_list:
        print(f"\nCPD per la variabile '{cpd.variable}':")
        print(cpd)
        print("=" * 40)


# Funzione che crea la rete bayesiana
def bNetCreation(dataSet): 

    # Ricerca della struttura ottimale
    hc_k2 = HillClimbSearch(dataSet)
    k2_model = hc_k2.estimate(scoring_method='k2')
    # Creazione della rete bayesiana

    model = DiscreteBayesianNetwork(k2_model.edges)

    # model = BayesianNetwork(edges)
    model.fit(dataSet, estimator=MaximumLikelihoodEstimator, n_jobs=-1)
    # Salvo la rete bayesiana su file
    with open('modello.pkl', 'wb') as output:
        pickle.dump(model, output)
    visualizeBayesianNetwork(model)
    # visualizeInfo(model)
    return model

# Funzione che carica la rete bayesiana da file


def loadBayesianNetwork():
    with open('modello.pkl', 'rb') as input:
        model = pickle.load(input)
    visualizeBayesianNetwork(model)
    # visualizeInfo(model)
    return model

# Predico il valore di differentialColumn per l'esempio


def predici(bayesianNetwork, example, differentialColumn):
    inference = VariableElimination(bayesianNetwork)
    result = inference.query(variables=[differentialColumn], evidence=example)
    print(result)

# genera un esempio randomico


def generateRandomExample(bayesianNetwork):
    return bayesianNetwork.simulate(n_samples=1).drop(columns=['Rating'])