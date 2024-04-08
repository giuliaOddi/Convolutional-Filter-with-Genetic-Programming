import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from numpy import *
from deap import base, creator, tools, algorithms, gp
import operator
import numpy as np

# Reading .csv file
df = df = pd.read_csv(r'./data_bilanced_6000elementi.csv')
labels = df.iloc[:,0] # Labels is the first column
data = df.iloc[:,1:] # Data is the dataset withput the first column

#######
# Random Forest Classificator 
#######
seed_state = 1200

# Train-set and test-set division
X = data
y = labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed_state)

# Implementation of the Random Forest Classificator
clf = RandomForestClassifier(n_estimators=100, random_state=seed_state)
# Training on the train-set
clf.fit(X_train, y_train)
# Testing on the test set
predictions = clf.predict(X_test)

# Score f1
f1 = f1_score(y_test, predictions)
print("F1 score Random Forest: ", f1)


#######
# Genetic Programming
#######

# Train-set splitting in train-set and validation-set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed_state)

# Fitness and individual creation
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

# Primitive set inizialization and kernel dimension definition 
KERNEL_SIZE = 7 

pset = gp.PrimitiveSet("MAIN", KERNEL_SIZE)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(operator.mul, 2)
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1
pset.addPrimitive(protectedDiv, 2)

# Convolution of a single row 
def apply_convolution_to_row(row, func):
    num_windows = len(row) - KERNEL_SIZE + 1
    result = []
    for i in range(num_windows):
        window = row[i:i + KERNEL_SIZE]
        valore = func(*window) 
        result.append(valore)

    return result

# Fitness function for individual evaluation 
def fitness_function(individual):

    clf = gp.PrimitiveTree(individual)
    func = gp.compile(clf, pset)

    # Dataframe to store results  
    conv_train = pd.DataFrame()
    conv_val = pd.DataFrame()

    # Convolution on train-set and storing results
    for row_train in X_train.iterrows():
        index, row_train = row_train
        conv_row_train = apply_convolution_to_row(row_train, func)
        if conv_row_train is not None:
            row_temp = pd.DataFrame(conv_row_train).T  # .T traspose matrix in order to obtain a row
            conv_train = pd.concat([conv_train, row_temp], ignore_index=True)

    # Convolution on validation-set and storing results
    for row_val in X_val.iterrows():
        index, row_val = row_val
        conv_row_val = apply_convolution_to_row(row_val, func)
        if conv_row_val is not None:
            row_temp = pd.DataFrame(conv_row_val).T  # .T  traspose matrix in order to obtain a row
            conv_val = pd.concat([conv_val, row_temp], ignore_index=True)

    # Random Forest training, using transformed signals
    rf = RandomForestClassifier(n_estimators=100, random_state=seed_state)    
    # Training on train-set
    rf.fit(conv_train, y_train)    
    # Previsions on validation-set 
    gp_predictions = rf.predict(conv_val)

    # Fitness evaluation using F1-score
    f1 = f1_score(y_val, gp_predictions)  
    nodes = len(individual)
    k = 10e-5
    fitness = f1 / (1 + k * nodes)

    # Results are the fitness and the classificator 
    return (fitness, rf)

# Toolbox definition
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
def evaluate_fitness(individual):
    fitness, rf = fitness_function(individual)
    return fitness,
toolbox.register("evaluate", evaluate_fitness)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=4)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# Initial population and number of generations 
population = toolbox.population(n=50)  
n_generations = 30

# Have access ti best kernel of the execution 
hof = tools.HallOfFame(1)

# Algorithm and statistics used
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)
algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.3, ngen=n_generations, stats=stats, halloffame=hof, verbose=True)

# Find the best indivual
best_individual = hof[0]
best_fitness, best_rf = fitness_function(best_individual)   # Fitness evaluation best indivual 
best_kernel_function = gp.compile(gp.PrimitiveTree(best_individual), pset)
conv_test = pd.DataFrame()

# Testing on the test-set 
for row_test in X_test.iterrows():
    index, row_test = row_test
    conv_row_test = apply_convolution_to_row(row_test, best_kernel_function)
    if conv_row_test is not None:
        row_temp = pd.DataFrame(conv_row_test).T
        conv_test = pd.concat([conv_test, row_temp], ignore_index=True)

# Previsions on test-set
gp_predictions_test = best_rf.predict(conv_test)

# Score f1 evaluation for the best kernel  
f1_conv = f1_score(y_test, gp_predictions_test)
print("F1 score after convolution: ", f1_conv)