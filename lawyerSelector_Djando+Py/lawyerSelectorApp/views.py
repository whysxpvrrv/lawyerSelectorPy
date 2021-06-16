from urllib import request
from django.http.response import HttpResponse
from django.shortcuts import render
# Bibliotecas Lenguaje Natural Toolkit
import nltk
from nltk.stem.lancaster import LancasterStemmer
# Bibliotecas aux
import os
import json
import datetime
import numpy as np
import time
from django.http import HttpResponseNotFound
from django.template import loader


# ----------- REDIRECCION TEMPLATES  -----------
# Vista Index
def index(request):
    return render(request, "lawyerSelectorApp/index.html")

# Vista Recomendacion
def yourSort(request):
    mensaje = request.GET["prd"]
    prediccion = classify(mensaje)
    if prediccion == "penal":
        return render(request, "lawyerSelectorApp/penalLawyer.html")
    if prediccion == "familiar":
        return render(request, "lawyerSelectorApp/familyLawyer.html")
    if prediccion == "bancario":
        return render(request, "lawyerSelectorApp/bankLawyer.html")
    if prediccion == "medico":
        return render(request, "lawyerSelectorApp/medicineLawyer.html")

def Errorhandler404(request, exception):
    content = loader.render_to_string('lawyerSelectorApp/404.html', {}, request)
    return HttpResponseNotFound(content)

# ----------- RED NEUORONAL -----------
stemmer = LancasterStemmer()

# Entrenamientos
training_data = []
# Penal
training_data.append({"class": "penal", "sentence": "Condena, multa"})
training_data.append(
    {"class": "penal", "sentence": "Robo o atraco a mano armada"})
training_data.append(
    {"class": "penal", "sentence": "Violencia de genero, asesinato, blanqueo de dinero,pelea"})
training_data.append(
    {"class": "penal", "sentence": "Detencion ilegal, secuestro o amenaza"})

# Bancario
training_data.append({"class": "bancario", "sentence": "Banco, bancario"})
training_data.append({"class": "bancario", "sentence": "Suma de dinero"})
training_data.append(
    {"class": "bancario", "sentence": "Hipoteca, prestamo, banco, bancario, cargos a deber"})

# Familiar
training_data.append(
    {"class": "familiar", "sentence": "Familia, hijos, padres, tio, suegro, nuero"})
training_data.append(
    {"class": "familiar", "sentence": "Matriminio de mutuo acuerdo"})
training_data.append({"class": "familiar", "sentence": "Divorcio"})

# Medico
training_data.append({"class": "medico", "sentence": "Dolores"})
training_data.append(
    {"class": "medico", "sentence": "Aborto, lesiones, manipulacion genetica"})
training_data.append({"class": "medico", "sentence": "Negligencia"})

print("\n\t*----------------- Neuronal Network Info v.3.2 -----------------*\n \t|\t\t\tBy Fernando Parra\n\t|")
print('\t|\t%s' % "%s frases utilizadas para el entrenamiento" %
      len(training_data))

# Arrays Aux
words = []
classes = []
documents = []
ignore_words = ['?']
# Recorremos nuestro training_data y tokenizamos cada palabra
for pattern in training_data:
    # Tokenizamos cada palabra en la frase
    w = nltk.word_tokenize(pattern['sentence'])
    # Añadimos el token a la palabra
    words.extend(w)
    documents.append((w, pattern['class']))
    # Añadimos el token+palabra a su clase correspondiente
    if pattern['class'] not in classes:
        classes.append(pattern['class'])

# Eliminamos duplicados y minusculas
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = list(set(words))
classes = list(set(classes))


# Creamos nuestro modelo de entrenamiento
training = []
output = []
# Creamos un array vacio para los resultados
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # stem each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create our bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    training.append(bag)
    # output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    output.append(output_row)


# sample training/output
i = 0
w = documents[i][0]


# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative


def sigmoid_output_to_derivative(output):
    return output*(1-output)


def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence


def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)

    return(np.array(bag))


def think(sentence, show_details=False):
    x = bow(sentence.lower(), words, show_details)
    if show_details:
        print("sentence:", sentence, "\n bow:", x)
    # input layer is our bag of words
    l0 = x
    # matrix multiplication of input and hidden layer
    l1 = sigmoid(np.dot(l0, synapse_0))
    # output layer
    l2 = sigmoid(np.dot(l1, synapse_1))
    return l2

    # ANN and Gradient Descent code from


def train(X, y, hidden_neurons=10, alpha=1, epochs=50000, dropout=False, dropout_percent=0.5):

    print('\t|\t%s' % "Entrenamiento con %s neuronas, alpha:%s, dropout:%s %s" % (
        hidden_neurons, str(alpha), dropout, dropout_percent if dropout else ''))
    print('\t|\t%s' % "Matriz de Entrada: %sx%s    Matriz de Salida: %sx%s" %
          (len(X), len(X[0]), 1, len(classes)))
    print('\t|')
    np.random.seed(1)

    last_mean_error = 1
    # randomly initialize our weights with mean 0
    synapse_0 = 2*np.random.random((len(X[0]), hidden_neurons)) - 1
    synapse_1 = 2*np.random.random((hidden_neurons, len(classes))) - 1

    prev_synapse_0_weight_update = np.zeros_like(synapse_0)
    prev_synapse_1_weight_update = np.zeros_like(synapse_1)

    synapse_0_direction_count = np.zeros_like(synapse_0)
    synapse_1_direction_count = np.zeros_like(synapse_1)

    for j in iter(range(epochs+1)):

        # Feed forward through layers 0, 1, and 2
        layer_0 = X
        layer_1 = sigmoid(np.dot(layer_0, synapse_0))

        if(dropout):
            layer_1 *= np.random.binomial([np.ones((len(X), hidden_neurons))],
                                          1-dropout_percent)[0] * (1.0/(1-dropout_percent))

        layer_2 = sigmoid(np.dot(layer_1, synapse_1))

        # how much did we miss the target value?
        layer_2_error = y - layer_2

        if (j % 10000) == 0 and j > 5000:
            # if this 10k iteration's error is greater than the last iteration, break out
            if np.mean(np.abs(layer_2_error)) < last_mean_error:

                print('\t|\t%s' % "Delta "+str(j)+" iteraciones: " +
                      str(np.mean(np.abs(layer_2_error))))
                last_mean_error = np.mean(np.abs(layer_2_error))
            else:
                print("break:", np.mean(np.abs(layer_2_error)),
                      ">", last_mean_error)
                break

        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)

        # how much did each l1 value contribute to the l2 error (according to the weights)?
        layer_1_error = layer_2_delta.dot(synapse_1.T)

        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)

        synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))
        synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))

        if(j > 0):
            synapse_0_direction_count += np.abs(
                ((synapse_0_weight_update > 0)+0) - ((prev_synapse_0_weight_update > 0) + 0))
            synapse_1_direction_count += np.abs(
                ((synapse_1_weight_update > 0)+0) - ((prev_synapse_1_weight_update > 0) + 0))

        synapse_1 += alpha * synapse_1_weight_update
        synapse_0 += alpha * synapse_0_weight_update

        prev_synapse_0_weight_update = synapse_0_weight_update
        prev_synapse_1_weight_update = synapse_1_weight_update

    now = datetime.datetime.now()

    # persist synapses
    synapse = {'synapse0': synapse_0.tolist(), 'synapse1': synapse_1.tolist(),
               'datetime': now.strftime("%Y-%m-%d %H:%M"),
               'words': words,
               'classes': classes
               }
    synapse_file = "synapses.json"

    with open(synapse_file, 'w') as outfile:
        json.dump(synapse, outfile, indent=4, sort_keys=True)
    print('\t|\t%s' % "Pesos sinapticos guardados en:", synapse_file)


X = np.array(training)
y = np.array(output)

start_time = time.time()

train(X, y, hidden_neurons=20, alpha=0.1,
      epochs=100000, dropout=False, dropout_percent=0.2)

elapsed_time = time.time() - start_time
print('\t|\t%s' % "\n\t|\tHe tardado", elapsed_time,
      "segundos en entrenarme con\n\t|\tlas frases de ejemplo y compilar mi modelo!\n\t|\n\t|\tEstoy listo para procesar texto y predecir :)")

# probability threshold
ERROR_THRESHOLD = 0.2
# Guardamos los pesos sinapticos en synanpses.json
synapse_file = 'synapses.json'
with open(synapse_file) as data_file:
    synapse = json.load(data_file)
    synapse_0 = np.asarray(synapse['synapse0'])
    synapse_1 = np.asarray(synapse['synapse1'])
print("\t|\n\t*---------------------------------------------------------------*\n\n ")

# Funcion de clasificacion


def classify(sentence, show_details=False):
    results = think(sentence, show_details)
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_results = [[classes[r[0]], r[1]] for r in results]
    # Info de la Prediccion
    print("\n\n\t*-------------- Neuronal Network Results  --------------*\n\t|")
    print("\t|\tFrase: %s" % sentence)
    print("\t|\tPrediccion: %s" % return_results[0][0])
    print("\t|\tMargen de Acierto: %s" % return_results[0][1])
    print("\t|\n\t*---------------------------------------------------------*\n\n")

    return return_results[0][0]
