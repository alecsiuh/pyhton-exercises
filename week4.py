import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
import math

from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras import Model
from keras.layers import Input, Dense, BatchNormalization
from keras.utils import to_categorical
import keras.datasets
from keras.optimizers import Adam, RMSprop
from livelossplot import PlotLossesKeras
from sklearn.model_selection import train_test_split

# Discriminant Analysis

# Question 1

wine = pd.read_csv('../datasets/bordeaux.csv', delimiter=';')
print(wine)

# a)
X = wine[['temperature', 'sun', 'heat', 'rain']]
y = wine['quality']
lda = LinearDiscriminantAnalysis()
lda.fit(X, y)  # independent variable, dependent variable

# b)
def LDA_coefficients(X, lda):
    nb_col = X.shape[1]
    matrix = np.zeros((nb_col+1, nb_col), dtype=int)
    Z = pd.DataFrame(data=matrix, columns=X.columns)
    for j in range(0, nb_col):
        Z.iloc[j, j] = 1
    LD = lda.transform(Z)
    nb_funct = LD.shape[1]
    result = pd.DataFrame()
    index = ['const']
    for j in range(0, LD.shape[0]-1):
        index = np.append(index, 'C'+str(j+1))
    for i in range(0, LD.shape[1]):
        coef = [LD[-1][i]]
        for j in range(0, LD.shape[0]-1):
            coef = np.append(coef, LD[j][i]-LD[-1][i])
        coef_column = pd.Series(coef)
        coef_column.index = index
        column_name = 'LD' + str(i+1)
        result[column_name] = coef_column
    return result


print("Coefficients:")
print(LDA_coefficients(X, lda))  # 2-dimensional -> N = min(3-1, 4) = 2

def calculateDimensionality(X, y):
    a = len(X.columns)-1
    b = len(y.unique())
    return min(a, b)

print("DIM", calculateDimensionality(X, y))

#c)
classes = wine.quality
LD = lda.transform(X)
LD_df = pd.DataFrame(zip(LD[:, 0], LD[:, 1], classes), columns=['LD1', 'LD2', 'Target'])
scatter_x = np.array(LD_df['LD1'])
scatter_y = np.array(LD_df['LD2'])
group = np.array(LD_df['Target'])
cdict = {'good': 'green', 'medium': 'orange', 'bad': 'red'}

fig, ax = plt.subplots()
for g in np.unique(group):
    ix = np.where(group == g)
    ax.scatter(scatter_x[ix], scatter_y[ix], c = cdict[g], label = g)
ax.legend()
# plt.show()

# d)
predictions = lda.predict(X)
predictions = pd.Series(predictions)
print(predictions)
print(y)
correct_count = 0
for i in range(0, len(predictions)):
    if predictions[i] == y[i]:
        correct_count += 1
print(correct_count/len(predictions))  # 0.7941



# Question 2
birthwt = pd.read_csv('../datasets/birthwt.csv', delimiter=';')
#a)
print(birthwt.describe())
#b) dependent
# low, smoke, race
#c) independent
# age, lwt, bwt
#d)
X = birthwt[['age', 'lwt', 'bwt']]
y = birthwt['smoke']
lda = LinearDiscriminantAnalysis()
lda.fit(X, y)  # independent variable, dependent variable
#e)
print(birthwt.smoke.unique())
# 1
#f)
def visualize_results(lda, X, classes, showIndividualPlots=True):
    # mapping the independent variables based on the discriminant functions of the model to their N discriminant values
    LD = lda.transform(X)

    # combine with the original dependent variable
    LD_df = pd.DataFrame(zip(LD[:, 0], classes), columns=['LD1', 'Target'])

    labels = classes.unique()

    plt.figure()
    colors = list(mcolors.CSS4_COLORS.keys())
    if showIndividualPlots:
        color = random.choice(colors)
        LD_df.hist(column=['LD1'], by='Target', bins=25, density=True, edgecolor='black', color=color, sharex=True,
                   sharey=False, figsize=(10, 10))
    fig, ax = plt.subplots(figsize=(10, 5))
    for label in labels:
        color = random.choice(colors)
        LD_df['LD1'][LD_df['Target'] == label].hist(ax=ax, bins=25, density=True, edgecolor='black', color=color,
                                                    alpha=0.7, label=label)
    ax.legend()
    ax.grid(False)
    plt.show()

# visualize_results(lda, X, birthwt['smoke'], True)

# Question 3
cars = pd.read_csv("../datasets/Cars93.csv", delimiter=";", decimal='.')

#a)
print(cars.describe())
#b)
cars_train = cars.iloc[:90]
cars_test = cars.iloc[-3:]
#c)
# manufacturer, model, type, airbags, man.trans.avail, drivetrain, origin, make, cylinders
#d)
# min.price, price, max.price, mpg.city, mpg.highway, enginesize, horsepower, rpm, rev.per.mile,
# fuel.tank.capacity, passengers, length, wheelbase, width, turn.circle, rear.seat.room, luggage.room, weight
#e)
# print(cars_train.columns)
cars_train = cars_train.dropna()
print(cars_train)
X = cars_train[['Min.Price', 'Price', 'Max.Price',
                'MPG.city', 'MPG.highway',
                'EngineSize', 'Horsepower', 'RPM', 'Rev.per.mile',
                'Fuel.tank.capacity', 'Passengers', 'Length', 'Wheelbase', 'Width',
                'Turn.circle', 'Rear.seat.room', 'Luggage.room', 'Weight']]
y = cars_train['Type']
lda = LinearDiscriminantAnalysis()
lda.fit(X, y)  # independent variable, dependent variable
#f)
print(calculateDimensionality(X, y))  # 5
#g)
testX = cars_test[['Min.Price', 'Price', 'Max.Price',
                   'MPG.city', 'MPG.highway',
                   'EngineSize', 'Horsepower', 'RPM', 'Rev.per.mile',
                   'Fuel.tank.capacity', 'Passengers', 'Length', 'Wheelbase', 'Width',
                   'Turn.circle', 'Rear.seat.room', 'Luggage.room', 'Weight']]
testY = cars_test['Type']
print(lda.predict(testX))
print(testY)
# Just the last two


# Neural Networks

# Question 1
#a)
simpsons = pd.read_csv("../datasets/The Simpsons original.csv")
#b)
# Classification
#c)
# name, gender
#d)

def min_max_norm(col):
    minimum = col.min()
    range = col.max() - minimum
    print("min", minimum)
    print("range", range)
    return (col-minimum)/range

def normalized_values(df, norm_funct):
    df_norm = pd.DataFrame()
    for column in df:
        print(column)
        df_norm[column] = norm_funct(df[column])
    return df_norm

'''
y_simpsons = pd.get_dummies(simpsons.gender)
# print(y_simpsons)0
# y_simpsons = simpsons.gender
# y_simpsons = y_simpsons.astype(str)
# y_simpsons = y_simpsons.str.replace('M', '1')
# y_simpsons = y_simpsons.str.replace('F', '0')
# y_simpsons = y_simpsons.astype(int)
# y_simpsons = to_categorical(y_simpsons)
# print(y_simpsons)

# y_simpsons = y_simpsons.astype(int)
# print(y_simpsons)
x_simpsons = simpsons.drop(['name', 'gender'], axis=1)
print(x_simpsons)
x_simpsons = normalized_values(x_simpsons, min_max_norm)
print(x_simpsons)

#e)
# print(len(x_simpsons.columns))
inputs = Input(shape=(len(x_simpsons.columns),))
layer = BatchNormalization()(inputs)
layer = Dense(64, activation='relu')(layer)  # Hidden Layer 1
layer = Dense(44, activation='relu')(layer)  # Hidden Layer 2
outputs = Dense(2, activation='softmax')(layer)

model = Model(inputs, outputs, name='simpsons')
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=keras.losses.binary_crossentropy,
              metrics=['accuracy'])
history = model.fit(
    x_simpsons,  # training data
    y_simpsons,  # training data
    epochs=100
)

print(model.summary())  # -> "param #" refers to the number of connections between the layers
plot_model(model, to_file='model_simpsons_plot.png', show_shapes=True, show_layer_names=True)

#f)
print(np.argmax(model.predict(x_simpsons), axis=1))  #
print(y_simpsons)
# matches all except nr 6
# x_simpsons = simpsons.drop(['name', 'gender'], axis=1)
comic = pd.Series(data={'hair length': 8, 'weight': 500, 'age': 38}, index=['hair length', 'weight', 'age'])
# return (col-minimum)/range
comic['hair length'] = (comic['hair length'] - 0)/10
comic.weight = (comic.weight - 20)/230
comic.age = (comic.age - 1)/69
x_simpsons = x_simpsons.append(comic, ignore_index=True)
print(x_simpsons)
print(np.argmax(model.predict(x_simpsons), axis=1))
'''

'''
# Question 2
#a)
comp = pd.read_csv('../datasets/forcastdemo.csv', delimiter=';')
#b)
# Regression
#c)
# first two columns = input, last column = output
#d)
x_forcast = comp[['Year', 'Quarter']]
y_forcast = comp[['Revenu']]
y_forcast = normalized_values(y_forcast, min_max_norm)
#e)
inputs = Input(shape=(2,))
x = Dense(4, activation='relu')(inputs)
x = Dense(8, activation='relu')(x)
x = Dense(16, activation='relu')(x)
x = Dense(8, activation='relu')(x)
x = Dense(4, activation='relu')(x)
# x = Dense(2, activation='relu')(x)
# x = Dense(8, activation='linear')(x)
# x = Dense(4, activation='linear')(x)
outputs = Dense(1, activation='linear')(x)

model = Model(inputs, outputs, name='model')
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.001), loss=keras.losses.MeanAbsoluteError(),
              # metrics=keras.metrics.MeanAbsolutePercentageError())
              metrics=['accuracy'])
history = model.fit(
    x_forcast,  # training data
    y_forcast,  # training data
    epochs=1000
)
plot_model(model, to_file='model_company_plot.png', show_shapes=True, show_layer_names=True)


#f)
# return (col-minimum)/range
minval = 13
rangeval = 432
predictions = model.predict(x_forcast)
print(predictions)
prediction_arr = []
for prediction in predictions:
    prediction[0] *= rangeval
    prediction[0] += minval
    prediction_arr.append(prediction[0])
print(prediction_arr)
for i in range(0, len(y_forcast)):
    print(y_forcast.iloc[i])
    y_forcast.iloc[i] *= rangeval
    y_forcast.iloc[i] += minval

plt.figure()
plt.plot(range(0, len(y_forcast)), y_forcast.Revenu, label='actual')
plt.plot(range(0, len(prediction_arr)), prediction_arr, label='predict')
plt.legend()
plt.show()
'''


# Question 3

#a)
iris = pd.read_csv('../datasets/iris.csv', decimal='.')
#b)
# classification
#c)
# first four = input, last = output
#d)
def decimal_scaling_norm(col):
    maximum = col.max()
    tenfold = 1
    while maximum > tenfold:
        tenfold += 10
    return col/tenfold

x_iris = iris.iloc[:,:4]
x_iris_norm = normalized_values(x_iris, decimal_scaling_norm)
y_iris = iris.target
print(y_iris)
counter = 1
for cat in y_iris.unique():
    print(cat)
    print(counter)
    y_iris = y_iris.str.replace(str(cat), str(counter))
    counter += 1

y_iris = y_iris.astype(int)
print(y_iris)

#e)
x_train_iris, x_test_iris, y_train_iris, y_test_iris = train_test_split(x_iris_norm, y_iris, test_size=0.2)
#f)
y_train_iris = to_categorical(y_train_iris)
y_test_iris = to_categorical(y_test_iris)

inputs = Input(shape=(4,))
x = BatchNormalization()(inputs)
x = Dense(64, activation='sigmoid')(x)
x = Dense(32, activation='sigmoid')(x)
x = Dense(16, activation='sigmoid')(x)
x = Dense(8, activation='sigmoid')(x)
outputs = Dense(4, activation='softmax')(x)

model = Model(inputs, outputs, name='iris')
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=keras.losses.categorical_crossentropy,
              metrics=['accuracy'])
history = model.fit(
    x_train_iris,  # training data
    y_train_iris,  # training data
    epochs=500,
    validation_split=0.1
)

#g)
print(model.evaluate(x_test_iris, y_test_iris))  # 0.9667 accuracy
