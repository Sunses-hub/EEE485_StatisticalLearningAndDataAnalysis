
import pandas as pd
import numpy as np
from os import path, makedirs, walk
from time import time
import matplotlib.pyplot as plt
import seaborn as sn

np.random.seed(4)
######################################## CLASSES ##########################################


class Layer(object):

    def __init__(self, neuron_num, activation, input_dim, std=0.01):

        self.neuron_num = neuron_num
        self.input_dim = input_dim
        self.weights = np.random.normal(size=(neuron_num, input_dim), loc=0, scale=std)
        self.bias = np.random.normal(size=(neuron_num, 1), loc=0, scale=std)
        self.weight_ext = np.concatenate((self.weights, self.bias), axis=1)

        self.activation = activation
        self.prev_weight = 0

        self.out = None
        self.delta = None
        self.prev_update = 0

    def output(self, X, activation):

        X_ext = np.concatenate((X, (-1 * np.ones(X.shape[0])).reshape((X.shape[0], 1))), axis=1)
        act_pot = X_ext @ self.weight_ext.T
        if activation == "softmax":
            self.out = np.exp(act_pot - self.logsumexp(act_pot, axis=1, keepdims=True))
        elif activation == "sigmoid":
            self.out = np.exp(act_pot) / ( 1 + np.exp(act_pot) )
        else:
            self.out = np.tanh(act_pot)
        return self.out

    def logsumexp(self, X, axis=None, keepdims=False):

        X_max = np.amax(X, axis=axis, keepdims=keepdims)
        if X_max.ndim > 0:
            X_max[~np.isfinite(X_max)] = 0
        elif not np.isfinite(X_max):
            X_max = 0
        temp = np.exp(X - X_max)
        with np.errstate(divide='ignore'):
            z = np.sum(temp, axis=axis, keepdims=keepdims)
            output = np.log(z)
        if not keepdims:
            X_max = np.squeeze(X_max, axis=axis)
        output+=X_max
        return output


class NeuralNetwork(object):

    def __init__(self, early_stopping=None):

        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward_pass(self, X):
        temp = X
        for layer in self.layers:
            temp = layer.output(temp, activation=layer.activation)
        return temp

    def backward_pass(self, X, Y, learning_rate, batch_size=0, moment=0):

        Y_pred = self.forward_pass(X)
        out_error = Y - Y_pred

        #Find the delta values
        for j in range(len(self.layers)-1, -1, -1):
            #Input layer
            if j == 0:
                second_layer, first_layer = self.layers[1], self.layers[0]
                weights = second_layer.weight_ext[:, :-1].reshape((second_layer.weight_ext.shape[0], second_layer.weight_ext.shape[1] - 1))
                first_layer.delta = (weights.T @ second_layer.delta)
            else:
                second_layer, first_layer = self.layers[j], self.layers[j - 1]
                first_ext = np.concatenate(
                    (first_layer.out, (-1 * np.ones(first_layer.out.shape[0])).reshape(first_layer.out.shape[0], 1)),
                    axis=1)
                act_pot = first_ext @ second_layer.weight_ext.T
                if second_layer.activation == 'sigmoid':
                    sig = np.exp(act_pot) / ( 1 + np.exp(act_pot) )
                    gamma = sig * ( 1 - sig )
                else: # softmax and cross entropy case
                    gamma = 1
                # Output Layer
                if j == (len(self.layers)-1):
                    second_layer.delta = np.ones((second_layer.neuron_num, out_error.shape[0])) * out_error.T
                #Hidden Layers
                else:
                    third_layer = self.layers[j+1]
                    weights = third_layer.weight_ext
                    weights = weights[:,:-1].reshape((weights.shape[0], weights.shape[1]-1))
                    second_layer.delta = gamma.T * (weights.T @ third_layer.delta)

        #Updating weights
        for j in range(len(self.layers)-1, -1, -1):
            if j != 0:
                second_layer, first_layer = self.layers[j], self.layers[j-1]
                X_ext = np.concatenate((first_layer.out, (-1 * np.ones(first_layer.out.shape[0])).reshape(first_layer.out.shape[0],1)), axis=1)
                update = learning_rate * (second_layer.delta @ X_ext) / batch_size
                second_layer.weight_ext += update + moment * second_layer.prev_update
                second_layer.prev_update = update + moment * second_layer.prev_update # Keep the previous update
            else:
                layer = self.layers[0]
                X_ext = np.concatenate((X, (-1 * np.ones(X.shape[0])).reshape((X.shape[0], 1))), axis=1)
                if moment == 0:
                    layer.weight_ext += learning_rate * ( layer.delta @ X_ext ) / batch_size
                else:
                    change = learning_rate * ( layer.delta @ X_ext ) / batch_size
                    layer.weight_ext += change + moment * layer.prev_update
                    layer.prev_update = change + moment * layer.prev_update

    def predict(self, X):
        Y_pred = self.forward_pass(X)
        Y_pred_raw = np.copy(Y_pred)
        Y_pred = np.argmax(Y_pred_raw, axis=1)
        return Y_pred_raw, Y_pred

    def fit(self, X, Y, X_test, Y_test, batch_size, learning_rate, epoch, moment=0):

        cross_ent_hist_tst = []
        cross_ent_hist = []
        X_train = X
        Y_train = self.oneHotEncoder(Y=Y)
        X_vall = X_test
        Y_vall = self.oneHotEncoder(Y=Y_test)

        stop_train = 0

        mceT_history = []
        mce_history = []
        prev_cross_val = 0

        batch_num = X_train.shape[0] // batch_size

        for epc in range(epoch):

            #print(f"Epoch {epc+1}")

            indices = np.random.permutation(X.shape[0])
            #Y_pred_raw_tst, Y_pred_tst = self.predict(X_vall)
            #mce = np.mean(Y_pred_tst == Y_test, axis=0)

            for j in range(batch_num):

                index = indices[j * batch_size: (j + 1) * batch_size]
                X_batch, Y_batch = X_train[index], Y_train[index]
                self.backward_pass(X_batch, Y_batch, learning_rate, batch_size, moment)

            Y_pred_raw_tst, Y_pred_tst = self.predict(X_vall)
            Y_pred_raw, Y_pred = self.predict(X_train)



            mce = np.mean(Y_pred_tst == Y_test, axis=0)
            #print("MCE Validation Accuracy:", mce)
            mce_history.append(mce)
            mce2 = np.mean(Y_pred == np.argmax(Y_train, axis=1), axis=0)
            #print("MCE Train Accuracy:", mce2)
            mceT_history.append(mce2)

            cross_entropy_val = self.cross_entropy(Y_vall, Y_pred_raw_tst)
            #print("Cross Entropy Validation Loss:", cross_entropy_val)
            cross_ent_hist_tst.append(cross_entropy_val)

            cross_entropy = self.cross_entropy(Y_train, Y_pred_raw)
            #print("Cross Entropy Train Loss:", cross_entropy)
            cross_ent_hist.append(cross_entropy)


        return {"train_cross_entropy_history": cross_ent_hist,
                "val_cross_entropy_history": cross_ent_hist_tst,
                "train_mce_history": mceT_history,
                "val_mce_history": mce_history}


    def cross_entropy(self, Y, Y_pred):
        return -np.mean(np.sum(Y * np.log(Y_pred), axis=1), axis=0)

    def oneHotEncoder(self, Y=None):
        encoded = np.zeros((Y.shape[0], 3))
        for obs in range(Y.shape[0]):
            vector = np.zeros(3)
            vector[Y[obs]] = 1
            encoded[obs,:] = vector

        return encoded

class PCA (object):

    """
    A class to represent one PCA transformation.

    Attributes
    ----------
    explained_variances : list
        explained variance by new transformed features
    num_components : int
        number of principle components

    Methods
    -------
    fit(X):
        Outputs the transformed dataset
    transform(X):
        By using eigen vectors, calculates the transformed features

    """

    def __init__(self, num_components=1):
        self.explained_variances = []
        self.num_components = num_components

    def fit(self, X):
        X_scaled = StandardScale(X)
        #Find Covariance Matrix
        cov_matrix = np.cov(X_scaled.T)
        #Eigenvalue Decomposition
        self.eig_vals, self.eig_vec = np.linalg.eig(cov_matrix)
        for i in range(0, len(self.eig_vals)):
            self.explained_variances.append(self.eig_vals[i] / np.sum(self.eig_vals))
        #print(np.sum(self.explained_variances))
        #print(self.explained_variances)
        indices = sorted(range(len(self.eig_vals)), key=lambda k: self.eig_vals[k])
        self.eig_vals = sorted(self.eig_vals, reverse=True)
        self.eig_vec = [self.eig_vec[i] for i in indices]

        return self.transform(X_scaled)

    def transform (self, X):
        projections = []
        for component in range(self.num_components):
            projections.append(X @ self.eig_vec[component])
        return pd.DataFrame(data=projections)


class MultiClassLogisticRegression(object):
    """
    A class to represent one multiclass logistic regression operation.

    X is (N,M) matrix
    W is (M,C) matrix
    Y is (N) matrix

    N: Number of observations
    M: Number of features
    C: Number of classes

    Attributes
    ----------
    ler_rate : float
        learning rate of gradient descent algorithm
    max_iter : float
        maximum number of iteration
    reg : float
        regularization term in logistic regression log loss function
    """

    def __init__(self, ler_rate=0.02, max_iter=1000, reg=0, scale=0.01):

        self.ler_rate = ler_rate * scale
        self.max_iter = max_iter
        self.reg = reg
        self.W = np.zeros((19, 3))

    def log_loss(self, X, Y, W):

        H = - X @ W
        loss = (np.trace(X @ W @ Y.T) + np.sum(np.log(np.sum(np.exp(H), axis=1)))) / X.shape[0]
        return loss

    def get_gradient(self, X, W, Y):

        H = - X @ W
        #Compute the softmax in logspace for numerical stability
        y_prob =  self.softmax(H, axis=1)#softmax(H, axis=1)
        #softmax(H, axis=1)#np.exp(H) / (np.sum(np.exp(H), axis=1, keepdims=True) * np.ones(H.shape))
        grad = (X.T @ (Y - y_prob)) + 2 * self.reg * W
        #print(y_prob)
        return grad

    def oneHotEncoder(self, Y):

        classes = np.unique(Y)
        self.classes = classes
        encoded_y = np.zeros((len(Y), len(classes)))
        for i, category in enumerate(classes):
            indices = np.where(Y == category)
            encoded = np.zeros(len(classes))
            encoded[i] = 1
            encoded_y[indices] = encoded

        return encoded_y

    def fit(self, X, Y, X_val, Y_val):
        y_train, Y_vall = self.oneHotEncoder(Y), self.oneHotEncoder(Y_val)
        X_train, X_vall = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1), np.concatenate((X_val, np.ones((X_val.shape[0], 1))), axis=1)
        W = np.zeros((X_train.shape[1], y_train.shape[1]))
        loss_hist, loss_val = [], []
        acc_train, acc_test = [], []
        for iter in range(self.max_iter):

            W -= self.ler_rate * self.get_gradient(X_train, W, y_train)
            loss_hist.append(self.log_loss(X_train, y_train, W))
            loss_val.append(self.log_loss(X_vall, Y_vall, W))
            self.W = W
            y_pred = self.predict(X_val)[1]
            #print(y_pred)
            y_predt = self.predict(X)[1]
            acc = np.sum(y_pred == np.argmax(Y_vall, axis=1), axis=0) / y_pred.shape[0]
            acct = np.sum(y_predt == np.argmax(y_train, axis=1), axis=0) / y_train.shape[0]

            acc_train.append(acct)
            acc_test.append(acc)

        return np.array(loss_hist), np.array(loss_val), np.array(acc_train), np.array(acc_test)

    def predict(self, X):

        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        H = - X @ self.W
        y_prob = np.exp(H) / (np.sum(np.exp(H), axis=1, keepdims=True) * np.ones(H.shape))
        return y_prob, self.classes[np.argmax(y_prob, axis=1)]

    def logsumexp(self, X, axis=None, keepdims=False):

        X_max = np.amax(X, axis=axis, keepdims=keepdims)
        if X_max.ndim > 0:
            X_max[~np.isfinite(X_max)] = 0
        elif not np.isfinite(X_max):
            X_max = 0
        temp = np.exp(X - X_max)
        with np.errstate(divide='ignore'):
            z = np.sum(temp, axis=axis, keepdims=keepdims)
            output = np.log(z)
        if not keepdims:
            X_max = np.squeeze(X_max, axis=axis)
        output+=X_max
        return output

    def softmax(self, X, axis=None):
        #For numerical stability, calculate the softmax in logspace
        return np.exp(X-self.logsumexp(X, axis=axis, keepdims=True))

#############################################################################################

######################################## FUNCTIONS ##########################################

def oneHotEncoder(Y):

    classes = np.unique(Y)
    encoded_y = np.zeros((len(Y), len(classes)))
    for i, category in enumerate(classes):
        indices = np.where(Y == category)
        encoded = np.zeros(len(classes))
        encoded[i] = 1
        encoded_y[indices] = encoded

    return encoded_y

def StandardScale(X):
    X -= np.mean(X, axis=0)
    return X / np.std(X, axis=0)

#Gaussian Naive Bayes Classifier

def train_classifier(clf, X_train, y_train):
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    print("Model trained in {:2f} seconds".format(end - start))


def predict_labels(clf, features, target):
    start = time()
    y_pred = clf.predict(features)
    end = time()
    print("Made Predictions in {:2f} seconds".format(end - start))

    acc = sum(target == y_pred) / float(len(y_pred))

    return acc


def model(clf, X_train, y_train, X_test, y_test):
    train_classifier(clf, X_train, y_train)

    acc = predict_labels(clf, X_train, y_train)
    print("Training Info:")
    print("-" * 20)
    print("Accuracy:{}".format(acc))

    f1, acc = predict_labels(clf, X_test, y_test)
    print("Test Metrics:")
    print("-" * 20)
    print("Accuracy:{}".format(acc))


def derive_clean_sheet(src):
    arr = []
    n_rows = src.shape[0]

    for data in range(n_rows):

        # [HTHG, HTAG]
        values = src.iloc[data].values
        cs = [0, 0]

        if values[0] == 0:
            cs[1] = 1

        if values[1] == 0:
            cs[0] = 1

        arr.append(cs)

    return arr

def dataClassifier(X_train):
    dfA = pd.DataFrame()
    dfH = pd.DataFrame()
    dfD = pd.DataFrame()

    X_train = pd.DataFrame(X_train)
    for i in range(len(X_train)):

        if X_train.iloc[i]['HTR'] == "A":
            newRow = X_train.iloc[i]
            dfA = dfA.append(newRow)
        elif X_train.iloc[i]['HTR'] == "H":
            newRow = X_train.iloc[i]
            dfH = dfH.append(newRow)
        elif X_train.iloc[i]['HTR'] == "D":
            newRow = X_train.iloc[i]
            dfD = dfD.append(newRow)

    # dfA = dfA.drop("HTR", axis=1)

    # dfH = dfH.drop("HTR", axis=1)

    # dfD = dfD.drop("HTR", axis=1)

    return dfA, dfH, dfD

def meanAndVar(x, y, z):
    '''FOR HOME WINS'''
    mValHome = x.mean()
    vValHome = x.var()

    '''FOR AWAY WINS'''
    mValAway = y.mean()
    vValAway = y.var()

    '''FOR DRAW'''
    mValDraw = z.mean()
    vValDraw = z.var()

    return mValHome, mValDraw, mValAway, vValHome, vValDraw, vValAway

def get_prior_probs(h, a, d, trainData):
    # Prob of Home wins / whole data
    priorHome = len(h) / len(trainData)

    # Prob of Away wins / whole data
    priorAway = len(a) / len(trainData)

    # Prob of Draw / whole data
    priorDraw = len(d) / len(trainData)

    return priorHome, priorAway, priorDraw

def calculate_pdf(newInputValue, trainMean, trainVar):
    probDensity = (1 / np.sqrt(2 * np.pi * trainVar)) / np.exp(-1 / 2 * ((newInputValue - trainMean) / (trainVar)))
    return probDensity


def prior_prob_frame(mValHome, mValDraw, mValAway, vValHome, vValDraw, vValAway, X_train):
    checkIndex = ["HTHG", "HTAG", "HST", "AST", "HS", "AS"]

    priorFrameforHome = pd.DataFrame()
    priorFrameforAway = pd.DataFrame()
    priorFrameforDraw = pd.DataFrame()

    for i in checkIndex:
        x = calculate_pdf(X_train[i], mValHome[i], vValHome[i])
        priorFrameforHome[i] = x

    for i in checkIndex:
        x = calculate_pdf(X_train[i], mValAway[i], vValAway[i])
        priorFrameforAway[i] = x

    for i in checkIndex:
        x = calculate_pdf(X_train[i], mValDraw[i], vValDraw[i])
        priorFrameforDraw[i] = x

    return priorFrameforHome, priorFrameforAway, priorFrameforDraw

def final_result_probabilities(probFrameForHome, probFrameForAway, probFrameForDraw):
    productAway = probFrameForAway.product(axis=1)

    productHome = probFrameForHome.product(axis=1)

    productDraw = probFrameForDraw.product(axis=1)

    draw = productDraw * priDraw
    away = productAway * priAway
    home = productHome * priHome

    drawResult = np.log(draw)

    awayResult = np.log(away)

    homeResult = np.log(home)

    finalProbFrame = pd.DataFrame()

    finalProbFrame['Home Wins'] = homeResult

    finalProbFrame['Away Wins'] = awayResult

    finalProbFrame['Draw'] = drawResult

    return finalProbFrame

def accuracy_calculator(finalProbFrame, Y_train):
    finalArr = finalProbFrame.to_numpy()

    finalLoc = []
    for i in range(len(finalArr)):
        finalLoc.append(np.argmax(finalArr[i]))

    finalLocName = []
    for i in finalLoc:
        if i == 0:
            finalLocName.append('H')
        elif i == 1:
            finalLocName.append('A')
        elif i == 2:
            finalLocName.append('D')

    arrPred = np.array(finalLocName)
    arrTrain = np.array(Y_train)

    accuracy = sum(arrPred == arrTrain) / arrPred.shape[0]
    return accuracy, finalLoc

def train_test_split(X, y, split=0.2, shuffle=False):
    np.random.seed(10) # Put a random seed so that there will be no data leakage
    indices = np.arange(0, len(y))
    if shuffle:
        np.random.shuffle(indices)
    test_limit = round(len(y) * split)
    X_test, y_test = X[indices[-test_limit:]], y[indices[-test_limit:]]
    X_train, y_train = X[indices[:-test_limit]], y[indices[:-test_limit]]
    return X_train, y_train, X_test, y_test

def label_encoder(y):
    temp = y
    unique = np.unique(temp)
    for i in range(len(unique)):
        temp = np.where(temp == unique[i], i, temp)
    return temp

def confusion_matrix(y_pred, y):
    class_num = len(np.unique(y))
    matrix = np.zeros((class_num, class_num))
    for pred in range(len(y)):
        matrix[y[pred]][y_pred[pred]] += 1
    return matrix

def train_logreg_cv(params, X, y, k):

    log_t_losses = []
    log_v_losses = []
    mce_acc_train = []
    mce_acc_test = []

    kfolds = np.random.choice(np.arange(0, len(X)), size=(k, len(X) // k), replace=False)

    for i in range(len(params)):
        learning_rate = params[i]
        logv_loss, logt_loss, mce_train, mce_test = 0, 0, 0, 0

        for fold in range(k):
            ############################################   LOGISTIC REGRESSION    ############################################
            x_val, y_val = X[kfolds[fold]], y[kfolds[fold]]
            x_train, y_train = X[(np.delete(kfolds, fold, axis=0)).reshape(-1)], y[(np.delete(kfolds, fold, axis=0)).reshape(-1)]
            model = MultiClassLogisticRegression(ler_rate=learning_rate, max_iter=50)
            logt, logv, acctr, acc = model.fit(x_train, y_train, x_val, y_val)


            logv_loss += logv / k
            logt_loss += logt / k
            mce_train += acctr / k
            mce_test += acc / k

        print("#" * 20, f"Model {i + 1}", "#" * 20)
        print("Mean Logarithmic Loss (Train):", logv_loss[-1])
        print("Mean Logarithmic Loss (Validation):", logt_loss[-1])
        print("Mean Classification Accuracy (Train):", mce_train[-1])
        print("Mean Classification Accuracy (Validation):", mce_test[-1])

        log_t_losses.append(logt_loss)
        log_v_losses.append(logv_loss)
        mce_acc_train.append(mce_train)
        mce_acc_test.append(mce_test)

    plt.figure()
    plt.title("Log Loss vs. Epoch (Train)")
    plt.xlabel("Epoch")
    plt.ylabel("Log Loss")
    for i in range(len(params)):
        plt.plot(log_t_losses[i])
    plt.legend([f"Model {i + 1}" for i in range(len(params))])
    plt.show()

    plt.figure()
    plt.title("Log Loss vs. Epoch (Validation)")
    plt.xlabel("Epoch")
    plt.ylabel("Log Loss")
    for i in range(len(params)):
        plt.plot(log_v_losses[i])
    plt.legend([f"Model {i + 1}" for i in range(len(params))])
    plt.show()

    plt.figure()
    plt.title("Mean Classification Accuracy vs. Epoch (Train)")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Classification Accuracy")
    for i in range(len(params)):
        plt.plot(mce_acc_train[i])
    plt.legend([f"Model {i + 1}" for i in range(len(params))])
    plt.show()

    plt.figure()
    plt.title("Mean Classification Accuracy vs. Epoch (Validation)")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Classification Accuracy")
    for i in range(len(params)):
        plt.plot(mce_acc_test[i])
    plt.legend([f"Model {i + 1}" for i in range(len(params))])
    plt.show()

def train_neural_cv(params, X, y, k):

    kfolds = np.random.choice(np.arange(0, len(X)), size=(k, len(X) // k), replace=False)

    models = []

    t_cross, v_cross, t_mce, v_mce = [], [], [], []


    for model in range(len(params)):

        n1, n2, batch_size, learning_rate = params[model]
        train_cross, val_cross, train_mce, val_mce = [0,0,0,0]

        for fold in range(k):

            #print("*"*20, f"Fold {fold+1}", "*"*20)

            x_val, y_val = X[kfolds[fold]], y[kfolds[fold]]
            x_train, y_train = X[(np.delete(kfolds, fold, axis=0)).reshape(-1)], y[(np.delete(kfolds, fold, axis=0)).reshape(-1)]

            net = NeuralNetwork(early_stopping=30)
            net.add_layer(Layer(neuron_num=n1, input_dim=X_scaled.shape[1], activation="sigmoid"))
            net.add_layer(Layer(neuron_num=n2, input_dim=n1, activation="sigmoid"))
            net.add_layer(Layer(neuron_num=3, input_dim=n2, activation="softmax"))
            history = net.fit(X=x_train, Y=y_train, X_test=x_val, Y_test=y_val,
                              batch_size=batch_size, learning_rate=learning_rate, epoch=80)

            train_c, val_c = np.array(history["train_cross_entropy_history"]), np.array(history["val_cross_entropy_history"])
            train_m, val_m = np.array(history["train_mce_history"]), np.array(history["val_mce_history"])

            train_cross += train_c/k
            val_cross += val_c/k
            train_mce += train_m/k
            val_mce += val_m/k

        print("#"*20, f"Model {model+1}", "#"*20)
        print("Mean Classification Accuracy (Train):", train_mce[-1])
        print("Mean Classification Accuracy (Validation):", val_mce[-1])
        print("Cross-Entropy Loss (Train):", train_cross[-1])
        print("Cross-Entropy Loss (Validation):", val_cross[-1])

        t_cross.append(train_cross)
        v_cross.append(val_cross)
        t_mce.append(train_mce)
        v_mce.append(val_mce)

        models.append(val_cross[-1])

    models = np.array(models)
    best_model = np.argmax(models)
    print(f"Best model is the model {best_model}")

    plt.figure()
    plt.title("Cross Entropy Loss vs. Epoch (Train)")
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    for i in range(len(models)):
        plt.plot(t_cross[i])
    plt.legend([f"Model {i+1}" for i in range(len(params))])
    plt.show()

    plt.figure()
    plt.title("Cross Entropy Loss vs. Epoch (Validation)")
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    for i in range(len(models)):
        plt.plot(v_cross[i])
    plt.legend([f"Model {i+1}" for i in range(len(params))])
    plt.show()

    plt.figure()
    plt.title("Mean Classification Accuracy vs. Epoch (Train)")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Classification Accuracy")
    for i in range(len(models)):
        plt.plot(t_mce[i])
    plt.legend([f"Model {i+1}" for i in range(len(params))])
    plt.show()

    plt.figure()
    plt.title("Mean Classification Accuracy vs. Epoch (Validation)")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Classification Accuracy")
    for i in range(len(models)):
        plt.plot(v_mce[i])
    plt.legend([f"Model {i+1}" for i in range(len(params))])
    plt.show()

    return best_model

def display_performance(y_pred, y):

    classes = np.sort(np.unique(y))
    scores = []
    for i in range(len(classes)):
        TP, TN, FP, FN = 0, 0, 0, 0
        recall, precision, f1 = 0, 0, 0
        cl = classes[i] # choose the class
        for j in range(len(y)):
            # Make other classes 0 except the positive class

            # Check whether the sample is positive value
            if y[j] == cl:
                # Sample is positive
                if y_pred[j] == y[j]:
                    TP += 1
                else:
                    FN += 1
            else:
                # Sample is negative
                if y_pred[j] == y[j]:
                    TN += 1
                else:
                    FP += 1

        recall, precision = TP / (TP + FN), TP / (TP + FP)
        f1 = (2 * recall * precision) / (recall + precision)

        scores.append(np.array([recall, precision, f1]))

    return scores





        #y_pred = model.predict(x_val)
        #acc = sum(y_pred == y_val) / y_pred.shape[0]
        #print("Logistic Regression")
        #print("-" * 20)
        #print(f"Accuracy for fold {fold + 1} is {acc}")
        #print()
        #accuracy.append(acc)


#############################################################################################


# Data gathering
en_data_folder = 'english-premier-league_zip'
es_data_folder = 'spanish-la-liga_zip'
fr_data_folder = 'french-ligue-1_zip'
ge_data_folder = 'german-bundesliga_zip'
it_data_folder = 'italian-serie-a_zip'

# data_folders = [es_data_folder]
data_folders = [en_data_folder, es_data_folder,
                fr_data_folder, ge_data_folder, it_data_folder]

season_range = (9, 18)

data_files = []
for data_folder in data_folders:
    for season in range(season_range[0], season_range[1] + 1):
        data_files.append(
            'data/{}/data/season-{:02d}{:02d}_csv.csv'.format(data_folder, season, season + 1))

data_frames = []

for data_file in data_files:
    if path.exists(data_file):
        data_frames.append(pd.read_csv(data_file))

data = pd.concat(data_frames).reset_index()
temp = data
#Drop the features having more than 100 NaN values.
input_filter = []
for feature in data.columns:
    if np.sum(data[feature].isna(), axis=0) < 100:
        input_filter.append(feature)

new_data = data[input_filter].dropna(axis=0)

#Drop Irrelevant Features
y, X = new_data["FTR"].values, new_data.drop(labels=["Div", "Date", "HomeTeam", "AwayTeam",
                                     "FTR", "HTR", "index"], axis=1)

y = label_encoder(y)
X.drop(columns=['FTHG', 'FTAG', 'HTAG'], axis=1, inplace=True)
#STANDARD SCALING
X_scaled = StandardScale(X)

#X_scaled, y = np.asarray(X_scaled).astype(np.int), np.asarray(y).astype(np.int)

x_train, y_train, x_test, y_test = train_test_split(X=X_scaled.values, y=y, split=0.2, shuffle=True)

# Parameters for hyperparameter tuning are as follows: neuron 1, neuron 2, batch_size, learning rate
params = [[32, 16, 500, 0.2],
          [64, 16, 500, 0.2],
          [128, 16, 500, 0.2],
          [128, 32, 250, 0.2],
          [128, 64, 250, 0.5],
          [128, 32, 250, 0.5],
          [128, 32, 250, 0.5],
          [128, 64, 250, 0.5],
          [256, 128, 500, 0.5],
          [256, 64, 500, 0.2]]
#train_neural_cv(params, x_train, y_train, 5)


print("#"*30, "FINAL TESTS WITH AND WITHOUT PCA", "#"*30)
############################################   LOGISTIC REGRESSION    ############################################

#PRINCIPLE COMPONENT ANALYSIS
X_transformed = PCA(num_components=18).fit(X_scaled).values.T

x_train, y_train, x_test, y_test = train_test_split(X=X_transformed, y=y, split=0.2, shuffle=True)

params = [0.0025, 0.005, 0.0075, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
#train_logreg_cv(params, x_train, y_train, k=5)


# learning rate of 0.01 is the best one
print("Logistic Regression is being trained...")
# Test the Logistic Regression on Unseen Data
model = MultiClassLogisticRegression(ler_rate=0.1, max_iter=100)
log_train, log_test, acc_train, acc_test = model.fit(x_train, y_train, x_test, y_test)
y_pred1, labels = model.predict(x_test)

A, D, H = display_performance(labels, y_test)
A, D, H = np.around(A, 4), np.around(D, 4), np.around(H, 4)

print("*"*20, "Logistic Regression", "*"*20)
print("Positive Class is chosen as A = Away Team Wins")
print("Precision:", A[1])
print("Recall:", A[0])
print("F1 Score", A[2])
print("-"*50)
print("Positive Class is chosen as D = Draw")
print("Precision:", D[1])
print("Recall:", D[0])
print("F1 Score", D[2])
print("-"*50)
print("Positive Class is chosen as H = Home Team Wins")
print("Precision:", H[1])
print("Recall:", H[0])
print("F1 Score", H[2])
print("-"*50)
print("Mean Classification Accuracy:", acc_test[-1])
print("Log Loss:", log_test[-1])

############################################   FEED-FORWARD NEURAL NETWORK    ############################################

x_train, y_train, x_test, y_test = train_test_split(X=X_scaled.values, y=y, split=0.2, shuffle=True)

n1, n2, batch_size, learning_rate = 128, 64, 250, 0.5
net = NeuralNetwork(early_stopping=30)
net.add_layer(Layer(neuron_num=n1, input_dim=X_scaled.shape[1], activation="sigmoid"))
net.add_layer(Layer(neuron_num=n2, input_dim=n1, activation="sigmoid"))
net.add_layer(Layer(neuron_num=3, input_dim=n2, activation="softmax"))
history = net.fit(X=x_train, Y=y_train, X_test=x_test, Y_test=y_test,
                                                    batch_size=batch_size, learning_rate=learning_rate, epoch=80)
y_pred2, labels = net.predict(x_test)

A, D, H = display_performance(labels, y_test)
A, D, H = np.around(A, 4), np.around(D, 4), np.around(H, 4)

print("*"*20, "Feed-forward Neural Network", "*"*20)
print("Positive Class is chosen as A = Away Team Wins")
print("Precision:", A[1])
print("Recall:", A[0])
print("F1 Score", A[2])
print("-"*50)
print("Positive Class is chosen as D = Draw")
print("Precision:", D[1])
print("Recall:", D[0])
print("F1 Score", D[2])
print("-"*50)
print("Positive Class is chosen as H = Home Team Wins")
print("Precision:", H[1])
print("Recall:", H[0])
print("F1 Score", H[2])
print("-"*50)
print("Mean Classification Accuracy:", history["val_mce_history"][-1])
print("Cross-Entropy Loss:", history["val_cross_entropy_history"][-1])

y_pred_ens = (y_pred1 + y_pred2)/2
A, D, H = display_performance(np.argmax(y_pred_ens, axis=1), y_test)
A, D, H = np.around(A, 4), np.around(D, 4), np.around(H, 4)

print("*"*20, "Ensemble (Logistic Regression+Neural Network)", "*"*20)
print("Positive Class is chosen as A = Away Team Wins")
print("Precision:", A[1])
print("Recall:", A[0])
print("F1 Score", A[2])
print("-"*50)
print("Positive Class is chosen as D = Draw")
print("Precision:", D[1])
print("Recall:", D[0])
print("F1 Score", D[2])
print("-"*50)
print("Positive Class is chosen as H = Home Team Wins")
print("Precision:", H[1])
print("Recall:", H[0])
print("F1 Score", H[2])
print("-"*50)
print("Mean Classification Accuracy:", (history["val_mce_history"][-1]+acc_test[-1])/2)



print("#"*30, "FINAL TESTS WITH MANUAL FEATURE SELECTION", "#"*30)

input_filter = ["HTHG","HST","AST","HS", "AS","HC", "AC", "HTAG"]
objects = ["FTR", "HTR"]
data[input_filter] = StandardScale(data[input_filter])

#Drop Irrelevant Features
y, X = new_data["FTR"].values, new_data[input_filter]

y = label_encoder(y)
#STANDARD SCALING
X_scaled = StandardScale(X)

#X_scaled, y = np.asarray(X_scaled).astype(np.int), np.asarray(y).astype(np.int)

x_train, y_train, x_test, y_test = train_test_split(X=X_scaled.values, y=y, split=0.2, shuffle=True)

# learning rate of 0.01 is the best one
print("Logistic Regression is being trained...")
# Test the Logistic Regression on Unseen Data
model = MultiClassLogisticRegression(ler_rate=0.1, max_iter=100)
log_train, log_test, acc_train, acc_test = model.fit(x_train, y_train, x_test, y_test)
y_pred1, labels = model.predict(x_test)

A, D, H = display_performance(labels, y_test)
A, D, H = np.around(A, 4), np.around(D, 4), np.around(H, 4)
print("#"*20, "FINAL TEST RESULTS", "#"*20)
print("*"*20, "Logistic Regression", "*"*20)
print("Positive Class is chosen as A = Away Team Wins")
print("Precision:", A[1])
print("Recall:", A[0])
print("F1 Score", A[2])
print("-"*50)
print("Positive Class is chosen as D = Draw")
print("Precision:", D[1])
print("Recall:", D[0])
print("F1 Score", D[2])
print("-"*50)
print("Positive Class is chosen as H = Home Team Wins")
print("Precision:", H[1])
print("Recall:", H[0])
print("F1 Score", H[2])
print("-"*50)
print("Mean Classification Accuracy:", acc_test[-1])
print("Log Loss:", log_test[-1])

############################################   FEED-FORWARD NEURAL NETWORK    ############################################

x_train, y_train, x_test, y_test = train_test_split(X=X_scaled.values, y=y, split=0.2, shuffle=True)

n1, n2, batch_size, learning_rate = 128, 64, 250, 0.5
net = NeuralNetwork(early_stopping=30)
net.add_layer(Layer(neuron_num=n1, input_dim=X_scaled.shape[1], activation="sigmoid"))
net.add_layer(Layer(neuron_num=n2, input_dim=n1, activation="sigmoid"))
net.add_layer(Layer(neuron_num=3, input_dim=n2, activation="softmax"))
history = net.fit(X=x_train, Y=y_train, X_test=x_test, Y_test=y_test,
                                                    batch_size=batch_size, learning_rate=learning_rate, epoch=80)
y_pred2, labels = net.predict(x_test)

A, D, H = display_performance(labels, y_test)
A, D, H = np.around(A, 4), np.around(D, 4), np.around(H, 4)

print("*"*20, "Feed-forward Neural Network", "*"*20)
print("Positive Class is chosen as A = Away Team Wins")
print("Precision:", A[1])
print("Recall:", A[0])
print("F1 Score", A[2])
print("-"*50)
print("Positive Class is chosen as D = Draw")
print("Precision:", D[1])
print("Recall:", D[0])
print("F1 Score", D[2])
print("-"*50)
print("Positive Class is chosen as H = Home Team Wins")
print("Precision:", H[1])
print("Recall:", H[0])
print("F1 Score", H[2])
print("-"*50)
print("Mean Classification Accuracy:", history["val_mce_history"][-1])
print("Cross-Entropy Loss:", history["val_cross_entropy_history"][-1])

y_pred_ens = (y_pred1 + y_pred2)/2
A, D, H = display_performance(np.argmax(y_pred_ens, axis=1), y_test)
A, D, H = np.around(A, 4), np.around(D, 4), np.around(H, 4)

print("*"*20, "Ensemble (Logistic Regression+Neural Network)", "*"*20)
print("Positive Class is chosen as A = Away Team Wins")
print("Precision:", A[1])
print("Recall:", A[0])
print("F1 Score", A[2])
print("-"*50)
print("Positive Class is chosen as D = Draw")
print("Precision:", D[1])
print("Recall:", D[0])
print("F1 Score", D[2])
print("-"*50)
print("Positive Class is chosen as H = Home Team Wins")
print("Precision:", H[1])
print("Recall:", H[0])
print("F1 Score", H[2])
print("-"*50)
print("Mean Classification Accuracy:", (history["val_mce_history"][-1]+acc_test[-1])/2)



'''
############################################   NAIVE BAYES CLASSIFIER    ############################################

input_filter = ["HTHG","HST","AST","HS", "AS","HC", "AC", "HTAG"]
objects = ["FTR", "HTR"]
data[input_filter] = StandardScale(data[input_filter])
cols_to_consider = input_filter + objects
data = data[cols_to_consider]

X = data[input_filter]
Y = data['HTR']
Z = data["FTR"]

x_train, y_train, x_test, y_test = train_test_split(X=X.values, y=Y, split=0.2, shuffle=True)
x_train, x_test = pd.DataFrame(x_train, columns=input_filter), pd.DataFrame(x_test, columns=input_filter)
x_train["HTR"] = y_train
x_test["HTR"] = y_test



#Drop Irrelevant Features


#y, X = new_data["FTR"].values, new_data.drop(labels=["Div", "Date", "HomeTeam", "AwayTeam", "index"], axis=1)
#X = X[input_filter]

#x_train, y_train, x_test, y_test = pd.DataFrame(x_train, columns=input_filter), y_train, \
 #                                  pd.DataFrame(x_test, columns=input_filter), y_test



dfA, dfH, dfD = dataClassifier(x_test)

mValHome, mValDraw, mValAway, vValHome, vValDraw, vValAway = meanAndVar(dfH[input_filter], dfA[input_filter],
                                                                        dfD[input_filter])
priHome, priAway, priDraw = get_prior_probs(dfH, dfA, dfD, x_test)

mValHome_test, mValDraw_test, mValAway_test, vValHome_test, vValDraw_test, vValAway_test = meanAndVar(
    dfH[input_filter], dfA[input_filter], dfD[input_filter])

priHome_test, priAway_test, priDraw_test = get_prior_probs(dfH, dfA, dfD, x_test)

probFrameForHome_test, probFrameForAway_test, probFrameForDraw_test = prior_prob_frame(mValHome_test, mValDraw_test,
                                                                                       mValAway_test, vValHome_test,
                                                                                       vValDraw_test, vValAway_test,
                                                                                       x_test)

y_pred3 = final_result_probabilities(probFrameForHome_test, probFrameForAway_test, probFrameForDraw_test)
results, loc = accuracy_calculator(y_pred3, y_train)
print(np.unique(loc))
print(loc)
A, D, H = display_performance(loc, y_test)
A, D, H = np.around(A, 4), np.around(D, 4), np.around(H, 4)

print("*"*20, "Feed-forward Neural Network", "*"*20)
print("Positive Class is chosen as A = Away Team Wins")
print("Precision:", A[1])
print("Recall:", A[0])
print("F1 Score", A[2])
print("-"*50)
print("Positive Class is chosen as D = Draw")
print("Precision:", D[1])
print("Recall:", D[0])
print("F1 Score", D[2])
print("-"*50)
print("Positive Class is chosen as H = Home Team Wins")
print("Precision:", H[1])
print("Recall:", H[0])
print("F1 Score", H[2])
print("-"*50)
#print("Mean Classification Accuracy:", history["val_mce_history"][-1])
#print("Cross-Entropy Loss:", history["val_cross_entropy_history"][-1])


y_pred_ens = (y_pred1 + y_pred2)/2
A, D, H = display_performance(y_pred_ens, y_test)
A, D, H = np.around(A, 4), np.around(D, 4), np.around(H, 4)

print("*"*20, "Ensemble (Logistic Regression+Neural Network)", "*"*20)
print("Positive Class is chosen as A = Away Team Wins")
print("Precision:", A[1])
print("Recall:", A[0])
print("F1 Score", A[2])
print("-"*50)
print("Positive Class is chosen as D = Draw")
print("Precision:", D[1])
print("Recall:", D[0])
print("F1 Score", D[2])
print("-"*50)
print("Positive Class is chosen as H = Home Team Wins")
print("Precision:", H[1])
print("Recall:", H[0])
print("F1 Score", H[2])
print("-"*50)
print("Mean Classification Accuracy:", (history["val_mce_history"][-1]+acc_test[-1])/2)
'''