
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from keras.backend import clear_session
import tensorflow
import numpy as np

# Function which performs the K-Fold Cross Validation
def kfoldCrossValidation(k_folds, feature, label, network, network_info, epochs):

    # Stratified K-fold Cross Validation
    stratified_kfold = MultilabelStratifiedKFold(n_splits = k_folds, random_state = 19, shuffle = True)

    # Early stopping object
    early_stopping = tensorflow.keras.callbacks.EarlyStopping(monitor = 'loss', patience = 3)
    
    # Lists with performance metrics for each iteration
    accuracy_kfold = []
    loss_kfold = []
    epochs_kfold = []

    # Compiling the network
    network.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    # Converting to numpy for splitting
    feature = np.array(feature)

    k = 0

    # Initial weights
    starting_weights = network.get_weights()

    # Splitting in training and validation set
    for train, val in stratified_kfold.split(feature, label):

        k = k + 1
        print(k)

        feature_train = tensorflow.convert_to_tensor(feature[train])
        feature_val = tensorflow.convert_to_tensor(feature[val])

        # Reset the network weights (starting from scratch)
        network.set_weights(starting_weights)

        # Training (fit Neural Network)
        training_history = network.fit(

            x = feature_train,
            y = label[train],
            batch_size = 512,
            epochs = epochs,
            callbacks = [ early_stopping ],
            verbose = 0

        )

        # Best number of epochs
        loss_train_history = training_history.history['loss']
        best_epochs = loss_train_history.index(min(loss_train_history))
        
        # Validation 
        score = network.evaluate(feature_val, label[val], verbose = 0)

        # Performance metrics
        loss_kfold.append(score[0])
        accuracy_kfold.append(score[1])
        epochs_kfold.append(best_epochs)

    # New information after K-Fold CV
    network_info['k_folds'] = 3
    network_info['best_number_epochs'] = round(np.mean(epochs_kfold), 0)
    network_info['optimizer'] = 'adam'
    network_info['rate'] = 0.5
    network_info['batch_size'] = 512
    network_info['loss'] = round(np.mean(loss_kfold), 3)
    network_info['accuracy'] = round(np.mean(accuracy_kfold), 3)

    return network_info