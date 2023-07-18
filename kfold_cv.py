
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import tensorflow
import numpy as np

early_stopping = tensorflow.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

# Function which performs the K-Fold Cross Validation
def kfoldCrossValidation(k_folds, feature, label, network, hyperparams_combination, epochs):

    # Stratified K-fold Cross Validation
    stratified_kfold = MultilabelStratifiedKFold(n_splits = k_folds, random_state = 19, shuffle = True)
    
    # Lists with performance metrics for each iteration
    accuracy_kfold = []
    loss_kfold = []
    epochs_kfold = []

    # Compiling the network
    network.compile(
        loss = 'categorical_crossentropy', 
        optimizer = hyperparams_combination['optimizer'], 
        metrics = ['accuracy']
    )

    # Converting to numpy for splitting
    feature = np.array(feature)

    k = 0

    # Splitting in training and validation set
    for train, val in stratified_kfold.split(feature, label):

        k = k + 1
        print(k)

        feature_train = tensorflow.convert_to_tensor(feature[train])
        feature_val = tensorflow.convert_to_tensor(feature[val])

        # Training (fit Neural Network)
        training_history = network.fit(

            x = feature_train,
            y = label[train],
            batch_size = hyperparams_combination['batch_size'],
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

    return {

        'Network': network.name,
        'Embedding': network.layers[1].name,
        'k_folds': k_folds,
        'filters': hyperparams_combination['filters'],
        'kernel_size': hyperparams_combination['kernel_size'],
        'rate': hyperparams_combination['rate'],
        'optimizer': hyperparams_combination['optimizer'],
        'batch_size': hyperparams_combination['batch_size'],
        'loss_kfold': round(np.mean(loss_kfold), 3),
        'accuracy_kfold': round(np.mean(accuracy_kfold), 3),
        'best_number_epochs': np.min(epochs_kfold),
        'n_epochs': epochs

    }