
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import tensorflow
import numpy as np

early_stopping = tensorflow.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

# Function which performs the K-Fold Cross Validation
def kfoldCrossValidation(k_folds, feature, label, network, hyperparams, epochs):

    neural_network = tensorflow.keras.models.clone_model(network)

    # Stratified K-fold Cross Validation
    stratified_kfold = MultilabelStratifiedKFold(n_splits = k_folds, random_state = 19, shuffle = True)

    results = []
    
    for hyperparams_combination in hyperparams:

        # Lists with performance metrics for each iteration
        accuracy_kfold = []
        loss_kfold = []
        epochs_kfold = []
    
        # Neural Network architecture with hyperparameters combination
        neural_network.layers[2].filters = hyperparams_combination['filters']
        neural_network.layers[2].kernel_size = (hyperparams_combination['kernel_size'],)
        neural_network.layers[4].rate = hyperparams_combination['rate']

        # Compiling the network
        neural_network.compile(
            loss = 'categorical_crossentropy', 
            optimizer = hyperparams_combination['optimizer'], 
            metrics = ['accuracy']
        )

        # Converting to numpy for splitting
        feature = np.array(feature)

        # Splitting in training and validation set
        for train, val in stratified_kfold.split(feature, label):

            feature_train = tensorflow.convert_to_tensor(feature[train])
            feature_val = tensorflow.convert_to_tensor(feature[val])

            # Training (fit Neural Network)
            training_history = neural_network.fit(

                x = feature_train,
                y = label[train],
                batch_size = hyperparams_combination['batch_size'],
                epochs = epochs,
                callbacks = [early_stopping]

            )

            # Best number of epochs
            accuracy_train_history = training_history.history['accuracy']
            best_epochs = accuracy_train_history.index(max(accuracy_train_history)) + 1
            
            # Validation 
            score = neural_network.evaluate(feature_val, label[val], verbose = 0)

            # Performance metrics
            loss_kfold.append(score[0])
            accuracy_kfold.append(score[1])
            epochs_kfold.append(best_epochs)

        # KFold CV results for hyperparams combination
        results.append({

            'Network': neural_network.name,
            'Embedding': neural_network.layers[1].name,
            'k_folds': k_folds,
            'filters': hyperparams_combination['filters'],
            'kernel_size': hyperparams_combination['kernel_size'],
            'rate': hyperparams_combination['rate'],
            'optimizer': hyperparams_combination['optimizer'],
            'batch_size': hyperparams_combination['batch_size'],
            'loss_kfold': round(np.mean(loss_kfold), 3),
            'accuracy_kfold': round(np.mean(accuracy_kfold), 3),
            'best_number_epochs': np.min(best_epochs),
            'n_epochs': epochs

        })

    return results