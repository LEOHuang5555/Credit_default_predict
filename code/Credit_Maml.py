import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import random
from sklearn.metrics import roc_curve, roc_auc_score, auc, confusion_matrix, RocCurveDisplay

class Credit_Maml:
    def __init__(self, train_X:np.ndarray, train_y:np.ndarray, n_meta_update:int=1, n_task:int=100, ways:list=[0,1], n_shots:int=5) -> None:
        self.n_meta_update = n_meta_update
        self.n_task = n_task
        self.ways = ways
        self.n_ways = len(ways)
        self.n_shots = n_shots
        self.train_X = train_X
        self.train_y = train_y
        self.train_y2D = train_y.reshape(train_y.shape[0], 1)
        self.seed_series = self.create_random_seeds('seed_sets')
        self.val_seed_series = self.create_random_seeds('val_seed_sets')
       


    def create_random_seeds(self, file_name):
        seed_sets = np.zeros((self.n_meta_update, self.n_task))
        for i in range(self.n_meta_update):
            seeds = np.random.choice(self.train_X.shape[0], self.n_task)
            seed_sets[i] = seeds
        np.save(f'{file_name}.npy', seed_sets)
        # 讀取預先製作的亂數組
        seed_sets = seed_sets.astype(int)
        return seed_sets
    
    def chose_one_class(self, random_state, way):
        # 每個 way 抽出 n_shot 筆資料
        np.random.seed(random_state)
        rand_index = np.random.choice(np.where(self.train_y==way)[0], self.n_shots)
        return self.train_X[rand_index], self.train_y2D[rand_index]
         
    
    def build_task_sample(self, random_state):
        # 建立 task 資料
        # train_X = train_X.values
        task_x = np.zeros((1, self.n_ways, self.n_shots, *self.train_X.shape[1:]))
        task_y = np.zeros((1, self.n_ways, self.n_shots, *self.train_y2D.shape[1:]))
        for i in range(self.n_ways):
            task_x[0][i], task_y[0][i] = self.chose_one_class(random_state, self.ways[i])
        return task_x, task_y
    
    def generate_multiple_tasks(self, seed_series):
        # 建立 task set:(n_task, n_ways, n_shot, n_features)
        task_sets_x = np.zeros((self.n_task, self.n_ways, self.n_shots, *self.train_X.shape[1:]))
        task_sets_y = np.zeros((self.n_task, self.n_ways, self.n_shots, *self.train_y2D.shape[1:]))
        for i in range(self.n_task):
            task_sets_x[i], task_sets_y[i] = self.build_task_sample(seed_series[i])
        y_to_categorical = tf.keras.utils.to_categorical(task_sets_y, self.n_ways)
        # y = np.zeros((self.n_task, self.n_ways, self.n_shots, 1))
        # y[:,0,:,:] = [1,0]
        # y[:,1,:,:] = [0,1]
        return task_sets_x, y_to_categorical
    
    def create_model(self, n_features):
        # 清除背景中沒在使用的 model
        tf.keras.backend.clear_session()
        # model 為 3 層之 MLP
        # set initial weights of model
        initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(64, input_shape=(self.n_ways, self.n_shots, *n_features), activation='relu', kernel_initializer=initializer))
        # model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        # model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(2, activation='softmax'))
        # opt = tf.keras.optimizers.Adam()
        # model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=opt, metrics=[tf.keras.metrics.BinaryAccuracy()])
        return model
    
    
    def CNN_model(self, n_features):
        # 清除背景中沒在使用的 model
        tf.keras.backend.clear_session()
        initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(self.n_ways, self.n_shots, *n_features, 1)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                # tf.keras.layers.MaxPooling3D((2, 2, 2)),
                # tf.keras.layers.Flatten(),
                # tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(2, activation='softmax')
            ])
        return model


    def train_step(self, inner_lr=0.001, outer_lr=0.0001, early_stop = 10, select_model = 'MLP'):
        best_val_loss = np.inf
        best_weights = None
        # Create early stopping callback
        patience = early_stop
        early_stop_counter = 0
        history = {'loss': [], 'val_loss': []}
        opt_inner = tf.keras.optimizers.Adam(learning_rate=inner_lr)
        opt_outer = tf.keras.optimizers.Adam(learning_rate=outer_lr)
        if select_model == 'MLP':
            model = self.create_model(n_features=self.train_X.shape[1:])
        else:
            self.train_X = series_to_image(self.train_X, 11)
            model = self.CNN_model(n_features=(11,11))
            # print(self.train_X.shape)
        for i in range(self.n_meta_update):
            task_sets, y_true = self.generate_multiple_tasks(self.seed_series[i])
            query_sets, y_val_true = self.generate_multiple_tasks(self.val_seed_series[i])
            with tf.GradientTape(persistent=True) as outer_tape:
                task_loss = 0
                task_val_loss = 0
                for i in range(task_sets.shape[0]):
                    with tf.GradientTape(persistent=True) as inner_tape:
                        if select_model =='MLP':
                            # 調整維度以符合模型的輸入
                            x = task_sets[i].reshape((1, *task_sets[i].shape))
                            y = y_true[i].reshape((1, *y_true[i].shape))
                            # predict validation sets
                            val_X = query_sets[i].reshape((1, *query_sets[i].shape))
                            val_y = y_val_true[i].reshape((1, *y_val_true[i].shape))
                        else:
                            # 調整維度以符合模型的輸入
                            x = task_sets[i].reshape((1, *task_sets[i].shape, 1))
                            y_mean = tf.reduce_mean(y_true, axis = 1)
                            y_mean = tf.reduce_mean(y_mean, axis = 1)
                            y = y_mean[i].numpy().reshape((1, *y_mean[i].shape))
                            # predict validation sets
                            val_X = query_sets[i].reshape((1, *query_sets[i].shape, 1))
                            y_val_mean = tf.reduce_mean(y_val_true, axis = 1)
                            y_val_mean = tf.reduce_mean(y_val_mean, axis = 1)
                            val_y = y_val_mean[i].numpy().reshape((1, *y_val_mean[i].shape))
                        y_pred = model(x)
                        # 給定預測值(y_pred)和實際值(y)，用以計算loss
                        loss = tf.keras.losses.BinaryCrossentropy()(y, y_pred)
                        # val_pred = model(val_X)
                        
                    grads = inner_tape.gradient(loss, model.trainable_variables)
                    # updated_weights = [w - inner_lr*g for w, g in zip(model.trainable_variables, grads)]
                    opt_inner.apply_gradients(zip(grads, model.trainable_variables))
                    # train 
                    y_pred = model(x, training=True)
                    task_loss += tf.keras.losses.BinaryCrossentropy()(y, y_pred)
                    # validation predict with newly updated weights
                    val_pred = model(val_X)
                    task_val_loss += tf.keras.losses.BinaryCrossentropy()(val_y, val_pred)
                task_loss /= task_sets.shape[0]
                task_val_loss /= task_sets.shape[0]
                # Store loss data from both training and validation set
                history['loss'].append(task_loss.numpy())
                history['val_loss'].append(task_val_loss.numpy())
                if not patience:
                    if task_val_loss < best_val_loss:
                        best_val_loss = task_val_loss
                        best_weights = model.get_weights()
                        early_stop_counter = 0
                    else:
                        early_stop_counter += 1
                        if early_stop_counter >= patience:
                            break

            # Create optimizer and compile the model
            outer_grads = outer_tape.gradient(task_loss, model.trainable_variables)
            opt_outer.apply_gradients(zip(outer_grads, model.trainable_variables))
            # updated_weights = [w - outer_lr*g for w, g in zip(model.trainable_variables, outer_grads)]
        return model.trainable_variables, history


def plot_confusion_matrix(y_train, y_predict, save_path=None):
    conf_matrix= confusion_matrix(y_train, y_predict)
    LABELS=['Non_Default','Default'] #給出類別名稱
    plt.figure(figsize=(8,8))
    sns.heatmap(conf_matrix,xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    if not save_path:
        plt.show()
    else:
        plt.savefig(f'{save_path}.jpg')

def series_to_image(train_X:np.ndarray, shape):
    input_cnn = np.zeros((train_X.shape[0], shape**2))
    fill_empty = np.zeros(shape**2- train_X.shape[1])
    for i in range(train_X.shape[0]):
        input_cnn[i] = np.append(train_X[i], fill_empty)
    input_cnn = input_cnn.reshape((input_cnn.shape[0], shape, shape))
    return input_cnn

def predit_classes(weights, X, n_data=None):
    def network_dot(n_layers, current_layer, X_val, weights):
        output = np.dot(X_val, weights[2*current_layer])+weights[2*(current_layer+1)-1]
        if current_layer+1 == n_layers:
            return output
        return network_dot(n_layers, current_layer+1, output, weights)
    
    layers = (len(weights)//2)+1 if len(weights)%2==1 else len(weights)//2
    layer_output = network_dot(layers, current_layer=0, X_val=X, weights=weights)
    pred_class = np.argmax(layer_output.numpy(), axis=-1)
    pred_class = pred_class.reshape((n_data))
    return pred_class

def plot_roc_curve(y_train, y_predict, save_path=None):
    fpr, tpr, threshold = roc_curve(y_train, y_predict)
    print(fpr, tpr, threshold)
    auc1 = auc(fpr, tpr)
    ## Plot the result
    plt.figure(figsize=(10,6))
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, color = 'orange', label = 'AUC = %0.3f' % auc1)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.01,1])
    plt.ylim([0,1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    if not save_path:
        plt.show()
    else:
        plt.savefig(f'{save_path}.jpg')

def generate_validation_data(train_X, train_y, n_data, n_ways, n_shots, random_state):
    np.random.seed(random_state)
    rand_index = np.random.choice(train_X.values.shape[0], n_data)
    X_val = train_X.values[rand_index].reshape(int(n_data/(n_ways*n_shots)), n_ways, n_shots, train_X.values.shape[-1])
    y_val = train_y.values[rand_index]
    # X_val.shape, y_val.shape
    return X_val, y_val