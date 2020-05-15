
from keras.layers import Bidirectional, Dense, Embedding, Input, Lambda, LSTM, RepeatVector, TimeDistributed, Layer, Activation, Dropout
from keras.preprocessing.sequence import pad_sequences
from keras.layers.advanced_activations import ELU
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras import backend as K
from keras.models import Model
from scipy import spatial
import tensorflow as tf
import pandas as pd
import numpy as np
import tensorflow_addons as tfa

import json
from functools import reduce

## import embedding results
## used for Embedding Layer of LSTM-VAE
file = []
for line in open('embedding json_part-00000-bb252c65-2c2e-4036-8fb7-77257a5b93be-c000.json', 'r'):
    file.append(json.loads(line))

## import patients events sequence 
with open('patient_event.json', 'r') as myfile:
     patient_event = json.load(myfile)

embed_dict = dict()
for i in range(len(file)):
    cur_word = file[i]['word']
    embed_dict[cur_word] = file[i]['vector']['values']

## Generate events corpus
corpus = []
for i,j in patient_event.items():
    corpus.extend(j)

## Count for every event
word_count = dict()
for i in corpus:
    if i in word_count :
        word_count[i] +=1
    else:
        word_count[i] = 1

## Delete rare event happening only once
infre_event = []
for i,j in word_count.items():
    if j == 1:
        infre_event.append(i)

for i in patient_event:
    current = patient_event[i]
    edit = []
    for j in current:
        if j in infre_event:
            continue
        else:
            edit.append(j)
    patient_event[i] = edit

## Generate weights of Reconstruction Loss function for each prediction 
corpus = []
for i,j in patient_event.items():
    corpus.extend(j)

word_count = dict()
for i in corpus:
    if i in word_count :
        word_count[i] +=1
    else:
        word_count[i] = 1

word_weights = {}
for i in word_count:
    current = word_count[i]
    word_weights[str(i).lower()] = 1/np.log(current)

## drop rare vocab in embedding dictionary
embed_drop = {}
for i in embed_dict:
    if i in [str(i).lower() for i in infre_event]:
        continue
    else:
        embed_drop[i] = embed_dict[i]
embed_drop.pop('*nf*')

## Generate index2weights dict
word_index = dict()
j = 1
for i in embed_drop.keys():
    word_index[i] = j
    j+=1

index2weights = {}
for i in word_weights:
    index2weights[word_index[i]] = word_weights[i]

## tranform patients events dict to format which can fit into the LSTM network
sequences = []
for i in patient_event:
    current = patient_event[i]
    sequences.append([word_index.get(str(i).lower()) for i in current])

MAX_SEQUENCE_LENGTH = 300

index2word = {v: k for k, v in word_index.items()}
print('Found %s unique tokens' % len(word_index))
data_1 = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', data_1.shape)
NB_WORDS = len(word_index) + 1  #+1 for zero padding


## set training data
data_train = data_1[0:12600]

## Generate embedding matrix to feed into Keras Embedding layer
EMBEDDING_DIM = 200
embedding_matrix = np.zeros((NB_WORDS, EMBEDDING_DIM))
for word, i in word_index.items():
    if i < NB_WORDS:      
            embedding_vector = embed_drop.get(word)
            embedding_matrix[i] = embedding_vector

batch_size = 100
max_len = MAX_SEQUENCE_LENGTH
emb_dim = EMBEDDING_DIM
latent_dim = 16
intermediate_dim = 300
epsilon_std = 1.0
kl_weight = 0.01
act = ELU()

x = Input(batch_shape=(None,max_len))
x_embed = Embedding(NB_WORDS, emb_dim, weights=[embedding_matrix],
                            input_length=max_len, trainable=False, mask_zero=True)(x)
h = Bidirectional(LSTM(intermediate_dim, return_sequences=False, recurrent_dropout=0.2), merge_mode='concat')(x_embed)
#h = Bidirectional(LSTM(intermediate_dim, return_sequences=False), merge_mode='concat')(h)
#h = Dropout(0.2)(h)
#h = Dense(intermediate_dim, activation='linear')(h)
#h = act(h)
#h = Dropout(0.2)(h)

## hidden layer to learn mean and covariance matrix
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

##Reparametrization trick to get latent vector
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

##Lambda layer to get latent layer(Bottleneck of our network)
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

repeated_context = RepeatVector(max_len)
decoder_h = Bidirectional(LSTM(intermediate_dim, return_sequences=True, recurrent_dropout=0.2),merge_mode = 'concat')
decoder_mean = TimeDistributed(Dense(NB_WORDS, activation='softmax'))
h_decoded = decoder_h(repeated_context(z))
x_decoded_mean = decoder_mean(h_decoded)


# placeholder loss
def zero_loss(y_true, y_pred):
    return K.zeros_like(y_pred)

## Generate weights for each prediction, used in reconstruction weighted cross entropy loss
weights = [0]
for i in word_index:
    try:
      weights.append(word_weights[i])
    except:
      weights.append(0)
      print(word_index[i])
      
## Custom Variational Layer
class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)
        

    def vae_loss(self, x, x_decoded_mean):
        global weights
        
        weights = K.variable(weights)
        x = K.cast(x,dtype = 'int32')
        x = K.one_hot(x,num_classes = NB_WORDS)
       
        y_pred = x_decoded_mean/K.sum(x_decoded_mean, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = x * K.log(y_pred) * weights + (1-x)*K.log(1-y_pred)
        xent_loss = -K.mean(K.sum(loss, -1),-1)
  
       
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        
        xent_loss = K.mean(xent_loss)
        kl_loss = K.mean(kl_loss)
        return K.mean(xent_loss + kl_weight * kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean = inputs[1]
        print(x.shape, x_decoded_mean.shape)
        loss = self.vae_loss(x, x_decoded_mean)
        self.add_loss(loss, inputs=inputs)
        # we don't use this output, but it has to have the correct shape:
        return K.ones_like(x)
    
## Use KL-loss as metrics    
def kl_loss(x, x_decoded_mean):
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    kl_loss = kl_weight * kl_loss
    return kl_loss

loss_layer = CustomVariationalLayer()([x, x_decoded_mean])
vae = Model(x, [loss_layer])
opt = Adam(lr=0.01) 
vae.compile(optimizer='adam', loss=[zero_loss], metrics=[kl_loss])

vae.summary()

vae.fit(data_train, data_train,
     shuffle=True,
     epochs=100,
     batch_size=batch_size)


# build a model to project sentences on the latent space
encoder = Model(x, z_mean)

# build a generator that can sample sentences from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(repeated_context(decoder_input))
_x_decoded_mean = decoder_mean(_h_decoded)
_x_decoded_mean = Activation('softmax')(_x_decoded_mean)
generator = Model(decoder_input, _x_decoded_mean)

##Encode the patients' events into latent space 
sent_encoded = encoder.predict(data_train,batch_size= 100)


