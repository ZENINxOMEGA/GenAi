# ========== Importing Libraries ==========
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
import re
import nltk
import string
import json
from time import time
import pickle
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Dense, Dropout, Embedding, LSTM, Add

# ========== File Paths ==========
IMG_PATH = "C:/OMEGA/Flickr8k/Images/"
CAPTIONS_FILE = "C:/OMEGA/Flickr8k/captions.txt"
GLOVE_PATH = "C:/OMEGA/Flickr8k/glove.6B.50d.txt"
TRAIN_IMAGES_FILE = "C:/OMEGA/Flickr8k/Flickr_8k.trainImages.txt"
TEST_IMAGES_FILE = "C:/OMEGA/Flickr8k/Flickr_8k.testImages.txt"
SAVED_PATH = "C:/OMEGA/Flickr8k/saved/"
MODEL_WEIGHTS_PATH = "C:/OMEGA/Flickr8k/model_weights/"

# ========== Function to Read Text Files ==========
def readTextFile(path):
    with open(path) as f:
        captions = f.read()
    return captions

# ========== Load and Process Captions ==========
captions = readTextFile(CAPTIONS_FILE).split('\n')[:-1]
descriptions = {}
for x in captions:
    first, second = x.split('\t')
    img_name = first.split(".")[0]
    if descriptions.get(img_name) is None:
        descriptions[img_name] = []
    descriptions[img_name].append(second)

# ========== Clean Captions ==========
def clean_text(sentence):
    sentence = sentence.lower()
    sentence = re.sub("[^a-z]+", " ", sentence)
    words = sentence.split()
    sentence = [w for w in words if len(w) > 1]
    return " ".join(sentence)

for key, caption_list in descriptions.items():
    for i in range(len(caption_list)):
        caption_list[i] = clean_text(caption_list[i])

# ========== Save and Reload Descriptions ==========
with open(SAVED_PATH + "descriptions_1.txt", "w") as f:
    f.write(str(descriptions))

with open(SAVED_PATH + "descriptions_1.txt", 'r') as f:
    descriptions = f.read()
descriptions = json.loads(descriptions.replace("'", "\""))

# ========== Build Vocabulary ==========
vocab = set()
for key in descriptions.keys():
    for sentence in descriptions[key]:
        vocab.update(sentence.split())

total_words = []
for key in descriptions.keys():
    [total_words.append(i) for des in descriptions[key] for i in des.split()]

import collections
counter = collections.Counter(total_words)
freq_cnt = dict(counter)
sorted_freq_cnt = sorted(freq_cnt.items(), reverse=True, key=lambda x: x[1])
threshold = 10
sorted_freq_cnt = [x for x in sorted_freq_cnt if x[1] > threshold]
total_words = [x[0] for x in sorted_freq_cnt]

# ========== Process Train/Test Images ==========
train_file_data = readTextFile(TRAIN_IMAGES_FILE)
test_file_data = readTextFile(TEST_IMAGES_FILE)
train = [row.split(".")[0] for row in train_file_data.split("\n")[:-1]]
test = [row.split(".")[0] for row in test_file_data.split("\n")[:-1]]

# Add Start and End Tokens
train_descriptions = {}
for img_id in train:
    train_descriptions[img_id] = []
    for cap in descriptions[img_id]:
        cap_to_append = "startseq " + cap + " endseq"
        train_descriptions[img_id].append(cap_to_append)

# ========== Load Pre-trained ResNet50 ==========
model = ResNet50(weights="imagenet", input_shape=(224, 224, 3))
model_new = Model(model.input, model.layers[-2].output)

# ========== Image Preprocessing and Encoding ==========
def preprocess_img(img):
    img = image.load_img(img, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def encode_image(img):
    img = preprocess_img(img)
    feature_vector = model_new.predict(img, verbose=0)
    return feature_vector.reshape((-1,))

# ========== Encode Training Images ==========
encoding_train = {}
start = time()
for ix, img_id in enumerate(train):
    img_path = IMG_PATH + img_id + ".jpg"
    encoding_train[img_id] = encode_image(img_path)
    if ix % 100 == 0:
        print(f"Encoding Training Image {ix}")
end_t = time()
print("Total Time Taken for Training Encoding:", end_t - start)
with open(SAVED_PATH + "encoded_train_features.pkl", "wb") as f:
    pickle.dump(encoding_train, f)

# ========== Encode Testing Images ==========
encoding_test = {}
start = time()
for ix, img_id in enumerate(test):
    img_path = IMG_PATH + img_id + ".jpg"
    encoding_test[img_id] = encode_image(img_path)
    if ix % 100 == 0:
        print(f"Encoding Testing Image {ix}")
end_t = time()
print("Total Time Taken for Testing Encoding:", end_t - start)
with open(SAVED_PATH + "encoded_test_features.pkl", "wb") as f:
    pickle.dump(encoding_test, f)

# ========== Vocabulary Indexing ==========
word_to_idx = {}
idx_to_word = {}
for i, word in enumerate(total_words):
    word_to_idx[word] = i + 1
    idx_to_word[i + 1] = word

idx_to_word[len(word_to_idx) + 1] = 'startseq'
word_to_idx['startseq'] = len(word_to_idx) + 1
idx_to_word[len(word_to_idx) + 1] = 'endseq'
word_to_idx['endseq'] = len(word_to_idx) + 1

vocab_size = len(word_to_idx) + 1
max_len = max(len(cap.split()) for caps in train_descriptions.values() for cap in caps)

# ========== Data Generator ==========
def data_generator(train_descriptions, encoding_train, word_to_idx, max_len, batch_size):
    X1, X2, y = [], [], []
    n = 0
    while True:
        for key, desc_list in train_descriptions.items():
            n += 1
            photo = encoding_train[key]
            for desc in desc_list:
                seq = [word_to_idx[word] for word in desc.split() if word in word_to_idx]
                for i in range(1, len(seq)):
                    xi = seq[0:i]
                    yi = seq[i]
                    xi = pad_sequences([xi], maxlen=max_len, value=0, padding='post')[0]
                    yi = to_categorical([yi], num_classes=vocab_size)[0]
                    X1.append(photo)
                    X2.append(xi)
                    y.append(yi)
                if n == batch_size:
                    yield ([np.array(X1), np.array(X2)], np.array(y))
                    X1, X2, y = [], [], []
                    n = 0

# ========== Load GloVe Embeddings ==========
embedding_index = {}
with open(GLOVE_PATH, encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        word_embedding = np.array(values[1:], dtype='float')
        embedding_index[word] = word_embedding

def get_embedding_matrix():
    emb_dim = 50
    matrix = np.zeros((vocab_size, emb_dim))
    for word, idx in word_to_idx.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            matrix[idx] = embedding_vector
    return matrix

embedding_matrix = get_embedding_matrix()

# ========== Build Model ==========
input_img_features = Input(shape=(2048,))
inp_img1 = Dropout(0.3)(input_img_features)
inp_img2 = Dense(256, activation='relu')(inp_img1)

input_captions = Input(shape=(max_len,))
inp_cap1 = Embedding(input_dim=vocab_size, output_dim=50, mask_zero=True)(input_captions)
inp_cap2 = Dropout(0.3)(inp_cap1)
inp_cap3 = LSTM(256)(inp_cap2)

decoder1 = Add()([inp_img2, inp_cap3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[input_img_features, input_captions], outputs=outputs)
model.layers[2].set_weights([embedding_matrix])
model.layers[2].trainable = False
model.compile(loss='categorical_crossentropy', optimizer="adam")
model.summary()

# ========== Train Model ==========
epochs = 20
batch_size = 32
steps = len(train_descriptions) // batch_size

def train():
    for i in range(epochs):
        generator = data_generator(train_descriptions, encoding_train, word_to_idx, max_len, batch_size)
        model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)
        if (i + 1) % 5 == 0:
            model.save(MODEL_WEIGHTS_PATH + 'model_' + str(i) + '.keras')

train()

# ========== Load Trained Model ==========
model = load_model(MODEL_WEIGHTS_PATH + 'model_19.keras')

# ========== Prediction Function ==========
def predict_caption(photo):
    in_text = "startseq"
    for i in range(max_len):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence], maxlen=max_len, padding='post')
        ypred = model.predict([photo, sequence], verbose=0)
        ypred = ypred.argmax()
        word = idx_to_word[ypred]
        in_text += ' ' + word
        if word == "endseq":
            break
    return ' '.join(in_text.split()[1:-1])

# ========== Test and Show Captions ==========
for i in range(15):
    idx = np.random.randint(0, len(encoding_test))
    all_img_names = list(encoding_test.keys())
    img_name = all_img_names[idx]
    photo_2048 = encoding_test[img_name].reshape((1, 2048))
    img = plt.imread(IMG_PATH + img_name + ".jpg")
    caption = predict_caption(photo_2048)
    plt.title(caption)
    plt.imshow(img)
    plt.axis("off")
    plt.show()
