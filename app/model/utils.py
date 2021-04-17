import tensorflow_hub as hub
import tensorflow as tf
from official.nlp.bert.tokenization import FullTokenizer

my_bert_layer = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
    trainable=False
)

vocab_file = my_bert_layer.resolved_object.vocab_file.asset_path.numpy()

do_lower_case = my_bert_layer.resolved_object.do_lower_case.numpy()

tokenizer = FullTokenizer(vocab_file, do_lower_case)



def is_whitespace(c):
    '''
    Indica si un cadena de caracteres se corresponde con un espacio en blanco / separador o no.
    '''
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def whitespace_split(text):
    '''
    Toma el texto y devuelve una lista de "palabras" separadas segun los 
    espacios en blanco / separadores anteriores.
    '''
    doc_tokens = []
    prev_is_whitespace = True
    for c in text:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
    return doc_tokens


def tokenize_context(text_words):
    '''
    Toma una lista de palabras (devueltas por whitespace_split()) y tokeniza cada
    palabra una por una. También almacena, para cada nuevo token, la palabra original
    del parámetro text_words.
    '''
    text_tok = []
    tok_to_word_id = []
    for word_id, word in enumerate(text_words):
        word_tok = tokenizer.tokenize(word)
        text_tok += word_tok
        tok_to_word_id += [word_id]*len(word_tok)
    return text_tok, tok_to_word_id


def get_ids(tokens):
    return tokenizer.convert_tokens_to_ids(tokens)


def get_mask(tokens):
    return np.char.not_equal(tokens, "[PAD]").astype(int)


def get_segments(tokens):
    seg_ids = []
    current_seg_id = 0
    for tok in tokens:
        seg_ids.append(current_seg_id)
        if tok == "[SEP]":
            current_seg_id = 1-current_seg_id # Convierte 1 en 0 y viceversa
    return seg_ids


def create_input_dict(question, context):
    '''
    Take a question and a context as strings and return a dictionary with the 3
    elements needed for the model. Also return the context_words, the
    context_tok to context_word ids correspondance and the length of
    question_tok that we will need later.
    '''
    question_tok = tokenizer.tokenize(my_question)

    context_words = whitespace_split(context)
    context_tok, context_tok_to_word_id = tokenize_context(context_words)

    input_tok = question_tok + ["[SEP]"] + context_tok + ["[SEP]"]
    input_tok += ["[PAD]"]*(384-len(input_tok)) # in our case the model has been
                                                # trained to have inputs of length max 384
    input_dict = {}
    input_dict["input_word_ids"] = tf.expand_dims(tf.cast(get_ids(input_tok), tf.int32), 0)
    input_dict["input_mask"] = tf.expand_dims(tf.cast(get_mask(input_tok), tf.int32), 0)
    input_dict["input_type_ids"] = tf.expand_dims(tf.cast(get_segments(input_tok), tf.int32), 0)

    return input_dict, context_words, context_tok_to_word_id, len(question_tok)