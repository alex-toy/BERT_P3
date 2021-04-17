import tensorflow as tf

class BERTSquad(tf.keras.Model):
    
    def __init__(self,
                 name="bert_squad"):
        super(BERTSquad, self).__init__(name=name)
        
        self.bert_layer = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
            trainable=True)
        
        self.squad_layer = BertSquadLayer()
    
    def apply_bert(self, inputs):
        _ , sequence_output = self.bert_layer([inputs["input_word_ids"],
                                               inputs["input_mask"],
                                               inputs["input_type_ids"]])
        return sequence_output

    def call(self, inputs):
        seq_output = self.apply_bert(inputs)

        start_logits, end_logits = self.squad_layer(seq_output)
        
        return start_logits, end_logits