import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from model import ChatBotModel
from chatbot import _check_restore_parameters, _find_right_bucket, run_step, _construct_response
import config
import data
#from chatbot import chat

#tf.reset_default_graph()
_, enc_vocab = data.load_vocab(os.path.join(config.PROCESSED_PATH, 'vocab.enc'))
inv_dec_vocab, _ = data.load_vocab(os.path.join(config.PROCESSED_PATH, 'vocab.dec'))
model = ChatBotModel(True, batch_size=1)
model.build_graph()

saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
_check_restore_parameters(sess, saver)
# Decode from standard input.
max_length = config.BUCKETS[-1][0]
#print('Welcome to TensorBro. Say something. Enter to exit. Max length is', max_length)
def chat_2(input_text):
        line = input_text
        if len(line) > 0 and line[-1] == '\n':
            line = line[:-1]
        if line == '':
            return "Received Empty Line"
        # Get token-ids for the input sentence.
        token_ids = data.sentence2id(enc_vocab, str(line))
        if (len(token_ids) > max_length):
            return 'Max length I can handle is:'+ str(max_length)
        # Which bucket does it belong to?
        bucket_id = _find_right_bucket(len(token_ids))
        # Get a 1-element batch to feed the sentence to the model.
        encoder_inputs, decoder_inputs, decoder_masks = data.get_batch([(token_ids, [])], 
                                                                        bucket_id,
                                                                        batch_size=1)
        # Get output logits for the sentence.
        _, _, output_logits = run_step(sess, model, encoder_inputs, decoder_inputs,
                                       decoder_masks, bucket_id, True)
        response = _construct_response(output_logits, inv_dec_vocab)
        return response
