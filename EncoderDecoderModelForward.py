import numpy as np
from chainer import Chain, Variable, cuda, functions, links, optimizer, optimizers, serializers
from EncoderDecoderModel import EncoderDecoderModel
from word2vec.word2vec_load import SkipGram,SoftmaxCrossEntropyLoss

unit = 300
vocab = 9982
loss_func = SoftmaxCrossEntropyLoss(unit, vocab)
w2v_model = SkipGram(vocab, unit, loss_func)
serializers.load_hdf5("word2vec/word2vec_chainer.model", w2v_model)


class EncoderDecoderModelForward(EncoderDecoderModel):

    def forward(self, src_batch, trg_batch, src_vocab, trg_vocab, encdec, is_training, generation_limit):
        batch_size = len(src_batch)
        src_len = len(src_batch[0])
        trg_len = len(trg_batch[0]) if trg_batch else 0
        src_stoi = src_vocab.stoi
        trg_stoi = trg_vocab.stoi
        trg_itos = trg_vocab.itos
        encdec.reset(batch_size)

        x = self.common_function.my_array([src_stoi('</s>') for _ in range(batch_size)], np.int32)
        encdec.encode(x)
        for l in reversed(range(src_len)):
            x = self.common_function.my_array([src_stoi(src_batch[k][l]) for k in range(batch_size)], np.int32)
            encdec.encode(x)

        t = self.common_function.my_array([trg_stoi('<s>') for _ in range(batch_size)], np.int32)
        hyp_batch = [[] for _ in range(batch_size)]

        if is_training:
            loss = self.common_function.my_zeros((), np.float32)
            for l in range(trg_len):
                y = encdec.decode(t)
                t = self.common_function.my_array([trg_stoi(trg_batch[k][l]) for k in range(batch_size)], np.int32)
                loss += functions.softmax_cross_entropy(y, t)
                output = cuda.to_cpu(y.data.argmax(1))
                for k in range(batch_size):
                    hyp_batch[k].append(trg_itos(output[k]))
            return hyp_batch, loss

        else:
            while len(hyp_batch[0]) < generation_limit:
                y = encdec.decode(t)
                output = cuda.to_cpu(y.data.argmax(1))
                t = self.common_function.my_array(output, np.int32)
                for k in range(batch_size):
                    hyp_batch[k].append(trg_itos(output[k]))
                if all(hyp_batch[k][-1] == '</s>' for k in range(batch_size)):
                    break

        return hyp_batch
