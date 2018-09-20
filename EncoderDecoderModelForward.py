# 表示用に使用しています。
import EncoderDecoder
from util.functions import trace
import numpy as np

from chainer import functions, optimizers

# cpu計算とgpu計算で使い分けるラッパー
from util.chainer_cpu_wrapper import wrapper

from EncoderDecoderModel import EncoderDecoderModel


class EncoderDecoderModelParameter():

    def __init__(self, is_training, src_batch, encoderDecoderModel, trg_batch=None, generation_limit=None):
        self.model = encoderDecoderModel.model
        self.tanh = functions.tanh
        self.lstm = functions.lstm
        self.batch_size = len(src_batch)
        self.src_len = len(src_batch[0])
        # 翻訳元言語を単語からインデックスにしている（ニューラルネットの空間で扱うため）
        self.src_stoi = encoderDecoderModel.src_vocab.stoi
        # 翻訳先言語を単語からインデックスにしている（ニューラルネットの空間で扱うため）
        self.trg_stoi = encoderDecoderModel.trg_vocab.stoi
        # 翻訳先言語をインデックスから単語にしている(翻訳結果として保持するため、翻訳先言語だけ用意している)
        self.trg_itos = encoderDecoderModel.trg_vocab.itos
        # lstmのために状態を初期化
        self.state_c = wrapper.zeros((self.batch_size, encoderDecoderModel.n_hidden))
        self.trg_batch = trg_batch
        self.generation_limit = generation_limit


class EncoderDecoderModelEncoding():

    def encoding(self, src_batch, parameter, trg_batch=None, generation_limit=None):
        # --------Hands on------------------------------------------------------------------#
        # encoding
        # 翻訳元言語の末尾</s>を潜在空間に射像し、隠れ層に入力、lstmで出力までをバッチサイズ分行う
        # 予め末尾の設定をしていないと終了単語が分からないため
        # 1:翻訳元言語の入力x:図のx部分に相当
        state_x = wrapper.make_var([parameter.src_stoi('</s>') for _ in range(parameter.batch_size)], dtype=np.int32)
        # 2:翻訳元言語の入力xを潜在空間に射像する。（次元数を圧縮するため）:図のi部分に相当
        state_i = parameter.tanh(parameter.model.weight_xi(state_x))
        # 3:潜在空間iの入力をlstmに入力し、次の単語予測に使用する:図のp部分に相当
        parameter.state_c, state_p = parameter.lstm(parameter.state_c, parameter.model.weight_ip(state_i))

        # 翻訳元言語を逆順に上記と同様の処理を行う
        for l in reversed(range(parameter.src_len)):
            # 翻訳元言語を語彙空間に写像
            state_x = wrapper.make_var([parameter.src_stoi(src_batch[k][l]) for k in range(parameter.batch_size)],
                                       dtype=np.int32)
            # 語彙空間を潜在空間（次元数が減る）に射像
            state_i = parameter.tanh(parameter.model.weight_xi(state_x))
            # 状態と出力結果をlstmにより出力。lstmの入力には前の状態と語彙空間の重み付き出力と前回の重み付き出力を入力としている
            parameter.state_c, state_p = parameter.lstm(parameter.state_c, parameter.model.weight_ip(state_i)
                                                        + parameter.model.weight_pp(state_p))

        # 次のミニバッチ処理のために最終結果をlstmで出力。翻訳の仮説用のリストを保持
        parameter.state_c, state_q = parameter.lstm(parameter.state_c, parameter.model.weight_pq(state_p))
        hyp_batch = [[] for _ in range(parameter.batch_size)]
        return state_q, hyp_batch


# --------Hands on------------------------------------------------------------------#

class EncoderDecoderModelDecoding():

    def decoding(self, is_training, src_batch, parameter, state_q, hyp_batch, trg_batch=None, generation_limit=None):

        # --------Hands on------------------------------------------------------------------#
        # decoding
        """
　　     学習
        """
        if is_training:
            # 損失の初期化及び答えとなる翻訳先言語の長さを取得。（翻訳元言語と翻訳先言語で長さが異なるため）
            # 損失が最小となるように学習するため必要
            accum_loss = wrapper.zeros(())
            trg_len = len(parameter.trg_batch[0])

            # ニューラルネットの処理は基本的にEncodingと同一であるが、損失計算と翻訳仮説候補の確保の処理が加わっている
            for l in range(trg_len):
                # 1:翻訳元言語に対するニューラルの出力qを受け取り、潜在空間jに射像
                state_j = parameter.tanh(parameter.model.weight_qj(state_q))
                # 2:潜在空間jから翻訳先言語yの空間に射像
                result_y = parameter.model.weight_jy(state_j)
                # 3:答えとなる翻訳結果を取得
                state_target = wrapper.make_var([parameter.trg_stoi(parameter.trg_batch[k][l])
                                                 for k in range(parameter.batch_size)], dtype=np.int32)
                # 答えと翻訳結果により損失を計算
                accum_loss += functions.softmax_cross_entropy(result_y, state_target)
                # 複数翻訳候補が出力されるため、出力にはもっとも大きな値を選択
                output = wrapper.get_data(result_y).argmax(1)

                # 翻訳仮説確保(インデックスから翻訳単語に直す処理も行っている）
                for k in range(parameter.batch_size):
                    hyp_batch[k].append(parameter.trg_itos(output[k]))

                # 状態と出力結果をlstmにより出力。lstmの入力には前の状態と語彙空間の重み付き出力と前回の重み付き出力を入力としている
                parameter.state_c, state_q = parameter.lstm(parameter.state_c, parameter.model.weight_yq(state_target)
                                                             + parameter.model.weight_qq(state_q))
            return hyp_batch, accum_loss
        else:
            """
            予測部分
            """
            # 末尾に</s>が予測できないと無限に翻訳してしまうため、予測では予測する翻訳言語の長さに制約をしている
            while len(hyp_batch[0]) < parameter.generation_limit:
                state_j = parameter.tanh(parameter.model.weight_qj(state_q))
                result_y = parameter.model.weight_jy(state_j)
                # 複数翻訳候補が出力されるため、出力にはもっとも大きな値を選択
                output = wrapper.get_data(result_y).argmax(1)

                # 翻訳仮説確保(インデックスから翻訳単語に直す処理も行っている）
                for k in range(parameter.batch_size):
                    hyp_batch[k].append(parameter.trg_itos(output[k]))

                # ミニバッチサイズ分の翻訳仮説の末尾が</s>になったときにDecoding処理が終わるようになっている。
                if all(hyp_batch[k][-1] == '</s>' for k in range(parameter.batch_size)): break

                # 翻訳仮説をニューラルネットで扱える空間に射像している
                state_y = wrapper.make_var(output, dtype=np.int32)
                # 次のlstmの処理のために出力結果と状態を渡している
                parameter.state_c, state_q = parameter.lstm(parameter.state_c, parameter.model.weight_yq(state_y)
                                                             + parameter.model.weight_qq(state_q))

            return hyp_batch

# --------Hands on------------------------------------------------------------------#

class EncoderDecoderModelForward(EncoderDecoderModel):

    def forward(self, is_training, src_batch, trg_batch=None, generation_limit=None):
        # パラメータ設定
        parameter = EncoderDecoderModelParameter(is_training, src_batch, self, trg_batch, generation_limit)

        # encoding
        encoder = EncoderDecoderModelEncoding()
        s_q, hyp_batch = encoder.encoding(src_batch, parameter)
        # decoding
        decoder = EncoderDecoderModelDecoding()
        if is_training:
            return decoder.decoding(is_training, src_batch, parameter, s_q, hyp_batch, trg_batch, generation_limit)
        else:
            return decoder.decoding(is_training, src_batch, parameter, s_q, hyp_batch, trg_batch, generation_limit)


parameter_dict = {}
train_path = "util/"
test_path = "util/"
parameter_dict["source"] = train_path + "player_1.txt"
parameter_dict["target"] = train_path + "player_2.txt"
parameter_dict["test_source"] = test_path + "test1000.ja"
parameter_dict["test_target"] = test_path + "test1000_hyp.en"
parameter_dict["reference_target"] = test_path + "test1000.en"
parameter_dict["word2vec"] = "word2vec/word2vec_chainer.model"
parameter_dict["word2vecFlag"] = False
parameter_dict["encdec"] = EncoderDecoder
#--------Hands on  2----------------------------------------------------------------

"""
下記の値が大きいほど扱える語彙の数が増えて表現力が上がるが計算量が爆発的に増えるので大きくしない方が良いです。
"""
parameter_dict["vocab"] = 550

"""
この数が多くなればなるほどモデルが複雑になります。この数を多くすると必然的に学習回数を多くしないと学習は
収束しません。
語彙数よりユニット数の数が多いと潜在空間への写像が出来ていないことになり結果的に意味がない処理になります。
"""
parameter_dict["embed"] = 500

"""
この数も多くなればなるほどモデルが複雑になります。この数を多くすると必然的に学習回数を多くしないと学習は
収束しません。
"""
parameter_dict["hidden"] = 20

"""
学習回数。基本的に大きい方が良いが大きすぎると収束しないです。
"""
parameter_dict["epoch"] = 20

"""
ミニバッチ学習で扱うサイズです。この点は経験的に調整する場合が多いが、基本的に大きくすると学習精度が向上する
代わりに学習スピードが落ち、小さくすると学習精度が低下する代わりに学習スピードが早くなります。
"""
parameter_dict["minibatch"] = 64

"""
予測の際に必要な単語数の設定。長いほど多くの単語の翻訳が確認できるが、一般的にニューラル翻訳は長い翻訳には
向いていないので小さい数値がオススメです。
"""
parameter_dict["generation_limit"] = 256


trace('initializing ...')
wrapper.init()

encoderDecoderModel = EncoderDecoderModelForward(parameter_dict)
encoderDecoderModel.train()
