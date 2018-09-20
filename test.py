import EncoderDecoder
from util.functions import trace
from EncoderDecoderModelForward import EncoderDecoderModelForward
from util.chainer_cpu_wrapper import wrapper


parameter_dict = {}
train_path = "util/"
test_path = "util/"
parameter_dict["source"] = train_path + "player_1.txt"
parameter_dict["target"] = train_path + "player_2.txt"
parameter_dict["test_source"] = train_path + "player_1.txt"
parameter_dict["test_target"] = train_path + "player_2.txt"
parameter_dict["reference_target"] = train_path + "player_2.txt"
parameter_dict["word2vecFlag"] = False
parameter_dict["word2vec"] = "word2vec/word2vec_chainer.model"
parameter_dict["encdec"] = EncoderDecoder

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

#--------Hands on  2----------------------------------------------------------------#

trace('initializing ...')
wrapper.init()

encoderDecoderModel = EncoderDecoderModelForward(parameter_dict)
encoderDecoderModel.test()
