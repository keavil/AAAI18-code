import argparse
import time
class Parser(object):
    def getParser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--name', type=str, default='test')
        parser.add_argument('--fasttest', type=int, default=0, choices=[0, 1])
        parser.add_argument('--seed', type=int, default=int(1000*time.time()))
        parser.add_argument('--dataset', type=str, default='../TrainData/MR')
        parser.add_argument('--maxlenth', type=int, default=70)
        parser.add_argument('--grained', type=int, default=2)
        parser.add_argument('--optimizer', type=str, default='Adam', \
                choices=['SGD', 'Adagrad', 'Adadelta', 'Adam', 'Nadam'])
        parser.add_argument('--lr', type=float, default=0.0005)
        parser.add_argument('--epoch', type=int, default=5)
        parser.add_argument('--batchsize', type=int, default=5)
        parser.add_argument('--samplecnt', type=int, default=5)
        parser.add_argument('--attention', type=int, default=0, choices=[0, 1])
        parser.add_argument('--word_vector', type=str, default='../WordVector/vector.300dim')
        parser.add_argument('--dim', type=int, default=300)
        parser.add_argument('--tau', type=float, default=0.1)
        parser.add_argument('--dropout', type=float, default=0.5)
        parser.add_argument('--alpha', type=float, default=0.1)
        parser.add_argument('--LSTMpretrain', type=str, default='')
        parser.add_argument('--RLpretrain', type=str, default='')
        parser.add_argument('--pretype', type=str, default='N1', choices=['N1', 'ALL1', 'ALL0', 'RANDOM'])
        return parser
