import logging
import yaml
import torch
from datasets import *
from models import GCN1, GCN2
import pickle
from sklearn.metrics import recall_score


if __name__ == '__main__':
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='%(message)s',
        level=logging.DEBUG,
    )
    
    config_file = open('./config.yaml', 'r')
    config = yaml.safe_load(config_file.read())
    
    device = torch.device('cpu')
    
    logger.info('Loading dataset...')
    # dataset = Alpha(root='./data/Alpha')
    dataset = OTC(root='./data/OTC')
    data = dataset[0].to(device, 'x', 'edge_index', 'y')
    
    # teacher
    # checkpoint = torch.load(config['teacher']['ckpt_path']['MIS'])
    checkpoint = torch.load(config['teacher']['best_ckpt_path']['MIS']) # best
    encoder = GCN1(
        in_channels=config['teacher']['in_channels'],
        hidden_channels=config['teacher']['hidden_channels'],
        out_channels=config['teacher']['out_channels']
    ).to(device)
    encoder.load_state_dict(checkpoint['model'], strict=True)

    encoder.eval()
    out = encoder(data)
    y = data.y
    preds = out.argmax(dim=-1)
    recall = recall_score(y_true=y, y_pred=preds)
    logger.info('recall: {:.4f}'.format(recall))
    acc_test = int((preds == y).sum()) / len(y)
    logger.info('acc_test: {:.4f}'.format(acc_test))
    
    # f = open('./preds/Cora_MIS_Teacher.preds', 'wb')
    # pickle.dump({'pred': preds.tolist()}, f)
    
    # student1
    # checkpoint = torch.load(config['student']['ckpt_path']['MIS'])
    checkpoint = torch.load(config['student']['best_ckpt_path']['MIS']) # best
    encoder = GCN2(
        in_channels=config['student']['in_channels'],
        hidden_channels=config['student']['hidden_channels'],
        out_channels=config['student']['out_channels']
    ).to(device)
    encoder.load_state_dict(checkpoint['model'], strict=True)

    encoder.eval()
    out = encoder(data)
    y = data.y
    preds = out.argmax(dim=-1)
    recall = recall_score(y_true=y, y_pred=preds.tolist())
    logger.info('recall: {:.4f}'.format(recall))
    acc_test = int((preds == y).sum()) / len(y)
    logger.info('acc_test: {:.4f}'.format(acc_test))
    
    # f = open('./preds/Cora_MIS_Student.preds', 'wb')
    # pickle.dump({'pred': preds.tolist()}, f)