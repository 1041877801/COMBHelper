import logging
import yaml
import torch
import torch.optim as optim
from datasets import Cora
from models import GCN1, StudentModel


def train():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='%(message)s',
        level=logging.DEBUG,
    )
    
    logger.info('Loading config...')
    config_file = open('./config.yaml', 'r')
    config = yaml.safe_load(config_file.read())
    
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    logger.info('Loading dataset...')
    dataset = Cora(root=config['dataset_path'])
    data = dataset[0].to(device, 'x', 'edge_index', 'y', 'weight')
    
    logger.info('Loading teacher model...')
    checkpoint = torch.load(config['teacher']['ckpt_path']['MVC'])
    encoder_t = GCN1(
        in_channels=config['teacher']['in_channels'],
        hidden_channels=config['teacher']['hidden_channels'],
        out_channels=config['teacher']['out_channels']
    )
    encoder_t.load_state_dict(checkpoint['model'])
    logger.info('Teacher GNN backbone is GraphSAGE')
    logger.info('In channels: {}'.format(config['teacher']['in_channels']))
    logger.info('Hidden channels: {}'.format(config['teacher']['hidden_channels']))
    logger.info('Out channels: {}'.format(config['teacher']['out_channels']))
    
    logger.info('Loading weights...')
    weights = checkpoint['weights'].detach()
    
    logger.info('Get student model')
    encoder_s = GCN1(
        in_channels=config['student']['in_channels'],
        hidden_channels=config['student']['hidden_channels'],
        out_channels=config['student']['out_channels']
    )
    logger.info('Student GNN backbone is GraphSAGE')
    logger.info('In channels: {}'.format(config['student']['in_channels']))
    logger.info('Hidden channels: {}'.format(config['student']['hidden_channels']))
    logger.info('Out channels: {}'.format(config['student']['out_channels']))
    
    model = StudentModel(
        encoder_t=encoder_t,
        encoder_s=encoder_s,
        T=config['student']['T'],
        alpha=config['student']['alpha'],
        beta=config['student']['beta'],
        boosting=config['student']['boosting'],
        num_class=config['student']['out_channels']
    ).to(device)
    
    logger.info('Reseting model parameters...')
    model.reset_parameters()
    
    logger.info('Get optimizer')
    optimizer = optim.Adam(model.parameters(), lr=config['student']['lr'], weight_decay=config['student']['weight_decay'])
    logger.info('Learning rate: {}'.format(config['student']['lr']))
    logger.info('Weight decay: {}'.format(config['student']['weight_decay']))
    
    logger.info('Start training...')
    acc_best = 0.0
    epochs = config['student']['epochs']
    for epoch in range(epochs):
        
        model.train()
        loss_train, acc_train = model(data, weights.to(device))
        loss_train.backward()
        optimizer.step()

        model.eval()
        acc_val, updated_weights = model.validate(data, weights.to(device))
        logger.info('Epoch: [{}/{}] loss_train: {:.4f} acc_train: {:.4f} acc_val: {:.4f}'.format(epoch + 1, epochs, loss_train, acc_train, acc_val))
        if acc_val > acc_best:
            acc_best = acc_val
            # torch.save({'model': model.encoder_s.state_dict(), 'weights': updated_weights}, config['student']['ckpt_path']['MVC'])
            logger.info('Acc_best is updated to {:.4f}. Model checkpoint is saved to {}'.format(acc_best, config['student']['ckpt_path']['MVC']))
            acc_test = model.test(data)
            logger.info('Test accuracy is {:.4f}'.format(acc_test))
            
    logger.info('Final accuracy is {:.4f}'.format(acc_test))
    

if __name__ == '__main__':
    train()