import logging
import yaml
import torch
import torch.optim as optim
from datasets import *
from models import GCN1, TeacherModel


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
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    logger.info('Loading dataset...')
    dataset = BA_Train(root=config['dataset_path'])
    data = dataset[0].to(device, 'x', 'edge_index', 'y', 'weight')
    
    logger.info('Initializing weights...')
    weights = torch.ones(data.x.size(0))
    weights = weights[data.train_mask]
    weights = weights / weights.sum()
    
    logger.info('Get teacher model')
    model = TeacherModel(
        GCN1(
            in_channels=config['teacher']['in_channels'],
            hidden_channels=config['teacher']['hidden_channels'],
            out_channels=config['teacher']['out_channels']
        )
    ).to(device)
    logger.info('In channels: {}'.format(config['teacher']['in_channels']))
    logger.info('Hidden channels: {}'.format(config['teacher']['hidden_channels']))
    logger.info('Out channels: {}'.format(config['teacher']['out_channels']))
    
    logger.info('Reseting model parameters...')
    model.reset_parameters()
    
    logger.info('Get optimizer')
    optimizer = optim.Adam(model.parameters(), lr=config['teacher']['lr'], weight_decay=config['teacher']['weight_decay'])
    logger.info('Learning rate: {}'.format(config['teacher']['lr']))
    logger.info('Weight decay: {}'.format(config['teacher']['weight_decay']))
    
    logger.info('Start training...')
    acc_best = 0.0
    epochs = config['teacher']['epochs']
    for epoch in range(epochs):
        
        model.train()
        loss_train, acc_train = model(data)
        loss_train.backward()
        optimizer.step()
        
        model.eval()
        acc_val, updated_weights = model.validate(data, weights.to(device))
        logger.info('Epoch: [{}/{}] loss_train: {:.4f} acc_train: {:.4f} acc_val: {:.4f}'.format(epoch + 1, epochs, loss_train, acc_train, acc_val))
        if acc_val > acc_best:
            acc_best = acc_val
            # torch.save({'model': model.encoder.state_dict(), 'weights': updated_weights}, config['teacher']['ckpt_path']['MVC'])
            logger.info('Acc_best is updated to {:.4f}. Model checkpoint is saved to {}.'.format(acc_best, config['teacher']['ckpt_path']['MVC']))
            acc_test = model.test(data)
            logger.info('Test accuracy is {:.4f}'.format(acc_test))
        
    logger.info('Final accuracy is {:.4f}'.format(acc_test))
    

if __name__ == '__main__':
    train()