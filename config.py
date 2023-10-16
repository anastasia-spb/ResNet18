CONFIG = dict()

# Training setup
CONFIG['logs_dir'] = './'
CONFIG['data_folder'] = './data'
CONFIG['seed'] = 42
CONFIG['num_epochs'] = 10
CONFIG['batch_size'] = 32 
CONFIG['in_dim'] = (28 * 28) 
CONFIG['out_dim'] = 10
CONFIG['periodic_checkpoint_rate'] = 5 # Shall be smaller or equal than 'num_epochs'
CONFIG['checkpoint_path'] = '/workspaces/MLP_PL_demo/csv_logs/lightning_logs/version_5/checkpoints/epoch=5-step=11250.ckpt'
CONFIG['restore_from_checkpoint'] = False


# Optimization setup for optim.Adam
CONFIG['lr'] = 1.0e-3
CONFIG['weight_decay'] = 1.0e-4