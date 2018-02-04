import logging
import config as config

# get TF logger
log = logging.getLogger('tensorflow')
log.setLevel(logging.INFO)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# create file handler which logs even debug messages
fh = logging.FileHandler(config.log_file_path)

fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
log.addHandler(fh)
