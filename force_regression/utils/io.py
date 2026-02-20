import os


def create_folder(data_path):
    """
    Create a folder if it doesn't exist
    """
    if not os.path.exists(data_path):
        os.makedirs(data_path)


# from snn_prepare.py -> TODO: use a common function
def create_reg_results_and_figs_dirs(data_config):
    """
    Create and update the results and figures directories for the regression models
    """
    # results dir is the parent folder, results_path is the full path adding results_dir + subject
    if len(data_config.load_multi) > 1:
        data_config.results_path = os.path.join(data_config.results_path, 'multi_mvc')
        create_folder(data_config.results_path)