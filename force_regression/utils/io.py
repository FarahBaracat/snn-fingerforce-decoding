import os
import pandas as pd



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



def save_to_source_data(source_data_file_path, df, sheet_name):
    try:
        mode = 'a' if os.path.exists(source_data_file_path) else 'w'
        kwargs = {'if_sheet_exists': 'replace'} if mode == 'a' else {}
        with pd.ExcelWriter(source_data_file_path, mode=mode, **kwargs) as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            # Auto-fit column widths
            worksheet = writer.sheets[sheet_name]
            for col in worksheet.columns:
                max_len = max(len(str(cell.value)) if cell.value else 0 for cell in col)
                worksheet.column_dimensions[col[0].column_letter].width = max_len + 2
        print(f"Data successfully saved to {source_data_file_path} in sheet '{sheet_name}'.")
    except Exception as e:
        print(f"An error occurred while saving to Excel: {e}")
