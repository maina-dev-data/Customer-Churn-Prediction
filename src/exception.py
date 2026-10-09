import sys
from src.logger import logging

def get_detailed_error_message(error: Exception, error_detail: sys) -> str:
    """
    Extracts filename, line number, and error message from sys.exc_info()
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = f"Error occurred in script: [{file_name}] at line number: [{line_number}] - error message: [{str(error)}]"
    return error_message

class CustomException(Exception):
    def __init__(self, error_message: Exception, error_detail: sys):
        super().__init__(str(error_message))
        self.error_message = get_detailed_error_message(error_message, error_detail=error_detail)
        logging.error(self.error_message)

    def __str__(self) -> str:
        return self.error_message
