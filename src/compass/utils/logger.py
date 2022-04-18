import datetime
import inspect
import time
import journal

from compass.utils import error_codes
from .error_codes import ErrorCode


def get_severity_from_error_code(error_code):
    """
    Determines the log level (Critical, Warning, Info, or Debug) based on the
    provided error code.
    Parameters
    ----------
    error_code : int or ErrorCode
        The error code to map to a severity level.
    Returns
    -------
    severity : str
        The severity level associated to the provided error code.
    """
    # TODO: constants for max codes and severity strings
    error_code_offset = error_code % 10000

    if error_code_offset < error_codes.DEBUG_RANGE_START:
        return "Info"

    if error_code_offset < error_codes.WARNING_RANGE_START:
        return "Debug"

    if error_code_offset < error_codes.CRITICAL_RANGE_START:
        return "Warning"

    return "Critical"


class Logger:
    '''
    Class to handle log of info and error messages
    '''

    LOGGER_CODE_BASE = 900000
    QA_LOGGER_CODE_BASE = 800000

    def __init__(self, workflow=None,
                 error_code_base=None, channel=None):
        '''
        Constructor for Logger class

        Parameters
        ----------
        workflow: str
            Name of the workflow for which to use logging
        error_code_base: int
            Error code base
        channel: journal.info, journal.error
            journal channel object
        '''
        self.start_time = time.monotonic()
        self.log_count_by_severity = self._make_blank_log_count_by_severity_dict()
        self.channel = channel

        if not channel:
            self.channel = journal.info('logger.Logger')
        self.workflow = workflow
        self.error_code_base = (error_code_base
                                if error_code_base else Logger.LOGGER_CODE_BASE)

    def get_workflow(self):
        '''
        Get workflow name
        '''
        return self.workflow

    def set_workflow(self, workflow):
        '''
        Set workflow name to use for logging

        Parameters
        ----------
        workflow: str
            Workflow name to use for logging
        '''
        self.workflow = workflow

    def get_error_code_base(self):
        '''
        Get error code base
        '''
        return self.error_code_base

    def set_error_code_base(self, error_code_base: int):
        '''
        Set error code base

        Parameters
        ----------
        error_code_base: int
            Error code base
        '''
        self.error_code_base = error_code_base

    @staticmethod
    def _make_blank_log_count_by_severity_dict():
        '''
        Return blank log count by severity dictionary

        Returns
        -------
        dict: dict
            Return log count by severity dictionary
        '''
        return {
            "Debug": 0,
            "Info": 0,
            "Warning": 0,
            "Critical": 0
        }

    def get_log_count_by_severity(self):
        '''
        Get log count by severity
        '''
        return self.log_count_by_severity.copy()

    def increment_log_count_by_severity(self, severity):
        '''
        Increment log count by severity

        Parameters
        ----------
        severity: str
            Severity of the error to increment
        '''
        try:
            severity = severity.title()
            count = 1 + self.log_count_by_severity[severity]
            self.log_count_by_severity[severity] = count
        except:
            self.warning("Logger",
                         ErrorCode.LOGGING_COULD_NOT_INCREMENT_SEVERITY,
                         f"Could not increment severity level: '{severity}")

    def write(self, severity, module, error_code_offset, description,
              additional_back_frames=0):
        '''
        Write log message to channel

        Parameters
        ----------
        severity: str
            Severity of the log message to record
        module: str
            Module where the error message is occurring
        error_code_offset: int
            Error code offset
        description: str
            String describing the error message
        additional_back_frames: int
            Number of additional frames for inspected object
        '''
        severity = severity.title()
        self.increment_log_count_by_severity(severity)

        caller = inspect.currentframe().f_back

        for _ in range(additional_back_frames):
            caller = caller.f_back

        location = caller.f_code.co_filename + ':' + str(caller.f_lineno)
        workflow = self.workflow
        error_code_base = self.error_code_base

        now_tag = datetime.datetime.now()
        time_tag = now_tag.strftime("%Y-%m-%dT%H:%M:%S.%f")
        message = f'{time_tag}, {severity}, {workflow}, {module}, ' \
                  f'{str(error_code_base + error_code_offset)}, ' \
                  f'{location}, {description}'

        self.channel.log(message)

    def info(self, module, error_code_offset,
             description):
        '''
        Log an info message

        Parameters
        ----------
        module: str
            Module where the error message occur
        error_code_offset: int
            Error code offset
        description: str
            Description of the message to log
        '''
        self.write("Info", module, error_code_offset, description,
                   additional_back_frames=1)

    def debug(self, module, error_code_offset, description):
        '''
        Log an debug message

        Parameters
        ----------
        module: str
            Module where the error message occur
        error_code_offset: int
            Error code offset
        description: str
            Description of the message to log
        '''
        self.write("Debug", module, error_code_offset, description,
                   additional_back_frames=1)

    def warning(self, module, error_code_offset, description):
        '''
        Log an warning message

        Parameters
        ----------
        module: str
            Module where the error message occur
        error_code_offset: int
            Error code offset
        description: str
            Description of the message to log
        '''
        self.write("Warning", module, error_code_offset, description,
                   additional_back_frames=1)

    def critical(self, module, error_code_offset, description):
        '''
        Log an critical message and throw a RuntimeError

        Parameters
        ----------
        module: str
            Module where the error message occur
        error_code_offset: int
            Error code offset
        description: str
            Description of the message to log
        '''
        self.write("Critical", module, error_code_offset, description,
                   additional_back_frames=1)
        raise RuntimeError(description)

    def log(self, module, error_code_offset, description,
            additional_back_frames=0):
        '''
        Log any message

        Parameters
        ----------
        module: str
            Module where the error message occur
        error_code_offset: int
            Error code offset
        description: str
            Description of the message to log
        '''
        severity = get_severity_from_error_code(error_code_offset)
        self.write(severity, module, error_code_offset, description,
                   additional_back_frames=additional_back_frames + 1)
