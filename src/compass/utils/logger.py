#!/usr/bin/env python3

"""
Logging utilities for OPERA ADT. Adapted from OPERA PGE logging utilities
https://github.com/nasa/opera-sds-pge/blob/main/src/opera/util/logger.py
"""
import datetime
import inspect
import shutil
import time
from io import StringIO
from os.path import basename, isfile

from compass.utils import error_codes
from compass.utils.error_codes import ErrorCode


# from .usage_metrics import get_os_metrics


def write(log_stream, severity, workflow, module, error_code, error_location,
          description, time_tag=None):
    """
    Low-level logging write function.
    Parameters
    ----------
    log_stream : io.StringIO
        The log stream to write to.
    severity : str
        The severity level of the log message.
    workflow : str
        Name of the workflow where the logging took place.
    module : str
        Name of the module where the logging took place.
    error_code : int or ErrorCode
        The error code associated with the logged message.
    error_location : str
        File name and line number where the logging took place.
    description : str
        Description of the logged event.
    time_tag : str, optional
        ISO format time tag to associate to the message. If not provided,
        the current time is used.
    """
    if not time_tag:
        time_tag = datetime.datetime.now().isoformat(sep='T',
                                                     timespec='microseconds') + "Z"

    message_str = f'{time_tag}, {severity}, {workflow}, {module}, ' \
                  f'{str(error_code)}, {error_location}, "{description}"\n'

    log_stream.write(message_str)


def default_log_file_name():
    """
    Returns a path + filename that can be used for the log file right away.
    Returns
    -------
    file_path : str
        Path to the default log file name.
    """
    log_datetime_str = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    file_path = f"pge_{log_datetime_str}.log"

    return file_path


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


def standardize_severity_string(severity):
    """
    Returns the severity string in a consistent way.
    Parameters
    ----------
    severity : str
        The severity string to standardize.
    Returns
    -------
    severity : str
        The standardized severity string.
    """
    # First character upper case, rest lower case
    severity = severity.strip().title()

    # Convert some potential log level name variations
    if severity == 'Warn':
        severity = 'Warning'

    if severity == 'Error':
        severity = 'Critical'

    return severity


class Logger:
    """
    Logging class.
    """

    LOGGER_CODE_BASE = 900000
    QA_LOGGER_CODE_BASE = 800000

    def __init__(self, workflow=None, error_code_base=None, log_filename=None):
        """
        Constructor opens the log file as a stream
        Parameters
        ----------
        workflow : str, optional
            Name of the workflow this logger is associated to. Default to CSLC-S1
        error_code_base : int, optional
            The base error code value to associated to the logger. This gives
            the logger a de-facto severity level. Defaults to the Info level
            offset.
        log_filename : str, optional
            Path to write the log's contents to on disk. Defaults to the value
            provided by default_log_file_name().
        """
        self.start_time = time.monotonic()
        self.log_count_by_severity = self._make_blank_log_count_by_severity_dict()
        self.log_filename = log_filename

        if not log_filename:
            self.log_filename = default_log_file_name()

        # open as an empty stream that will be kept in memory
        self.log_stream = StringIO()
        self.log_stream.seek(0)

        self._workflow = (workflow if workflow else "CSLC-S1")

        self._error_code_base = (error_code_base
                                 if error_code_base else Logger.LOGGER_CODE_BASE)

    @property
    def workflow(self):
        """Return workflow attribute"""
        return self._workflow

    @workflow.setter
    def workflow(self, workflow: str):
        '''
        Set workflow attribute
        Parameters
        ----------
        workflow: str
            Name of the workflow
        '''
        self._workflow = workflow

    @property
    def error_code_base(self):
        """Return the error code base from error_codes.py"""
        return self._error_code_base

    @error_code_base.setter
    def error_code_base(self, error_code_base: int):
        '''
        Set error code attribute
        Parameters
        ----------
        error_code_base: int
            Error code from error_codes.py
        '''
        self._error_code_base = error_code_base

    def close_log_stream(self):
        """
        Writes the log summary to the log stream
        Writes the log stream to a log file and saves the file to disk
        Closes the log stream
        """
        if self.log_stream and not self.log_stream.closed:
            self.log_stream.seek(0)

            with open(self.log_filename, 'w', encoding='utf-8') as outfile:
                shutil.copyfileobj(self.log_stream, outfile)

            self.log_stream.close()

    def get_log_count_by_severity(self, severity):
        """
        Gets the number of messages logged for the specified severity
        Parameters
        ----------
        severity : str
            The severity level to get the log count of. Should be one of
            info, debug, warning, critical (case-insensitive).
        Returns
        -------
        log_count : int
            The number of messages logged at the provided severity level.
        """
        try:
            severity = standardize_severity_string(severity)
            count = self.log_count_by_severity[severity]
            return count
        except KeyError:
            self.warning("Logger",
                         ErrorCode.LOGGING_REQUESTED_SEVERITY_NOT_FOUND,
                         f"No messages logged with severity: '{severity}' ")
            return 0

    @staticmethod
    def _make_blank_log_count_by_severity_dict():
        return {
            "Debug": 0,
            "Info": 0,
            "Warning": 0,
            "Critical": 0
        }

    def get_log_count_by_severity_dict(self):
        """Returns a copy of the dictionary of log counts by severity."""
        return self.log_count_by_severity.copy()

    def increment_log_count_by_severity(self, severity):
        """
        Increments the logged message count of the provided severity level.
        Parameters
        ----------
        severity : str
            The severity level to increment the log count of. Should be one of
            info, debug, warning, critical (case-insensitive).
        """
        try:
            severity = standardize_severity_string(severity)
            count = 1 + self.log_count_by_severity[severity]
            self.log_count_by_severity[severity] = count
        except KeyError:
            self.warning("Logger",
                         ErrorCode.LOGGING_COULD_NOT_INCREMENT_SEVERITY,
                         f"Could not increment severity level: '{severity}' ")

    def write(self, severity, module, error_code_offset, description,
              additional_back_frames=0):
        """
        Write a message to the log.
        Parameters
        ----------
        severity : str
            The severity level to log at. Should be one of info, debug, warning,
            critical (case-insensitive).
        module : str
            Name of the module where the logging took place.
        error_code_offset : int
            Error code offset to add to this logger's error code base value
            to determine the final error code associated with the log message.
        description : str
            Description message to write to the log.
        additional_back_frames : int, optional
            Number of call-stack frames to "back up" to in order to determine
            the calling function and line number.
        """
        severity = standardize_severity_string(severity)
        self.increment_log_count_by_severity(severity)

        caller = inspect.currentframe().f_back

        for _ in range(additional_back_frames):
            caller = caller.f_back

        location = caller.f_code.co_filename + ':' + str(caller.f_lineno)

        write(self.log_stream, severity, self.workflow, module,
              self.error_code_base + error_code_offset,
              location, description)

    def info(self, module, error_code_offset, description):
        """
        Write an info-level message to the log.
        Parameters
        ----------
        module : str
            Name of the module where the logging took place.
        error_code_offset : int
            Error code offset to add to this logger's error code base value
            to determine the final error code associated with the log message.
        description : str
            Description message to write to the log.
        """
        self.write("Info", module, error_code_offset, description,
                   additional_back_frames=1)

    def debug(self, module, error_code_offset, description):
        """
        Write a debug-level message to the log.
        Parameters
        ----------
        module : str
            Name of the module where the logging took place.
        error_code_offset : int
            Error code offset to add to this logger's error code base value
            to determine the final error code associated with the log message.
        description : str
            Description message to write to the log.
        """
        self.write("Debug", module, error_code_offset, description,
                   additional_back_frames=1)

    def warning(self, module, error_code_offset, description):
        """
        Write a warning-level message to the log.
        Parameters
        ----------
        module : str
            Name of the module where the logging took place.
        error_code_offset : int
            Error code offset to add to this logger's error code base value
            to determine the final error code associated with the log message.
        description : str
            Description message to write to the log.
        """
        self.write("Warning", module, error_code_offset, description,
                   additional_back_frames=1)

    def critical(self, module, error_code_offset, description):
        """
        Write a critical-level message to the log.
        Since critical messages should be used for unrecoverable errors, any
        time this log level is invoked a RuntimeError is raised with the
        description provided to this function. The log file is closed and
        finalized before the exception is raised.
        Parameters
        ----------
        module : str
            Name of the module where the logging took place.
        error_code_offset : int
            Error code offset to add to this logger's error code base value
            to determine the final error code associated with the log message.
        description : str
            Description message to write to the log.
        Raises
        ------
        RuntimeError
            Raised when this method is called. The contents of the description
            parameter is provided as the exception string.
        """
        self.write("Critical", module, error_code_offset, description,
                   additional_back_frames=1)

        self.close_log_stream()

        raise RuntimeError(description)

    def log(self, module, error_code_offset, description,
            additional_back_frames=0):
        """
        Logs any kind of message.
        Determines the log level (Critical, Warning, Info, or Debug) based on
        the provided error code offset.
        Parameters
        ----------
        module : str
            Name of the module where the logging took place.
        error_code_offset : int
            Error code offset to add to this logger's error code base value
            to determine the final error code associated with the log message.
        description : str
            Description message to write to the log.
        additional_back_frames : int, optional
            Number of call-stack frames to "back up" to in order to determine
            the calling function and line number.
        """
        severity = get_severity_from_error_code(error_code_offset)
        self.write(severity, module, error_code_offset, description,
                   additional_back_frames=additional_back_frames + 1)

    def get_warning_count(self):
        """Returns the number of messages logged at the warning level."""
        return self.get_log_count_by_severity('Warning')

    def get_critical_count(self):
        """Returns the number of messages logged at the critical level."""
        return self.get_log_count_by_severity('Critical')

    def move(self, new_filename):
        """
        This function is useful when the log file has been given a default name,
        and needs to be assigned a name that is inline with user preferences.
        Parameters
        ----------
        new_filename : str
            The new filename (including path) to assign to this log file.
        """
        self.log_filename = new_filename

    def get_stream_object(self):
        """Return the stingIO object for the current log."""
        return self.log_stream

    def get_file_name(self):
        """Return the file name for the current log."""
        return self.log_filename

    def append(self, source):
        """
        Appends text from another file to this log file.
        Parameters
        ----------
        source : str
            The source text to append. If the source refers a file name, the
            contents of the file will be appended. Otherwise, the provided
            text is appended as is.
        """
        if isfile(source):
            with open(source, 'r', encoding='utf-8') as source_file_object:
                source_contents = source_file_object.read().strip()
        else:
            source_contents = source.strip()

        # Parse the contents to append to see if they conform to the expected log
        # formatting for OPERA
        for log_line in source_contents.split('\n'):
            try:
                parsed_line = self.parse_line(log_line)
                write(self.log_stream, *parsed_line)
                severity = parsed_line[0]
                self.increment_log_count_by_severity(severity)
            # If the line does not conform to the expected formatting, just append as-is
            except ValueError:
                self.log_stream.write(log_line + "\n")

    def parse_line(self, line):
        """
        Parses the provided formatted log line into its component parts according
        to the log formatting style for OPERA.
        Parameters
        ----------
        line : str
            The log line to parse
        Returns
        -------
        parsed_line : tuple
            The provided log line parsed into its component parts.
        Raises
        ------
        ValueError
            If the line cannot be parsed according to the OPERA log formatting style.
        """
        try:
            line_components = line.split(',', maxsplit=6)

            if len(line_components) < 7:
                raise ValueError(
                    'Line does not conform to expected formatting style')

            # Remove leading/trailing whitespace from all parsed fields
            line_components = tuple(
                str.strip(line_component) for line_component in line_components)

            (time_tag,
             severity,
             workflow,
             module,
             error_code,
             error_location,
             description) = line_components

            # Convert time-tag to expected iso format
            date_time = datetime.datetime.fromisoformat(time_tag)
            time_tag = date_time.isoformat(sep='T',
                                           timespec='microseconds') + "Z"

            # Standardize the error string
            severity = standardize_severity_string(severity)

            # Remove redundant quotes from log descriptions
            description = description.strip('"').strip("'")

            # Standardize on single quotes within log descriptions
            description = description.replace('"', "'")

            # Map the error code based on message severity
            error_code = {
                "Debug": ErrorCode.LOGGED_DEBUG_LINE,
                "Info": ErrorCode.LOGGED_INFO_LINE,
                "Warning": ErrorCode.LOGGED_WARNING_LINE,
                "Critical": ErrorCode.LOGGED_CRITICAL_LINE
            }[severity]

            # Add the error code base
            error_code += self.error_code_base

            # Return a tuple of the parsed, standardized log line
            return severity, workflow, module, error_code, error_location, description, time_tag
        except Exception as err:
            raise ValueError(
                f'Failed to parse log line "{line}" reason: {str(err)}'
            )
