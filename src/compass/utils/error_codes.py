"""
==============
error_codes.py
==============
Error codes for use with OPERA PGEs.
"""

from enum import IntEnum, auto, unique

CODES_PER_RANGE = 1000
"""Number of error codes allocated to each range"""

INFO_RANGE_START = 0
"""Starting value for the Info code range"""

DEBUG_RANGE_START = INFO_RANGE_START + CODES_PER_RANGE
"""Starting value for the Debug code range"""

WARNING_RANGE_START = DEBUG_RANGE_START + CODES_PER_RANGE
"""Starting value for the Warning code range"""

CRITICAL_RANGE_START = WARNING_RANGE_START + CODES_PER_RANGE
"""Starting value for the Critical code range"""


@unique
class ErrorCode(IntEnum):
    """
    Error codes for OPERA PGEs.
    Each code is combined with the designated error code offset defined by
    the RunConfig to determine the final, logged error code.
    """

    # Info - 0 to 999
    OVERALL_SUCCESS = INFO_RANGE_START
    LOG_FILE_CREATED = auto()
    LOADING_RUN_CONFIG_FILE = auto()
    VALIDATING_RUN_CONFIG_FILE = auto()
    LOG_FILE_INIT_COMPLETE = auto()
    CREATING_WORKING_DIRECTORY = auto()
    DIRECTORY_SETUP_COMPLETE = auto()
    MOVING_LOG_FILE = auto()
    MOVING_OUTPUT_FILE = auto()
    SUMMARY_STATS_MESSAGE = auto()
    RUN_CONFIG_FILENAME = auto()
    PGE_NAME = auto()
    SCHEMA_FILE = auto()
    INPUT_FILE = auto()
    PROCESSING_INPUT_FILE = auto()
    USING_CONFIG_FILE = auto()
    CREATED_SAS_CONFIG = auto()
    CREATING_OUTPUT_FILE = auto()
    CREATING_CATALOG_METADATA = auto()
    CREATING_ISO_METADATA = auto()
    SAS_PROGRAM_STARTING = auto()
    SAS_PROGRAM_COMPLETED = auto()
    QA_SAS_PROGRAM_STARTING = auto()
    QA_SAS_PROGRAM_COMPLETED = auto()
    QA_SAS_PROGRAM_DISABLED = auto()
    RENDERING_ISO_METADATA = auto()
    CLOSING_LOG_FILE = auto()

    # Debug - 1000 – 1999
    CONFIGURATION_DETAILS = DEBUG_RANGE_START
    PROCESSING_DETAILS = auto()
    SAS_EXE_COMMAND_LINE = auto()
    SAS_QA_COMMAND_LINE = auto()

    # Warning - 2000 – 2999
    DATE_RANGE_MISSING = WARNING_RANGE_START
    NO_RENAME_FUNCTION_FOR_EXTENSION = auto()
    ISO_METADATA_CANT_RENDER_ONE_VARIABLE = auto()
    LOGGING_REQUESTED_SEVERITY_NOT_FOUND = auto()
    LOGGING_SOURCE_FILE_DOES_NOT_EXIST = auto()
    LOGGING_COULD_NOT_INCREMENT_SEVERITY = auto()
    LOGGING_RESYNC_FAILED = auto()

    # Critical - 3000 to 3999
    RUN_CONFIG_VALIDATION_FAILED = CRITICAL_RANGE_START
    DIRECTORY_CREATION_FAILED = auto()
    SAS_CONFIG_CREATION_FAILED = auto()
    CATALOG_METADATA_CREATION_FAILED = auto()
    LOG_FILE_CREATION_FAILED = auto()
    INPUT_NOT_FOUND = auto()
    OUTPUT_NOT_FOUND = auto()
    INVALID_INPUT = auto()
    INVALID_OUTPUT = auto()
    INVALID_CATALOG_METADATA = auto()
    FILE_MOVE_FAILED = auto()
    FILENAME_VIOLATES_NAMING_CONVENTION = auto()
    SAS_PROGRAM_NOT_FOUND = auto()
    SAS_PROGRAM_FAILED = auto()
    QA_SAS_PROGRAM_NOT_FOUND = auto()
    QA_SAS_PROGRAM_FAILED = auto()
    ISO_METADATA_TEMPLATE_NOT_FOUND = auto()
    ISO_METADATA_GOT_SOME_RENDERING_ERRORS = auto()
    ISO_METADATA_RENDER_FAILED = auto()
    SAS_OUTPUT_FILE_HAS_MISSING_DATA = auto()

    @classmethod
    def describe(cls):
        """
        Provides a listing of the available error codes and their associated
        integer values.
        """
        for name, member in cls.__members__.items():
            print(f'{name}: {member.value}')