'''
Class and function for helping set and determine dataset fill values
'''
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FillValues:
    '''
    Dataclass of fill values for float, complex, and int types with default
    values.
    '''
    # Catch all fill value for all float types (float32, float64, ...)
    float_fill: np.single = np.nan

    # Catch all fill value for all complex types (complex64, complex128, ...)
    complex_fill: np.csingle = np.nan * (0 + 1j)

    # Catch all fill value for all int types (int8, byte8, ...)
    # Currently hard coded for int8/layover_shadow
    int_fill: np.intc = 127

    @classmethod
    def from_user_defined_value(cls, user_defined_value):
        '''
        Create and return a FillValues class object populated with all default
        values populated to a single user defined value.

        Parameters
        ----------
        user_defined_value: float
            User defined value to be assigned to default value of all types

        Returns
        -------
        FillValues
            FillValues object with all default values set to user defined value
        '''
        return cls(np.single(user_defined_value),
                   np.csingle(user_defined_value),
                   np.intc(user_defined_value))


def determine_fill_value(dtype, usr_fill_val=None):
    '''
    Helper function to determine COMPASS specific fill values based on h5py
    Dataset type (dtype)

    Parameters
    ----------
    dtype: type
        Given numeric type whose corresponding fill value of same type is to be
        determined
    usr_fill_val: float
        User specified non-default dataset fill value

    Returns:
        Fill value of type dtype. An exception is raised if no appropriate
        value is found.
    '''
    if usr_fill_val is None:
        # Assign user provided non-default value to all fill values in
        # FillValues object with correct typing. Logic below will return on
        # accordingly.
        fill_values = FillValues.from_user_defined_value(usr_fill_val)
    else:
        fill_values = FillValues()

    # Check if float type and return float fill
    float_types = [np.double, np.single, np.float32, np.float64, 'float32',
                   'float64']
    if dtype in float_types:
        return fill_values.float_fill

    # Check if complex type and return complex fill
    complex_types = [np.complex128, 'complex64', np.complex64, 'complex32']
    if dtype in complex_types:
        return fill_values.complex_fill

    # Check if int type and return int fill
    int_types = [np.byte, np.int8]
    if dtype in int_types:
        return fill_values.int_fill

    # No appropriate fill value found above. Raise exception.
    raise ValueError(f'Unexpected COMPASS geocoded dataset type: {dtype}')
