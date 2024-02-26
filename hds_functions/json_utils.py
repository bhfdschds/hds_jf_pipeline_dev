"""
Module name: json_utils.py

Description:
    This module provides utilities for reading and writing JSON files in Python.
"""
import json
from typing import Any, Dict

def read_json_file(path: str) -> Dict[str, Any]:
    """
    Read a JSON file from the specified path and return its content as a Python dictionary. 

    This function reads a JSON file in binary mode from the specified path and returns its content as a Python dictionary. 
    It checks for duplicate keys within the JSON object and raises a ValueError if duplicates are found.

    Args:
        path (str): The path to the JSON file.

    Returns:
        dict: A Python dictionary containing the JSON content.

    Raises:
        ValueError: If the JSON file contains duplicate keys.

    Examples:
        >>> data = read_json_file("example.json")
        >>> print(data)
        {'key1': 'value1', 'key2': 'value2'}
    """

    def check_json_for_duplicate_keys(ordered_pairs):
        """
        Reject duplicate keys while loading JSON content.

        Args:
            ordered_pairs: A list of key-value pairs from the JSON object.

        Returns:
            dict: A dictionary representation of the JSON object.

        Raises:
            ValueError: If duplicate keys are found.
        """
        d = {}
        for k, v in ordered_pairs:
            if k in d:
                raise ValueError(f"JSON file '{path}' contains duplicate key: {k}")
            else:
                d[k] = v
        return d
    
    # Load JSON file and check for duplicate keys
    with open(path) as json_file:
        json_dict = json.load(json_file, object_pairs_hook=check_json_for_duplicate_keys)

    return json_dict


def write_json_file(data: Dict[str, Any], path: str, indent: int = 4) -> None:
    """
    Write a Python dictionary to a JSON file at the specified path.

    This function writes the contents of a Python dictionary to a JSON file at the specified path. 
    The JSON object is formatted with optional indentation for readability.

    Args:
        data (dict): The Python dictionary to be written to the JSON file.
        path (str): The path to the JSON file.
        indent (int, optional): The number of spaces used for indentation (default is 4).

    Returns:
        None

    Examples:
        >>> data = {'key1': 'value1', 'key2': 'value2'}
        >>> write_json_file(data, "output.json")
    """
    with open(path, 'w') as fp:
        json.dump(data, fp, indent=indent)
