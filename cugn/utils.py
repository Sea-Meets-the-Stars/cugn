""" Basic utilities for the CUGN project """

import numpy as np

import json
import gzip, os
import io

def line_endpoints(line:str):
    """
    Returns the longitude and latitude endpoints for a given line.

    Parameters:
        line (str): The line identifier.

    Returns:
        tuple: A tuple containing the longitude and latitude endpoints.

    Raises:
        ValueError: If the line identifier is not recognized.
    """

    if line == '56.0':
        lonendpts = [-123.328, -126.204]
        latendpts = [38.502, 37.186]
    elif line == '66.7':
        lonendpts = [-121.8371, -124.2000]
        latendpts = [36.8907, 35.7900]
    elif line == '80.0':
        lonendpts = [-120.4773,-123.9100]
        latendpts = [34.4703, 32.8200]
    elif line == '90.0':
        lonendpts = [-117.7475, -124.0000]
        latendpts = [33.5009, 30.4200]
    elif line == 'al':
        lonendpts = [-119.9593, -121.1500]
        latendpts = [32.4179, 34.1500]
    else:
        raise ValueError('line not recognized')

    return lonendpts, latendpts