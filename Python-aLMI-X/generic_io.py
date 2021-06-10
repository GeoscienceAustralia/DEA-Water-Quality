#! /usr/bin/env python3

import netCDF4 as NC            # supports netCDF3 and netCDF4
from collections import OrderedDict
import os.path
import sys

# Allowable file/handle types:
allowable_filetypes = [NC.Dataset]

auto_file_types = {
    '.nc': 'NC',
    '.nc4': 'NETCDF4',
    '.nc3': 'NETCDF3_CLASSIC'
}

def open(filename, file_fmt=None, access_mode='r', verbose=False):
    """
    Open a file.
    Allowed formats are 
        NC                  Defaults to NETCDF4
        NETDCF4
        NETCDF4_CLASSIC
        NETCDF3_CLASSIC
        NETCDF3_64BIT
    Supported access modes are:
        r     readonly (the default)
        w     write (creates new file; clobbers if already exists)
        a     append (read/write an existing file)

    Returns:
       a file handle of type NC.Dataset if successful
       Raises an exception if an unrecognised format or an unsupported access mode is requested.
    """

    ME = "generic_io.open"

    # Allowable file formats:
    allowable_nc_formats = ['NC',  'NETCDF4', 'NETCDF4_CLASSIC', 'NETCDF3_CLASSIC', 'NETCDF3_64BIT']
    allowable_file_formats = allowable_nc_formats

    # Allowable access modes:
    allowable_access_modes = ["r", "w", "a"]

    if access_mode not in allowable_access_modes:
        raise ValueError("access_mode must be in " + str(allowable_access_modes))

    if file_fmt is None:                # auto-detect
        if verbose:  print(ME + ":  infer file format from filename ...")
        (root, ext) = os.path.splitext(filename)
        if ext in auto_file_types:
            file_fmt = auto_file_types[ext]
            if verbose:  print(ME + ":  (auto) file_fmt = " + file_fmt)
        else:
            raise ValueError("filename must end in .nc to auto-detect")

    if file_fmt not in allowable_file_formats:
        raise ValueError("file_fmt must be in " + str(allowable_file_formats))

    if file_fmt in allowable_nc_formats:    # will be ignored unless access_mode = "w"
        if file_fmt == 'NC':
            format = 'NETCDF4'        # the default
        else:
            format = file_fmt
        try:
            handle = NC.Dataset(filename, mode=access_mode, format=format, clobber=True)
        except:
            print(ME + ":  exception while trying to open " + filename)
            raise
    else:
        print("file_fmt " + file_fmt + " is not implemented (bug)")
        exit(1)

    return handle        # type will be NC.Dataset

def close(handle, verbose=False):
    """
    Close the file.  
    Returns:
       True on success
       Raises an exception if the handle was not a valid type.
    """

    if isinstance(handle, NC.Dataset):
        handle.close()
    else:
        raise ValueError("handle type must be in " + str(allowable_filetypes))

    return True


def get_attrs(handle, var_name=None, verbose=False):
    """
    Return a dictionary with attributes from the file open on handle.
    The default is to return global (file) attributes unless the
    name of a variable is provided in var_name. 
    Raises an exception if the handle was not a valid type.
    An exception is thrown if var_name is specified, and it is not in the file.
    """

    if isinstance(handle, NC.Dataset):
        if var_name is not None:
            var_obj = handle.variables[var_name]    # a Variable (throws exception if var_name not in file)
            attrs = var_obj.__dict__        # an OrderedDict
        else:
            attrs = handle.__dict__        # an OrderedDict
    else:
        raise ValueError("handle type must be in " + str(allowable_filetypes))

    return attrs        # will be a dict


def get_group_names(handle, verbose=False):
    """
    Get the group names from a file.  
    Returns:
       a list with the names of groups
       Raises an exception if the handle was not a valid type.
    """
    if isinstance(handle, NC.Dataset):
        groups = handle.groups            # an OrderedDict
        group_names = list(groups.keys())
    else:
        raise ValueError("handle type must be in " + str(allowable_filetypes))

    return group_names        # will be a list


def get_group(handle, group_name, verbose=False):
    """
    Get a group from a file.  
    Returns:
       a handle to the group
       Raises an exception if the handle was not a valid type.
       Raises an exception if the group does not exist.
    """
    if isinstance(handle, NC.Dataset):
        groupObj = handle.groups[group_name]
    else:
        raise ValueError("handle type must be in " + str([NC.Dataset]))

    return groupObj        # will be a Group


def get_variable_names(handle, verbose=False):
    """
    Get the variable names from a file.  
    Returns:
       a list with the names of variables
       Raises an exception if the handle was not a valid type.
    """
    if isinstance(handle, NC.Dataset):
        vars = handle.variables            # an OrderedDict
        var_names = list(vars.keys())            # a list
        raise ValueError("handle type must be in " + str(allowable_filetypes))

    return var_names        # will be a list


def get_variable(handle, var_name, group_name=None, verbose=False):
    """
    Get a variable object.  
    Returns the variable object from the underlying API (netCDF).
    The caller then reads/writes the values of the array with usual syntax of that API.
    """
    if isinstance(handle, NC.Dataset):
        if group_name:
            group = handle.groups[group_name]
        else:
            group = handle                      # i.e., the root group
        var_obj = group.variables[var_name]        # a Variable
        var_obj.set_auto_maskandscale(False)           # no surprises!
    else:
        raise ValueError("handle type must be in " + str(allowable_filetypes))

    return var_obj
