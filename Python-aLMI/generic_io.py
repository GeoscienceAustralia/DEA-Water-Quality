#! /usr/bin/env python3
"""
 Generic code for reading (and eventually writing) data files of various formats.
 Supported formats:  netCDF, HDF

 The aim of this code is to provide a format independent interface to (at least)
 netCDF and HDF4 files types, the idea being that if theapplication code 
 can invoke a generic API, it will be easier to change storage formats over
 time.  Necessarily this involves some simplification of the range of 
 supported operations to a subset of common actions.

 After specifying the file_fmt in the "open" call, all other "methods" obtain 
 any necessary format-specific info from the handle.

 Exceptions from netCDF4 or pyhdf.SD are passed through to the caller, or re-raised; 
 errors that are detected by this code may result in "None" being returned to the 
 caller, or an exception being raised, as per documentation in each routine.

 Dimensions must be named, and can be shared between variables (as in netCDF).  
 The first dimension may be unlimited (specify size = 0).

 This code could also be implemented as a Python class.
"""

import netCDF4 as NC            # supports netCDF3 and netCDF4
import pyhdf.SD as HDF            # supports HDF4
import numpy as np
from collections import OrderedDict
import os.path
from struct import unpack
import sys

# Allowable file/handle types:
allowable_filetypes = [NC.Dataset, HDF.SD]

# The HDF datatypes that we support (do we need separate mappings for arrays and attrs?):
hdf_type = {
    np.dtype("?"):        HDF.SDC.UCHAR,    #  3
    np.dtype(str):        HDF.SDC.CHAR,        #  4
    np.dtype(np.int8):    HDF.SDC.INT8,        # 20
    np.dtype(np.uint8):   HDF.SDC.UINT8,    # 21
    np.dtype(np.int16):   HDF.SDC.INT16,    # 22
    np.dtype(np.uint16):  HDF.SDC.UINT16,    # 23
    np.dtype(np.int32):   HDF.SDC.INT32,    # 24
    np.dtype(int):        HDF.SDC.INT32,    # 24 (want int64, but HDF does not support it)
    np.dtype(np.uint32):  HDF.SDC.UINT32,    # 25
    np.dtype(np.float32): HDF.SDC.FLOAT32,    #  5
    np.dtype(np.float64): HDF.SDC.FLOAT64    #  6
}

hdf_int_types = [
    HDF.SDC.INT8,
    HDF.SDC.UINT8,
    HDF.SDC.INT16,
    HDF.SDC.UINT16,
    HDF.SDC.INT32,
    HDF.SDC.UINT32
]

hdf_float_types = [ HDF.SDC.FLOAT32, HDF.SDC.FLOAT64 ]

# Mapping from HDF data_type to NumPy data_type:
hdf2np_map = {
    HDF.SDC.CHAR:    np.char,
    HDF.SDC.CHAR8:   np.char,
    HDF.SDC.UCHAR8:  np.char,
    HDF.SDC.UINT8:   np.uint8,
    HDF.SDC.UINT16:  np.uint16,
    HDF.SDC.UINT32:  np.uint32,
    HDF.SDC.INT8:    np.int8,
    HDF.SDC.INT16:   np.int16,
    HDF.SDC.INT32:   np.int32,
    HDF.SDC.FLOAT32: np.float32,
    HDF.SDC.FLOAT64: np.float64
}

def is_int( hdf_attr_type ):
    return hdf_attr_type in hdf_int_types

def is_float( hdf_attr_type ):
    return hdf_attr_type in hdf_float_types

auto_file_types = {
    '.nc': 'NC',
    '.hdf': 'HDF',
    '.nc4': 'NETCDF4',
    '.nc3': 'NETCDF3_CLASSIC'
}

def open(filename, file_fmt=None, access_mode='r', verbose=False):
    """
    Open a file.
    Allowed formats are 
        HDF                 Defaults to HDF4
        HDF4                HDF4
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
       a file handle of type NC.Dataset or HDF.SD if successful
       Raises an exception if an unrecognised format or an unsupported access mode is requested.
    """

    ME = "generic_io.open"

    # Allowable file formats:
    allowable_hdf_formats = ['HDF', 'HDF4']
    allowable_nc_formats = ['NC',  'NETCDF4', 'NETCDF4_CLASSIC', 'NETCDF3_CLASSIC', 'NETCDF3_64BIT']
    allowable_file_formats = list(allowable_hdf_formats)        # we want a copy, not a view
    allowable_file_formats.extend(allowable_nc_formats)

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
            raise ValueError("filename must end in .nc or .hdf to auto-detect")

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
    elif file_fmt in allowable_hdf_formats:    # only 'HDF' is currently supported
        if access_mode == "r":
            hdf_mode = HDF.SDC.READ
        elif access_mode == "w":
            hdf_mode = HDF.SDC.WRITE + HDF.SDC.CREATE + HDF.SDC.TRUNC
        elif access_mode == "a":
            hdf_mode = HDF.SDC.WRITE
        try:
            handle = HDF.SD(filename, mode=hdf_mode)
        except:
            print(ME + ":  exception while trying to open " + filename)
            raise
    else:
        print("file_fmt " + file_fmt + " is not implemented (bug)")
        exit(1)

    return handle        # type will be NC.Dataset or HDF.SD

def get_file_format(handle, verbose=False):
    """
    Get the file format; will be one of ['HDF', 'NETCDF4', 'NETCDF4_CLASSIC', 'NETCDF3_CLASSIC', 'NETCDF3_64BIT']
    """

    if isinstance(handle, NC.Dataset):
        file_fmt = handle.file_format
    elif isinstance(handle, HDF.SD):
        file_fmt = 'HDF'		# (we don't support HDF5 in this module)
    else:
        raise ValueError("handle type must be in " + str(allowable_filetypes))

    return file_fmt

def close(handle, verbose=False):
    """
    Close the file.  
    Returns:
       True on success
       Raises an exception if the handle was not a valid type.
    """

    if isinstance(handle, NC.Dataset):
        handle.close()
    elif isinstance(handle, HDF.SD):
        handle.end()
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
    elif isinstance(handle, HDF.SD):
        if var_name is not None:
            hdf_obj = handle.select(var_name)        # an SDS (throws exception if var_name not in file)
        else:
            hdf_obj = handle
        attrs = get_hdf_attrs(hdf_obj)
    else:
        raise ValueError("handle type must be in " + str(allowable_filetypes))

    return attrs        # will be a dict

def set_attrs(handle, attrs, var_name=None, verbose=False):
    """
    Add a dictionary of attributes to the file open on handle.
    The default is to add global (file) attributes unless the
    name of a variable is provided in var_name. 
    Raises an exception if the handle was not a valid type.
    An exception is thrown if var_name is specified, and it is not in the file.

    NetCDF handles numeric attributes that are lists or numpy arrays.
    HDF handles numeric attributes that are lists.
    """

    ME = "generic_io.set_attrs"

    if not attrs:
        if verbose:  print(ME + ":  WARNING:  attrs =", attrs)
        return

    attr_name = None        # ensure it is defined
    try:
        if isinstance(handle, NC.Dataset):
            if var_name is not None:
                nc_obj = handle.variables[var_name]    # a Variable (throws exception if var_name not in file)
            else:
                nc_obj = handle
            if verbose:            # all DEBUG
                for attr_name in list(attrs.keys()):        # all DEBUG
                    attr_value = attrs[attr_name]        # may be multiple values
                    print(ME + ":  attr_name =", attr_name, "; attr_value =", attr_value)
                    py_attr_type = None
                    if type(attr_value) == np.ndarray:
                        attr_value_1d = attr_value.flatten()    # works for N-d, N = 0, 1, 2, ...
                        py_attr_type = type(attr_value_1d[0])
                        attr_value = attr_value_1d.tolist()
                        if verbose:  print(ME + ":  converted NumPy array to a list of type ", py_attr_type)
                    print(ME + ":  attr_name =", attr_name, "; attr_value =", attr_value)
                    if '__len__' in dir(attr_value) and len(attr_value) > 1:
                        if py_attr_type is None:
                            py_attr_type = type(attr_value[0])    # OK if only one level of "nesting"
                        print(ME + ":  py_attr_type =", py_attr_type, "; len(attr_value) =", len(attr_value))
                    else:
                        if py_attr_type is None:
                            py_attr_type = type(attr_value)
                        print(ME + ":  py_attr_type =", py_attr_type)
            nc_obj.setncatts(attrs)
        elif isinstance(handle, HDF.SD):
            if var_name is not None:
                hdf_obj = handle.select(var_name)        # an SDS (throws exception if var_name not in file)
            else:
                hdf_obj = handle
            for attr_name in list(attrs.keys()):
                attr = hdf_obj.attr(attr_name)        # an SDattr object
                attr_value = attrs[attr_name]        # may be multiple values
                py_attr_type = None
                if type(attr_value) == np.ndarray:
                    attr_value_1d = attr_value.flatten()    # works for N-d, N = 0, 1, 2, ...
                    py_attr_type = type(attr_value_1d[0])
                    attr_value = attr_value_1d.tolist()
                    if verbose:  print(ME + ":  converted NumPy array to a list of type ", py_attr_type)
                if verbose:  print(ME + ":  attr_name =", attr_name, "; attr_value =", attr_value)
                if '__len__' in dir(attr_value) and len(attr_value) > 1:
                    if py_attr_type is None:
                        py_attr_type = type(attr_value[0])    # OK if only one level of "nesting"
                    if verbose:  print(ME + ":  py_attr_type =", py_attr_type, "; len(attr_value) =", len(attr_value))
                    if py_attr_type == str:
                        use_attr_value = attr_value
                    else:
                        # use_attr_value = attr_value[-1]        # last element
                        use_attr_value = attr_value
                else:
                    if py_attr_type is None:
                        py_attr_type = type(attr_value)
                    if verbose:  print(ME + ":  py_attr_type =", py_attr_type)
                    use_attr_value = attr_value
                hdf_attr_type = hdf_type[np.dtype(py_attr_type)]
                if verbose:  print(ME + ":  hdf_attr_type =", hdf_attr_type)
                if verbose:  print(ME + ":  use_attr_value =", use_attr_value, "; type(use_attr_value) =", type(use_attr_value))
                if is_int( hdf_attr_type ) and type(use_attr_value) != list:
                    use_attr_value = int(use_attr_value)
                    if verbose:  print(ME + ":  now use_attr_value =", use_attr_value, "; type(use_attr_value) =", type(use_attr_value))
                elif is_float( hdf_attr_type ) and type(use_attr_value) != list:
                    use_attr_value = float(use_attr_value)
                    if verbose:  print(ME + ":  now use_attr_value =", use_attr_value, "; type(use_attr_value) =", type(use_attr_value))
                elif hdf_attr_type == HDF.SDC.CHAR and type(use_attr_value) != list and use_attr_value == '':
                    # We cannot seem to write out a zero length string so we'll need to convert it to something
                    # and this will have to do for the moment.
                    use_attr_value = 'NULL'

                attr.set(hdf_attr_type, use_attr_value)
        else:
            raise ValueError("handle type must be in " + str(allowable_filetypes))
    except:
        print(ME + ":  exception; attrs =", attrs, "; var_name =", var_name, "; attr_name =", attr_name)
        raise

    return

def get_attr(handle, attr_name, var_name=None, verbose=False):
    """
    Get the value of an particular attribute (global or variable).
    This is a convenience routine for calling get_attrs() and 
    offers the same return values, EXCEPT if the requested
    attribute does not exist, in which case None is returned.
    An exception is thrown if var_name is specified, and it is not in the file.
    """

    attr_value = get_attrs(handle, var_name=var_name).get(attr_name)        # None if doesn't exist

    return attr_value        # may be None; type may be str, unicode, list, float, numpy.float32, etc.

def get_dtype(handle, var_name):
    if isinstance(handle, NC.Dataset):
        return handle.variables[var_name].dtype
    elif isinstance(handle, HDF.SD):
        o = handle.select(var_name)
        sd_type = o.info()[3]
        for val in list(hdf_type.items()):
            if val[1] == sd_type:
                return val[0]
        return None
    else:
        raise ValueError("handle type must be in " + str(allowable_filetypes))

def get_datatype(var_obj, verbose=False):
    """
    Get the NumPy datatype of a variable object (a HDF.SDS or a NC.Variable).
    Returns the (Python) type (a class, which can be used to cast Python scalars), not the NumPy "dtype".

    The NumPy "dtype" can be obtained from the type by:  var_dtype = np.dtype(var_type)
    The type can be obtained from the NumPy "dtype" by:  var_type = var_dtype.type
    Either the type or the dtype can be used in "dtype=" keyword arguments to NumPy routines.
    """
    if isinstance(var_obj, NC.Variable):
        datatype = var_obj.dtype.type
        if verbose: print("NC.Variable; NumPy datatype =", datatype)
    elif isinstance(var_obj, HDF.SDS):
        datatype = hdf2np_map[var_obj.info()[3]]
        if verbose: print("HDF.SDS; NumPy datatype =", datatype)
    else:
        raise ValueError("var_obj is", type(var_obj), "; must be in " + str(["HDF.SDS", "NC.Variable"]))

    return datatype            # will be a NumPy datatype
    
def get_dims(handle, var_name=None, verbose=False):
    """
    Get the dimension info (global or variable).  
    Returns a dictionary keyed by dimension names & with sizes as values.
    Raises an exception if the handle was not a valid type.
    An exception is thrown if var_name is specified, and it is not in the file.
    """

    if isinstance(handle, NC.Dataset):
        dims = OrderedDict()            # to build the result
        if var_name is not None:
            var_obj = handle.variables[var_name]        # a Variable
            dim_names = var_obj.dimensions        # a tuple
            dim_sizes = var_obj.shape            # a tuple
            if verbose:  print("get_dims:  dim_names =", dim_names)        # DEBUG
            if verbose:  print("get_dims:  dim_sizes =", dim_sizes)        # DEBUG
            for j in range(len(dim_names)):
                dims[dim_names[j]] = dim_sizes[j]
        else:
            dims_dict = handle.dimensions        # an OrderedDict (names, Dimension objects)
            if verbose:  print("get_dims:  dims_dict =", dims_dict)        # DEBUG
            for dimName in list(dims_dict.keys()):
                dimSize = len(dims_dict[dimName])
                dims[dimName] = dimSize

    elif isinstance(handle, HDF.SD):
        if var_name is not None:
            dims = OrderedDict()            # to build the result
            var_obj = handle.select(var_name)        # an SDS
            dims_dict = var_obj.dimensions(full=True)    # a dict (names, dimension info in a tuple)
            if verbose:  print("get_dims:  dims_dict =", dims_dict)        # DEBUG
            index2name = {}            # mapping from dimIndex to dimName
            for dimName in list(dims_dict.keys()):
                dimIndex = dims_dict[dimName][1]        # the dimension index
                index2name[dimIndex] = dimName
            for dimIndex in sorted(index2name):
                dimName = index2name[dimIndex]
                dims[dimName] = dims_dict[dimName][0]                # the dimension size
        else:
            dims = {}            # to build the result
            vars = handle.datasets()        # a dict
            if verbose:  print("get_dims:  len(vars) =", len(vars))        # DEBUG
            if verbose:  print("get_dims:  vars =", vars)        # DEBUG
            # sort by variable index:
            index2name = {}            # mapping from varIndex to varName
            for varName in list(vars.keys()):
                varIndex = vars[varName][3]        # the variable index
                index2name[varIndex] = varName
                dimNames = vars[varName][0]        # a tuple
                dimSizes = vars[varName][1]        # a tuple
                for j in range(len(dimNames)):
                    dims[dimNames[j]] = dimSizes[j]

    else:
        raise ValueError("handle type must be in " + str(allowable_filetypes))

    return dims            # will be a dict; keys are the dimension names; values are the sizes

def create_group(handle, group_name, verbose=False):
    """
    Create a group.
    Returns:
       a handle to the group
       Raises an exception if the handle was not a valid type.
    """
    if isinstance(handle, NC.Dataset):
        groupObj = handle.createGroup(group_name)            # a Group
    else:
        raise ValueError("handle type must be in " + str([NC.Dataset]))

    return groupObj     # (only if we are returning the handle to the group)

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
    elif isinstance(handle, HDF.SD):
        # We don't handle groups for HDF but we also don't want
        # generic code to bomb.
        group_names = []
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
    elif isinstance(handle, HDF.SD):
        vars = handle.datasets()        # a dict
        var_names = list(vars.keys())            # a list (not in order)
    else:
        raise ValueError("handle type must be in " + str(allowable_filetypes))

    return var_names        # will be a list

def get_variable_indexed(handle, var_name, index_obj=None, verbose=False):
    """
    Get a variable array.  
    Returns the values as a NumPy array (may be a MaskedArray).
    Raises an exception if the handle was not a valid type.

    EAK - this is ok - but there will be times we don't want to read the whole variable
      so can't we just return var_obj, and let users access the data via var_obj[:,2] etc?
      as far as I can see that syntax works for both hdf and netcdf.
    """
    if isinstance(handle, NC.Dataset):
        var_obj = handle.variables[var_name]        # a Variable
        var_obj.set_auto_maskandscale(False)           # no surprises!
        if index_obj is not None:
            var = var_obj[index_obj]
        else:
            var = var_obj[:]
    elif isinstance(handle, HDF.SD):
        var_obj = handle.select(var_name)        # an SDS
        if index_obj is not None:
            var = var_obj[index_obj]
        else:
            var = var_obj.get()
    else:
        raise ValueError("handle type must be in " + str(allowable_filetypes))

    return var        # will be a NumPy array (may be a MaskedArray)

def get_variable(handle, var_name, group_name=None, verbose=False):
    """
    Get a variable object.  
    Returns the variable object from the underlying API (netCDF or HDF).
    The caller then reads/writes the values of the array with usual syntax of that API.
    """
    if isinstance(handle, NC.Dataset):
        if group_name:
            group = handle.groups[group_name]
        else:
            group = handle                      # i.e., the root group
        var_obj = group.variables[var_name]        # a Variable
        var_obj.set_auto_maskandscale(False)           # no surprises!
    elif isinstance(handle, HDF.SD):
        if group_name:
            raise ValueError("group_name keyword not implemented for", type(handle))
        var_obj = handle.select(var_name)        # an SDS
    else:
        raise ValueError("handle type must be in " + str(allowable_filetypes))

    return var_obj

def create_variable(handle, var_name, datatype, dims, CVs=None, fill_value=None, compress=False, verbose=False):
    """
    Create a variable array, for output.  Returns a handle to the variable.
    handle is the file (or group) handle.
    var_name is the variable name.
    datatype is the NumPy datatype (a numpy.dtype object, or something that can be converted to one)
    dims is an OrderedDict:  keys are dim names, values are dim sizes.
    compress is silently ignored if it is not implemented for the file's format
    TODO:  fill_value
    TODO:  CVs is an OrderedDict:  keys are dim names, values are the CV values (NumPy 1-D arrays).
    This is an unnecessary argument; CVs can be created with additional calls to this method.
    (But what about CVs in HDF files?)

    All writing of data is currently done using the "Python syntax".  E.g.:
    var = create_variable(...)
    var[indexExpr] = dataArray
    """

    ME = 'generic_io.create_variable'

    try:
        existing_dims = get_dims(handle)
    
        if isinstance(handle, NC.Dataset):
            if datatype == bool:
                raise ValueError("NetCDF does not support datatype = " + str(datatype))
            # First, we create the dimensions:
            for dimname in list(dims.keys()):
                if dimname not in existing_dims:
                    handle.createDimension(dimname, size=dims[dimname])
                else:
                    pass
    
            now_dims = get_dims(handle)
    
            var_obj = handle.createVariable(var_name, datatype, dimensions=list(dims.keys()), fill_value=fill_value, zlib=compress)
            var_obj.set_auto_maskandscale(False)               # no surprises!
        elif isinstance(handle, HDF.SD):
            hdf_datatype = hdf_type[np.dtype(datatype)]
            dim_names = list(dims.keys())
            dim_sizes = list(dims.values())
            var_obj = handle.create(var_name, hdf_datatype, dim_sizes)        # an SDS
            for j in range(len(dim_sizes)):
                dim_obj = var_obj.dim(j)
                dim_obj.setname(dim_names[j])
            if fill_value:  var_obj.setfillvalue(fill_value)
        else:
            raise ValueError("handle type must be in " + str(allowable_filetypes))
    except:
        print(ME + ":  exception; var_name =", var_name, "; datatype =", datatype, "; dims =", dims)
        raise

    if verbose:  print(ME + ":  " + var_name + "(" + str(datatype) + ") dims =", get_dims(handle,var_name=var_name))
    return var_obj        # (only if we are returning the handle to the variable)

"""
The following classes have not undergone rigid testing but
so far seem to work. ie use at your own risk.
Please note that the Dataset constructor takes the file format
and access mode in the opposite order to the open function above.
Also note that nested groups have not been tested.
"""
class Dataset:
    def __init__( self, filename, access_mode='r', file_fmt=None, verbose=False ):
        h = open( filename, file_fmt, access_mode, verbose )
        self.handle = h
        self.file_fmt = get_file_format( h )

    def close( self, verbose=False ):
        rc = close(self.handle,verbose )
        self.handle = None
        return rc

    def get_attrs(self, var_name=None, verbose=False):
        return get_attrs( self.handle, var_name, verbose )

    def set_attrs(self, attrs, var_name=None, verbose=False):
        return set_attrs( self.handle, attrs, var_name, verbose )

    def get_attr(self, attr_name, var_name=None, verbose=False):
        return get_attr( self.handle, attr_name, var_name, verbose )

    def get_dtype(self, var_name):
        return get_dtype( self.handle, var_name )

    def get_dims(self, var_name=None, verbose=False):
        return get_dims( self.handle, var_name, verbose )

    def create_group(self, group_name, verbose=False):
        gobj = create_group( self.handle, group_name, verbose)
        return Group( self, group_name, gobj )

    def get_group_names(self, verbose=False):
        return get_group_names( self.handle, verbose )

    def get_group(self, group_name, verbose=False):
        gobj = get_group( self.handle, group_name, verbose )
        return Group( self, group_name, gobj )

    def get_variable_names(self, verbose=False):
        return get_variable_names( self.handle, verbose )

    def get_variable(self, var_name, verbose=False):
        vobj = get_variable( self.handle, var_name, verbose )
        return Variable( self, var_name, vobj )

    def create_variable(self, var_name, datatype, dims, CVs=None, fill_value=None, verbose=False):
        vobj = create_variable( self.handle, var_name, datatype, dims, CVs, fill_value, verbose )
        return Variable( self, var_name, vobj )

    def get_file_format(self):
        return self.file_fmt

class Group:
    def __init__( self, handle, name, obj ):
        self.handle = handle
        self.name = name
        self.obj = obj

    def get_attrs(self, var_name=None, verbose=False):
        return get_attrs( self.obj, var_name, verbose )

    def set_attrs(self, attrs, var_name=None, verbose=False):
        return set_attrs( self.obj, attrs, var_name, verbose )

    def get_attr(self, attr_name, var_name=None, verbose=False):
        return get_attr( self.obj, attr_name, var_name, verbose )

    def get_dtype(self, var_name):
        return get_dtype( self.obj, var_name )

    def get_dims(self, var_name=None, verbose=False):
        return get_dims( self.obj, var_name, verbose )

    def create_group(self, group_name, verbose=False):
        #print 'Group::create_group() untested'
        gobj = create_group( self.obj, group_name, verbose)
        return Group( self, group_name, gobj )

    def get_group_names(self, verbose=False):
        #print 'Group::get_group_names() untested'
        return get_group_names( self.obj, verbose )

    def get_group(self, group_name, verbose=False):
        #print 'Group::get_group() untested'
        gobj = get_group( self.obj, group_name, verbose )
        return Group( self, group_name, gobj )

    def get_variable_names(self, verbose=False):
        return get_variable_names( self.obj, verbose )

    def get_variable(self, var_name, verbose=False):
        vobj = get_variable( self.obj, var_name, verbose )
        return Variable( self, var_name, vobj )

    def create_variable(self, var_name, datatype, dims, CVs=None, fill_value=None, verbose=False):
        vobj = create_variable( self.obj, var_name, datatype, dims, CVs, fill_value, verbose )
        return Variable( self, var_name, vobj )

class Variable:
    def __init__( self, parent, name, obj ):
        self.parent = parent
        self.name = name
        self.obj = obj
        # Disable auto masking and scaling by default for netCDF4 files
        # Ken has added this as the default to the main functions
        # so we no longer need to do this.
        #if isinstance(self.parent.handle, NC.Dataset):
        #    #print 'NC'
        #    self.obj.set_auto_maskandscale( False )
        #elif isinstance(self.parent.handle, Dataset):
        #    if isinstance(self.parent.handle.handle, NC.Dataset):
        #        #print 'NC'
        #        self.obj.set_auto_maskandscale( False )
        ##else:
        ##    print 'not NC'

    def get_attrs( self, verbose=False ):
        return self.parent.get_attrs( self.name, verbose )
    def set_attrs( self, attrs, verbose=False ):
        return self.parent.set_attrs( attrs, self.name, verbose )
    def get_attr( self, attr_name, verbose=False):
        return self.parent.get_attr( self.name, attr_name, verbose )

    def get_dims( self, verbose=False ):
        return self.parent.get_dims( self.name, verbose )
    def get_dtype( self ):
        return self.parent.get_dtype( self.name )

    #def __getitem__(self,obj, k):
    #    return self.obj(obj,k)
    def __getitem__(self,obj):
        return self.obj.__getitem__(obj)
    #def __setitem__(self,obj, k, v):
    #    return self.obj.setitem(obj,k, v)
    def __setitem__(self,obj, k):
        return self.obj.__setitem__(obj,k)


def mask_and_scale( data, attrs, out_dtype ):
    """
    Mask and scale data based on the way we believe they should be applied.
    This should be checked for each input dataset as various agencies
    interpret things differently to what we believe the standards to be.
    Returned will be a numpy masked array.
    """
    mdata = data
    # First try and mask out the data
    # For NASA bad_value_scaled is the raw stored value (not the scaled value ie it's counter-intuitive).
    for key in ['bad_value_scaled', '_FillValue', '_fillValue', 'fillValue', 'missing_value']:
        if key in attrs:
            mdata = np.ma.masked_equal( data, attrs[key] )
            break
    # We need to do the min/max value masking as NASA don't write out the bad_value_unscaled
    # and bad_value_scaled in the correct way (ie they write out the scaled value as the raw value!)
    if 'valid_min' in attrs:
        mdata = np.ma.masked_less( mdata, attrs['valid_min'] )
    if 'valid_max' in attrs:
        mdata = np.ma.masked_greater( mdata, attrs['valid_max'] )

    # Find the scale_factor
    scale_factor = 1.0
    for key in ['scale_factor', 'slope', 'Slope']:
        if key in attrs:
            scale_factor = attrs[key]
            break
    # Find the add_offset
    add_offset = 0.0
    for key in ['add_offset','intercept','Intercept']:
        if key in attrs:
            add_offset = attrs[key]
            break

    if isinstance( mdata, np.ma.MaskedArray ):
        return np.ma.asarray( mdata*scale_factor+add_offset, dtype=out_dtype )
    else:
        return np.asarray( mdata*scale_factor+add_offset, dtype=out_dtype )

def copy_variable( dsin, dsout, path ):

    inroot = dsin
    outroot = dsout
    for p in path[:-1]:
        # Create the groups if necessary
        if not p in inroot.get_group_names():
            print('Missing group %s in %s' % (p,'/'.join(path)))
            sys.exit(1)
        else:
            inroot = inroot.get_group( p )
        if not p in outroot.get_group_names():
            # create it
            outroot = outroot.create_group( p )
        else:
            outroot = outroot.get_group( p )

    # Make sure the variable is in the group
    varname = path[-1]
    if not path[-1] in inroot.get_variable_names():
        print('Missing variable %s in %s' % (varname,'/'.join(path)))
        sys.exit(1)
    vin = inroot.get_variable( varname )
    dims = vin.get_dims()
    dtype = vin.get_dtype()

    # Create it in the output
    vout = outroot.create_variable( varname, dtype, dims )

    # Copy attributes
    a = vin.get_attrs()
    vout.set_attrs( a )

    # Copy data
    vout[:] = vin[:]

def get_hdf_attrs(hdf_obj, verbose=False):        # (from remap.py)
    """
    Get the attrs (global or variable), convert to appropriate NumPy type, put in an OrderedDict

    HDF data types:
    SDC.CHAR       4    8-bit character
    SDC.UCHAR      3    unsigned 8-bit integer
    SDC.INT8      20    signed 8-bit integer
    SDC.UINT8     21    unsigned 8-bit integer
    SDC.INT16     22    signed 16-bit integer
    SDC.UINT16    23    unsigned 16-bit intege
    SDC.INT32     24    signed 32-bit integer
    SDC.UINT32    25    unsigned 32-bit integer
    SDC.FLOAT32    5    32-bit floating point
    SDC.FLOAT64    6    64-bit floating point
    """
    attrs = hdf_obj.attributes(True)
    sorted_items = sorted(list(attrs.items()), key=lambda t: t[1][1])         # sort by index
    result = OrderedDict()
    for item in sorted_items:
        key = item[0] 
        value = item[1][0]
        data_type = item[1][2]
        if data_type == HDF.SDC.CHAR or data_type == HDF.SDC.UCHAR:
            last_char = value[-1]
            #== last_val = unpack("h", last_char + "\0")[0]        # numeric value of last_char (convert tuple to scalar)
            last_val = unpack("h", bytes([ord(last_char), ord("\0")]) )[0]        # numeric value of last_char (convert tuple to scalar)
            if last_val == 0:
                new_value = value[0:-1]
            else:            # last char is a NUL
                new_value = value
            if verbose:  print("value = '" + value + "'; last_char = '" + last_char + "'; last_val = ", last_val, "; new_value = '" + new_value + "'")
        elif data_type == HDF.SDC.INT8:
            new_value = np.int8(value)
        elif data_type == HDF.SDC.UINT8:
            new_value = np.uint8(value)
        elif data_type == HDF.SDC.INT16:
            new_value = np.int16(value)
        elif data_type == HDF.SDC.UINT16:
            new_value = np.uint16(value)
        elif data_type == HDF.SDC.INT32:
            new_value = np.int32(value)
        elif data_type == HDF.SDC.UINT32:
            new_value = np.uint32(value)
        elif data_type == HDF.SDC.FLOAT32:
            new_value = np.float32(value)
        elif data_type == HDF.SDC.FLOAT64:
            new_value = np.float64(value)
        result[key] = new_value                 # just key and value
    return result

def test_read(filename, **cmd_args):
    """
    Unit tests of the read methods.
    """
    ME = "generic_io.test_read"
    print(ME + " starting; cmd_args =", cmd_args)

    file_format = cmd_args.get('file_format', None)    # argparse default is None
    access_mode = cmd_args.get('access_mode', None)    # argparse default is r
    var_name = cmd_args.get('var_name', None)
    verbose = cmd_args.get('verbose', None)        # argparse default is False
    print(ME + ":  file_format =", file_format)
    print(ME + ":  access_mode =", access_mode)
    print(ME + ":  var_name =", var_name)
    print(ME + ":  verbose =", verbose)

    print("\n" + ME + ":  open('" + filename + "', file_format=", file_format, ", mode='" + access_mode + "') ...")
    hh = open(filename, file_format, access_mode=access_mode, verbose=verbose)
    print(ME + ":  hh =", repr(hh))
    if hh is None: exit(1)

    print(ME + ":  get_file_format(hh) =", get_file_format(hh))

    print("\n" + ME + ":  get_attrs (global) ...")
    gattrs = get_attrs(hh, verbose=verbose)
    print(ME + ":  type =", type(gattrs), ", len =", len(gattrs))        # DEBUG
    # print ME + ":  gattrs =", gattrs        # may be a very long line!
    for attrName in list(gattrs.keys()):
        print("\t" + attrName + ": ", gattrs[attrName])
    if gattrs is None: exit(1)

    print("\n" + ME + ":  get_dims '" + filename + "' ...")
    dims = get_dims(hh, var_name=None, verbose=verbose)
    print(ME + ":  dims =", dims)

    if isinstance(hh, NC.Dataset):
        print("\n" + ME + ":  get_group_names ...")
        group_names = get_group_names(hh, verbose=verbose)
        print(ME + ":  type =", type(group_names), ", len =", len(group_names))        # DEBUG
        print(ME + ":  group_names =", group_names)

        for group_name in group_names:
            print("\n" + ME + ":  get_variable_names from " + group_name + " ...")
            group_obj = get_group(hh, group_name, verbose=verbose)
            var_names = get_variable_names(group_obj, verbose=verbose)
            print(ME + ":  type =", type(var_names), ", len =", len(var_names))        # DEBUG
            print(ME + ":  var_names =", var_names)
    else:
        print(ME + ":  don't do 'group' tests; hh is " + str(hh))

    print("\n" + ME + ":  get_variable_names ...")
    var_names = get_variable_names(hh, verbose=verbose)
    print(ME + ":  type =", type(var_names), ", len =", len(var_names))        # DEBUG
    print(ME + ":  var_names =", var_names)
    if var_names is None: exit(1)

    if var_name is not None:
        print("\n" + ME + ":  get_variable '" + var_name + "' ...")
        var_obj = get_variable(hh, var_name, verbose=verbose)
        print(ME + ":  type =", type(var_obj))        # DEBUG
        var_array = var_obj[:]
        print(ME + ":  type =", type(var_array), ", shape =", var_array.shape, ", dtype =", var_array.dtype)        # DEBUG
        print(ME + ":  var_array =", var_array)
        if var_array is None: exit(1)
    
        index_obj = (slice(1, None, 500), slice(2, 22, 3))
        print("\n" + ME + ":  get_variable_indexed '" + var_name + "' ", index_obj, "...")
        var_array = get_variable_indexed(hh, var_name, index_obj, verbose=verbose)
        print(ME + ":  type =", type(var_array), ", shape =", var_array.shape, ", dtype =", var_array.dtype)        # DEBUG
        print(ME + ":  var_array =", var_array)
        if var_array is None: exit(1)
    
        print("\n" + ME + ":  get_dims '" + var_name + "' ...")
        dims = get_dims(hh, var_name=var_name, verbose=verbose)
        print(ME + ":  dims =", dims)
    
        print("\n" + ME + ":  get_attrs '" + var_name + "' ...")
        var_attrs = get_attrs(hh, var_name=var_name, verbose=verbose)
        print(ME + ":  type =", type(var_attrs), ", len =", len(var_attrs))        # DEBUG
        # print ME + ":  var_attrs =", var_attrs
        for attrName in list(var_attrs.keys()):
            print("\t" + attrName + ": ", var_attrs[attrName])
        if var_attrs is None: exit(1)
    
        # test_attr_name = '_FillValue'
        test_attr_name = 'var_attr6'
    
        print("\n" + ME + ":  get_attr '" + var_name + "' '" + test_attr_name + "' ...")
        attr_value = get_attr(hh, test_attr_name, var_name=var_name, verbose=verbose)
        print(ME + ":  type =", type(attr_value))                    # DEBUG
        if '__len__' in dir(attr_value): print(ME + ":  len =", len(attr_value))        # DEBUG
        print(ME + ":  attr_value =", attr_value)
        if attr_value is None: exit(1)

    print("\n" + ME + ":  close ...")
    close(hh)

def test_write(filename, **cmd_args):
    """
    Unit tests of the write methods.
    """
    ME = "generic_io.test_write"
    print(ME + " starting; cmd_args =", cmd_args)

    file_format = cmd_args.get('file_format', None)    # argparse default is None
    access_mode = cmd_args.get('access_mode', None)    # argparse default is r
    verbose = cmd_args.get('verbose', None)        # argparse default is False
    print(ME + ":  file_format =", file_format)
    print(ME + ":  access_mode =", access_mode)
    print(ME + ":  verbose =", verbose)

    print("\n" + ME + ":  open('" + filename + "', file_format=", file_format, ", mode='" + access_mode + "') ...")
    hh = open(filename, file_format, access_mode=access_mode, verbose=verbose)
    if isinstance(hh, NC.Dataset):
        print(ME + ":  hh =", repr(hh), "; path =", hh.path)        # only works for NC
    else:
        print(ME + ":  don't do 'path' tests; hh is " + str(hh))
    if hh is None: exit(1)

    hh.glob_attr1 = 'Glob_Attr1'

    # create dimensions:
    dims = OrderedDict()
    dims['first_dim'] = 3
    dims['second_dim'] = 4

    # create array:
    shape = (dims['first_dim'], dims['second_dim'])
    nels = dims['first_dim'] * dims['second_dim']
    dtype = np.float32
    array = np.arange(nels, dtype=dtype) + 1000
    array.shape = shape

    # create variable:
    var_name = 'new_var'
    datatype = np.dtype(array.dtype)
    print("\n" + ME + ":  datatype =", datatype)
    var_obj = create_variable(hh, var_name, datatype, dims, CVs=None, fill_value=None, verbose=verbose)

    # write array to file:
    var_obj.var_attr0 = 'Var_Attr0'
    var_obj[:] = array
    var_obj.var_attr1 = 'Var_Attr1'

    hh.glob_attr2 = 'Glob_Attr2'
    # hh.glob_attr3 = ['Glob_Attr3a', 'Glob_Attr3b']    # joined into one string for nc; error for hdf

    # Add a dict of attrs:
    # attr_dict = {'var_attr3': 'Var_Attr3', 'var_attr4': 'Var_Attr4_aVar_Attr4_b', 'var_attr5': 55, 'var_attr6': [5, 555]}
    np6 = np.array(list(range(3)))
    print("np6 =", np6)
    np7 = np.array(list(range(1)))
    print("np7 =", np7)
    np8 = np.array(8)
    print("np8 =", np8)
    attr_dict = {'var_attr2': 'Var_Attr2', 'var_attr3': 'Var_Attr4_aVar_Attr4_b', 'var_attr4': 55, 'var_attr5': [5, 555], 'var_attr6': np6, 'var_attr7': np7, 'var_attr8': np8}
    set_attrs(hh, attr_dict, var_name=var_name, verbose=verbose)

    # create variable:
    var_name = 'new_var2'
    dims2 = OrderedDict()
    dims2['zeroth_dim'] = 0        # unlimited
    dims2['second_dim'] = 4
    dims2['third_dim'] = 6
    datatype = np.dtype(array.dtype)
    print("\n" + ME + ":  datatype =", datatype)
    var_obj2 = create_variable(hh, var_name, datatype, dims2, CVs=None, fill_value=None, verbose=verbose)
    shape2 = (dims2['second_dim'], dims2['third_dim'])        # one "plane"
    nels2 = dims2['second_dim'] * dims2['third_dim']
    array2 = np.arange(nels2, dtype=datatype) + 1000
    array2.shape = shape2
    print("array2 =", array2)
    var_obj2[0,:,:] = array2
    var_obj2[1,:,:] = array2 + 1000
    var_obj2[2,:,:] = array2 + 2000
    print("var_obj2 =", var_obj2)
    print(var_obj2[:])

    if isinstance(hh, NC.Dataset):
        # create a group:
        g1 = create_group(hh, "group_g1", verbose=verbose)
        print(ME + ":  type(g1) =", type(g1), "; path =", g1.path)
        var_name = 'new_var_in_g1'
        var_obj3 = create_variable(g1, var_name, datatype, dims2, CVs=None, fill_value=None, verbose=verbose)
        var_obj3[1,:,:] = array2 + 6000
        var_obj3.units = "no units"
        set_attrs(g1, {"attr1":"First", "attr2":222}, var_name=var_name, verbose=verbose)
        print("var_obj3 =", var_obj3)
        print(var_obj3[:])
    else:
        print(ME + ":  don't do 'create group' tests; hh is " + str(hh))

    # Attempt to write a variable of each supported datatype:
    supported_datatypes = list(hdf_type.keys())
    numeric_datatypes = set()
    for datatype in supported_datatypes:
        if hdf_type[datatype] != HDF.SDC.CHAR:
            numeric_datatypes.add(datatype)
    print("numeric_datatypes =", numeric_datatypes)
    datatype = float
    print("datatype =", datatype, "; str = '" + str(datatype) + "'")
    array_dt = np.array(array, dtype = datatype)        # use previously-created array
    var_name = 'var__float'
    var_obj = create_variable(hh, var_name, datatype, dims, CVs=None, fill_value=None, verbose=verbose)  # create variable
    datatype = 'float'
    print("datatype =", datatype, "; str = '" + str(datatype) + "'")
    array_dt = np.array(array, dtype = datatype)        # use previously-created array
    var_name = 'var__QfloatQ'
    var_obj = create_variable(hh, var_name, datatype, dims, CVs=None, fill_value=None, verbose=verbose)  # create variable
    datatype = int
    print("datatype =", datatype, "; str = '" + str(datatype) + "'")
    array_dt = np.array(array, dtype = datatype)        # use previously-created array
    var_name = 'var__int'
    var_obj = create_variable(hh, var_name, datatype, dims, CVs=None, fill_value=None, verbose=verbose)  # create variable
    datatype = np.float
    print("datatype =", datatype, "; str = '" + str(datatype) + "'")
    array_dt = np.array(array, dtype = datatype)        # use previously-created array
    var_name = 'var__np.float'
    var_obj = create_variable(hh, var_name, datatype, dims, CVs=None, fill_value=None, verbose=verbose)  # create variable
    for datatype in numeric_datatypes:
        dtype = np.dtype(datatype)      # make it a np.dtype object
        print("datatype =", datatype, "; str = '" + str(datatype) + "'", "; dtype =", dtype, "; str = '" + str(dtype) + "'")
        array_dt = np.array(array, dtype = dtype)        # use previously-created array
        var_name = 'var_' + str(dtype)
        if var_name not in get_variable_names(hh):
            try:
                var_obj = create_variable(hh, var_name, datatype, dims, CVs=None, fill_value=None, verbose=verbose)  # create variable
            except:
                print("variable '" + var_name + "' not created")
        else:
            print("already have variable " + var_name)

    print("\n" + ME + ":  close ...")
    close(hh)

# The '__main__' entry point is only used for testing.
if __name__ == '__main__':
    import argparse

    # Create an ArgumentParser:
    parser = argparse.ArgumentParser(description='Perform I/O on a data file (HDF, netCDF, etc.)')

    # Positional arguments (required): 
    parser.add_argument('data_file', metavar='DATA_FILE', type=str, help='Data file')

    # Optional arguments:
    parser.add_argument('--file_format', metavar='FILE_FORMAT', type=str, default=None, help='Format of data file (default:  None)')
    parser.add_argument('--access_mode', metavar='ACCESS_MODE', type=str, default='r', help='Access mode of data file (default:  r)')
    test_methods = ['test_read', 'test_write', 'test_append']
    parser.add_argument('--test_method', metavar='TEST_METHOD', type=str, default='test_read', choices=test_methods, help='Test method to run (default:  test_read)')
    parser.add_argument('--var_name', metavar='VAR_NAME', type=str, default=None, help='Name of variable to query (default:  None)')
    # parser.add_argument('--group_name', metavar='GROUP_NAME', type=str, default=None, help='Name of group to query (default:  None)')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose diagnostic messages (default:  False)')

    cmd_args = parser.parse_args()
    print("generic_io.py->__main__:  cmd_args =", cmd_args)

    kw_args = vars(cmd_args)            # convert to a dict
    data_file = kw_args.pop('data_file')                    # don't want it twice!
    print("generic_io.py->__main__:  kw_args =", kw_args)
    print("generic_io.py->__main__:  test_method =", kw_args.get('test_method'))

    # Here we execute the selected test method:
    if kw_args.get('test_method') == 'test_read':
        print("do test_read")
        test_read(data_file, **kw_args)
    elif kw_args.get('test_method') == 'test_write':
        print("do test_write")
        test_write(data_file, **kw_args)

    print(sys.argv[0] + " done")             # DEBUG
