import os
import re
import shutil

def remove_files_in_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def natural_sort(given_list):
    """ Sort the given list in the way that humans expect."""
    given_list.sort(key=alphanum_key)


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"] """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]


def tryint(s):
    try:
        return int(s)
    except:
        return s

def validate_file_format(file_in_path, allowed_format):
    if os.path.isfile(file_in_path) and os.path.splitext(file_in_path)[1][1:] in allowed_format:
        return True
    else:
        return False

def is_image(file_in_path):
    if validate_file_format(file_in_path, ['jpg', 'JPEG', 'png', 'JPG']):
        return True
    else:
        return False


def get_immediate_subfolder_paths(folder_path):
    subfolder_names = get_immediate_subfolder_names(folder_path)
    subfolder_paths = [os.path.join(folder_path, subfolder_name) for subfolder_name in subfolder_names]
    return subfolder_paths


def get_immediate_subfolder_names(folder_path):
    subfolder_names = [folder_name for folder_name in os.listdir(folder_path)
                      if os.path.isdir(os.path.join(folder_path, folder_name))]
    natural_sort(subfolder_names)
    return subfolder_names

def get_immediate_childfile_paths(folder_path, ext = None, exclude = None):
    files_names = get_immediate_childfile_names(folder_path, ext, exclude)
    files_full_paths = [os.path.join(folder_path, file_name) for file_name in files_names]
    return files_full_paths


def get_immediate_childfile_names(folder_path, ext = None, exclude = None):
    files_names = [file_name for file_name in next(os.walk(folder_path))[2]]
    if ext is not None:
        files_names = [file_name for file_name in files_names
                       if file_name.endswith(ext)]
    if exclude is not None:
        files_names = [file_name for file_name in files_names
                       if not file_name.endswith(exclude)]
    natural_sort(files_names)
    return files_names


def get_folder_name_from_path(folder_path):
    path, folder_name = os.path.split(folder_path)
    return folder_name


def get_parent_folder_from_path(folder_path):
    parent_folder_path = os.path.abspath(os.path.join(folder_path, os.pardir))
    parent_folder_name = os.path.basename(parent_folder_path)
    return parent_folder_path, parent_folder_name


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
