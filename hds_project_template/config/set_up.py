# Databricks notebook source
# COMMAND ----------

# Set up in-memory logging
# Load and check dependencies
# Find project folder root and check
# Save project config .json to file
# Validate project configuration schema
# Set up file logs and flush memory logs to file
# Set enviromental variables
# Create table catalog
# Databricks configurations 



# Place holder for in-memory logging setup

# COMMAND ----------

# Load and check dependencies
import sys
import os

def add_repo_parents_to_sys_path(dependencies):
    """
    Adds parent directories of repositories to sys.path if not already present.

    Args:
        dependencies (dict): Dictionary where keys are module names and values
                             are dicts with a 'repo' key specifying the absolute path
                             to the module's source directory.

    Raises:
        ValueError: If the repo path is not absolute, its parent directory doesn't exist,
                    or the folder name does not match the module name.
    """
    for module_name, dep_config in dependencies.items():
        repo_path = dep_config['repo']

        if not os.path.isabs(repo_path):
            raise ValueError(f"Path for module '{module_name}' must be absolute: {repo_path}")

        parent_dir = os.path.dirname(repo_path)
        folder_name = os.path.basename(repo_path)

        if folder_name != module_name:
            raise ValueError(
                f"Declared module name '{module_name}' does not match actual folder name '{folder_name}' "
                f"in repo path '{repo_path}'."
            )

        if not os.path.isdir(parent_dir):
            raise ValueError(
                f"Parent directory '{parent_dir}' for module '{module_name}' does not exist."
            )

        if parent_dir not in sys.path:
            sys.path.append(parent_dir)
            print(f"Directory '{parent_dir}' added to sys.path for module '{module_name}'.")
        else:
            print(f"Directory '{parent_dir}' already in sys.path for module '{module_name}'.")



def check_dependencies_versions(dependencies, enforce=False):
    """
    Checks whether dependency modules match the expected versions.

    Args:
        dependencies (dict): Dictionary with module names as keys. Each value should be a dict
                             containing at least 'repo', and optionally 'version'.
        enforce (bool): If True, raise RuntimeError on version mismatch. If False, print warnings.

    Raises:
        RuntimeError: If enforce is True and a version mismatch is found.
    """
    for name, dep in dependencies.items():
        expected_version = dep.get('version')

        if expected_version is None:
            continue  # Skip version check if not specified

        try:
            module = __import__(name)
            actual_version = getattr(module, '__version__', None)
        except ImportError:
            actual_version = None

        if expected_version != actual_version:
            message = (
                f"Version mismatch for {name}. "
                f"Expected {expected_version}, found {actual_version}"
            )
            if enforce:
                raise RuntimeError(message)
            else:
                print(f"Warning: {message}")
        else:
            print(f"Version match for {name}: {expected_version}")


def add_and_check_dependencies(project_config, enforce=True):
    """
    Adds dependency repo paths to sys.path and checks for version compatibility.

    Args:
        project_config (dict): Full project configuration containing a 'dependencies' section.
        enforce (bool): Whether to raise an error on version mismatch.
    """
    dependencies = project_config.get('dependencies', {})
    add_repo_parents_to_sys_path(dependencies)
    check_dependencies_versions(dependencies, enforce=enforce)


add_and_check_dependencies(project_config, enforce=True)

# COMMAND ----------

# Find project folder root and check

from hds_functions import find_project_folder

def set_and_validate_project_folder(project_config, marker_file=".dbxproj", workspace_prefix="/Workspace"):
    """
    Finds and validates the project root directory, and sets it as an environment variable.

    This function:
    - Locates the project root using a marker file.
    - Ensures the detected path is one of the allowed folders in project_config.
    - Sets it as the PROJECT_FOLDER environment variable.

    Args:
        project_config (dict): The full project configuration containing a 'project_folder' key.
        marker_file (str): Name of a file that marks the project root.
        workspace_prefix (str): Expected prefix for paths in the workspace.

    Raises:
        RuntimeError: If the project folder can't be found or is not among allowed folders.
    """
    try:
        project_folder = find_project_folder(marker_file=marker_file, workspace_prefix=workspace_prefix)
    except Exception as e:
        raise RuntimeError(f"Failed to locate project folder: {e}")

    allowed_folders = project_config.get("project_folder", [])

    if project_folder not in allowed_folders:
        raise RuntimeError(
            f"Detected project folder '{project_folder}' is not in the allowed list:\n{allowed_folders}"
        )

    os.environ["PROJECT_FOLDER"] = project_folder
    print(f"Set PROJECT_FOLDER to: {project_folder}")


# COMMAND ----------

# Save project config .json to file
from hds_functions import write_json_file

write_json_file(data=project_config, path='./config/project_config.json', indent=4)

# COMMAND ----------
# Validate project configuration schema
from hds_functions import read_json_file
from jsonschema import validate

project_config_loaded = read_json_file(path='./config/project_config.json')
project_config_schema = read_json_file(path='./config/project_config_schema.json')

validate(instance=project_config_loaded, schema=project_config_schema)

