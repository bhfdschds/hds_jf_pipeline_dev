"""
Module name: environment_utils.py

Utilities for environment setup in Databricks notebooks.

Provides functions to:
- get_spark_session(): Initialize and return a SparkSession.
- resolve_path(): Build paths relative to the project root.
- find_project_folder(marker='.dbxproj'): Recursively locate the project root directory by searching for a marker file.

These helpers ensure reliable path resolution and environment setup in shared workspace contexts
where notebook locations and relative path behavior can vary.
"""

import os
import pkg_resources
from pyspark.sql import SparkSession

def get_spark_session():
    """
    Creates or retrieves a SparkSession object.

    This function initializes a SparkSession with the specified app name if it does
    not exist; otherwise, it retrieves the existing SparkSession.

    Returns:
    - spark_session (SparkSession): A SparkSession object.

    Example:
        >>> spark = get_spark_session()
    """
    spark_session = (
        SparkSession.builder
        .appName('SparkSession')
        .getOrCreate()
    )

    return spark_session


def resolve_path(path: str, repo: str = None) -> str:
    """
    Resolve the path to a file, handling different formats and repositories.

    Args:
    - path (str): Path to the file. Can be an absolute path, a relative path with './', or a path within a repo.
    - repo (str): Name of the repo if the path is relative within a repo.

    Returns:
    - resolved_path (str): Resolved absolute path to the file.
    """
    assert (
        (os.path.isabs(path) and repo is None) or 
        (path.startswith("./") and repo is None) or 
        (repo is not None and not (path.startswith("./") or os.path.isabs(path)))
    ), "Specify either an absolute path, a relative path with './', or a path within a repo, not a combination of them."

    if path.startswith("./"):
        project_folder = os.environ.get('PROJECT_FOLDER', None)
        assert project_folder is not None, "Environment variable 'PROJECT_FOLDER' not set. Ensure './project_setup' is run at the start of the notebook."
        resolved_path = os.path.join(project_folder, path[2:])
    elif repo is not None:
        resolved_path = pkg_resources.resource_filename(repo, path)
    else:
        resolved_path = path

    return resolved_path


def find_project_folder(marker_file=".dbxproj", workspace_prefix="/Workspace") -> str:
    """
    Walks up from the current notebook path to find the folder containing the marker file.
    Returns the absolute path to the project root.

    Args:
        marker_file (str): The name of the file used to identify the project root directory. 
            Defaults to ".dbxproj".
        workspace_prefix (str): The base path prefix for the notebook path, usually the workspace root. 
            Defaults to "/Workspace".

    Returns:
        str: Absolute path to the project root directory.

    Raises:
        FileNotFoundError: If the marker file is not found in any parent directories of the notebook path.

    Example:
        If your notebook is located at:
            /Workspace/Users/alice/my_project/notebooks/analysis.ipynb

        And your project has a marker file at:
            /Workspace/Users/alice/my_project/.dbxproj

        Then:
            >>> find_project_folder()
            '/Workspace/Users/alice/my_project'
    """

    context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    notebook_folder = f"{workspace_prefix}{os.path.dirname(context.notebookPath().get())}"

    current_path = notebook_folder
    while True:
        if current_path in ("", "/"):
            raise FileNotFoundError(f"Marker file '{marker_file}' not found in any parent directories of {notebook_folder}.")

        try:
            if marker_file in os.listdir(current_path):
                return current_path
        except (FileNotFoundError, PermissionError):
            pass  # Might hit a path not visible or accessible

        current_path = os.path.dirname(current_path)
