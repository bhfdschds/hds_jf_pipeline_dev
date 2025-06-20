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

def add_repos_to_sys_path(dependencies):
    """
    Adds repository paths from dependencies to sys.path if not already present.

    Args:
        dependencies (dict): Dictionary where keys are module names and values
                             are dicts with a 'repo' key specifying the path.
    """
    for dep in dependencies.values():
        repo_path = dep['repo']
        if repo_path not in sys.path:
            sys.path.append(repo_path)
            print(f"Added {repo_path} to sys.path.")


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


def add_and_check_dependencies(project_config, enforce=True):
    """
    Adds dependency repo paths to sys.path and checks for version compatibility.

    Args:
        project_config (dict): Full project configuration containing a 'dependencies' section.
        enforce (bool): Whether to raise an error on version mismatch.
    """
    dependencies = project_config.get('dependencies', {})
    add_repos_to_sys_path(dependencies)
    check_dependencies_versions(dependencies, enforce=enforce)


add_and_check_dependencies(project_config, enforce=True)

