# Databricks notebook source
# COMMAND ----------

project_config = {
    "project_name": "ccu000_template",
    "project_folder": [
        "/Workspace/Shared/SHDS/Jamie/ccu000_template",
        "/Workspace/Repos/my_username/ccu000_template"
    ],
    "start_date": "2025-01-01",
    "end_date": None,

    "users": [
        "jf774@cam.ac.uk",
        "teammate@health.org"
    ],

    "databases": {
        "live_view": "hive_metastore.dars_nic_391419_j3w9t",
        "archived_tables": "hive_metastore.dars_nic_391419_j3w9t_collab",
        "user_workspace": "hive_metastore.dsa_391419_j3w9t_collab",
        "lookup_tables": "hive_metastore.dss_corporate",
        "byod": "hive_metastore.bhf_cvd_covid_uk_byod",
    },

    "dependencies": {
        "hds_functions": {
            "repo": "/Workspace/Repo/jf774@cam.ac.uk/hds_functions",
            "version": "0.0.1",
        },
        "hds_reference_data": {
            "repo": "/Workspace/Repo/jf774@cam.ac.uk/hds_reference_data",
            "version": "0.0.1",
        },
    },

}

# COMMAND ----------
# MAGIC %run ./set_up
