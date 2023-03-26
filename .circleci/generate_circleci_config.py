import yaml

python_versions = ["3.7", "3.8", "3.9", "3.10", "3.11"]
os_executor_map = {
    "macos": "macos-executor",
    "linux": "linux-executor",
    "windows": "windows-executor",
}

jobs = {}

for os_name, os_executor in os_executor_map.items():
    for python_version in python_versions:
        job_name = f"test-{os_name}-python-{python_version}"
        jobs[job_name] = {
            "executor": os_executor,
            "steps": [
                # Add your build and test steps here
            ],
            "parameters": {
                "python-version": {
                    "type": "string",
                    "default": python_version,
                },
            },
        }

config = {
    "version": 2.1,
    "orbs": {
        "windows": "circleci/windows@2.2.0",
        "macos": "circleci/macos@1.0.0",
        "linux": "circleci/linux@1.0.0",
    },
    "executors": {
        "macos-executor": {
            "macos": {
                "xcode": "12.5.1",
            },
        },
        "linux-executor": {
            "docker": [
                {
                    "image": "circleci/python:<< pipeline.parameters.python-version >>",
                },
            ],
        },
        "windows-executor": {
            "machine": {
                "image": "windows-server-2019-vs2019",
            },
        },
    },
    "jobs": jobs,
    "workflows": {
        "version": 2,
        "test-all-python-versions": {
            "jobs": [
                {"build-and-test": {"name": job_name, "executor": jobs[job_name]["executor"], "python-version": jobs[job_name]["parameters"]["python-version"]["default"]}}
                for job_name in sorted(jobs.keys())
            ],


        },
    },
}

with open("config.yml", "w") as f:
    yaml.dump(config, f)
