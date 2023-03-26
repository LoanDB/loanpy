import yaml

config = {
    "jobs": {
        f"test-{platform}-python-{version}": {
            "steps": [
                {
                    "name": "Run tests",
                    "run": f"echo 'Running tests on {platform} with Python {version}'",
                }
            ],
        }
        for platform in ["linux", "macos", "windows"]
        for version in ["3.7", "3.8", "3.9", "3.10", "3.11"]
    },
}

with open("fixed_config.yml", "w+") as f:
    yaml.dump(config, f, sort_keys=False)
