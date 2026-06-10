from setuptools import setup, find_packages

setup(
    name="captcha_uipath",
    version="1.0.0",
    packages=find_packages(),
    package_data={"captcha_uipath": ["prompt.md"]},
    python_requires=">=3.10",
)
