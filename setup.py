from setuptools import setup, find_packages

setup(
    name="ResolvedorCaptcha",
    version="1.0.0",
    packages=find_packages(),
    package_data={"resolvedor_captcha": ["prompt.md"]},
    python_requires=">=3.10",
)
