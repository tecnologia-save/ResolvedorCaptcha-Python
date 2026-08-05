from setuptools import setup, find_packages

setup(
    name="ResolvedorCaptcha",
    version="1.0.0",
    packages=find_packages(),
    package_data={"resolvedor_captcha": ["prompt.md"]},
    python_requires=">=3.10",
    # O objeto `page` vem do chamador (as automacoes usam patchright), entao
    # playwright NAO entra aqui: declarar instalaria um pacote que ninguem importa.
    install_requires=[
        "google-genai>=1.0.0",
        "Pillow>=10.0.0",
    ],
)
