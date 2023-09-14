from setuptools import setup, find_packages

setup(
    name="librispeech_samediff",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "click",
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "tqdm",
        "dtaidistance",
    ],
    package_data={"librispeech_samediff": ["data/*.csv"]},
    entry_points="""
        [console_scripts]
        libri-sd=librispeech_samediff.main:main
    """,
)
