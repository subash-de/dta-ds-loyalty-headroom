from setuptools import setup, find_packages
from version import VERSION

package_name = "customer_headroom"
setup(
    name=package_name,
    version=VERSION,
    description="Predicting a customer's category purchase Headroom.",
    author="Benjamin Tunbridge",
    install_requires=[
        "dtaml==1.*",
        "scikit-surprise==1.1.1",
        "scikit-learn==0.24.2",
        "seaborn==0.11.1",
        "great-expectations==0.13.37",
        "flake8==4.0.1",
        "pandas==1.3.1",
        "numpy==1.20.3",
        "scipy==1.6.3",
        "pytest==6.2.4",
        "pyspark",
        "offerallocationv2==2.0.15rc66923"
    ],
    packages=find_packages(exclude=['tests', 'tests.*']),
    include_package_data=True,
    package_data={"": ["config.yaml"]},
)