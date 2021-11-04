from setuptools import setup
from version import VERSION

package_name = "customer_headroom"
setup(
    name=package_name,
    version=VERSION,
    description="DESCRIPTION",
    author="AUTHOR",
    install_requires=[
        "dtaml>=1.*,<2.*",
    ],
    packages=[
        package_name,
    ],
    include_package_data=True,
    package_data={"": ["config.yaml"]},
)
