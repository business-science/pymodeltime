from setuptools import setup, find_packages

# Define a helper function to determine if the platform is Windows
def is_windows():
    from sys import platform
    return platform.startswith('win')

# Define the torch and torchvision requirements
torch_requirements = [
    'torch==2.0.1+cpu;platform_system!="Windows"',
    'torchvision==0.15.2+cpu;platform_system!="Windows"'
]

# If the system is Windows, use a different version or exclude these requirements
if is_windows():
    torch_requirements = [
        # You can specify Windows-compatible versions or exclude them altogether
        # 'torch',
        # 'torchvision'
    ]

setup(
    name='pymodeltime',
    version='0.1',
    packages=find_packages(),
    author='Shafiullah Qureshi, Matt Dancho',
    author_email='qureshi.shafiullah@gmail.com, mdancho@business-science.io',
    install_requires=[
        'pip>=20.0',
        'setuptools>=40.0',
        'wheel>=0.30',
        'torch',  # Remove the specific version
        'torchvision',  # Remove the specific version
        'autogluon',
        'autogluon',
        'prophet',
        'pmdarima',
        'pandas',
        'statsmodels',
        'pytimetk',
        'xgboost',
        'scikit-learn',
        'h2o'
    ],
    package_data={
        'pymodeltime': ['data/*']
    },
    include_package_data=True
)
