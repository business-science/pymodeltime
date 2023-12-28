from setuptools import setup, find_packages

setup(
    name='pymodeltime',
    version='0.1',
    packages=find_packages(),
    author='Shafiullah Qureshi, Matt Dancho',
    author_email='qureshi.shafiullah@gmail.com, mdancho@business-science.io',
    install_requires=[
        'torch==2.0.1+cpu',
        'torchvision==0.15.2+cpu',
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
