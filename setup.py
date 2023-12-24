from setuptools import setup, find_packages

setup(
    name='pymodeltime',
    version='0.1',
    packages=find_packages(),
    author='Shafiullah Qureshi',
    author_email='qureshi.shafiullah@gmail.com',
    install_requires=[
        'pystan~=2.19.1.1',
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
        'pymodeltime': ['data/*']  # Include data files within the pymodeltime package
    },
    include_package_data=True
)
