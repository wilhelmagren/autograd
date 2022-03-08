from setuptools import setup

setup(name='autograd',
        version='0.1.0',
        author='Wilhelm Ågren',
        install_requires=['numpy', 'requests', 'torch'],
        python_requires='>=3.8',
        include_package_data=True)
