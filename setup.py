from setuptools import setup

setup(name='autograd',
        version='0.1.0',
        author='Wilhelm Ågren',
        install_requires=['numpy', 'requests'],
        python_requires='>=3.8',
        extras_requires={'testing':['torch','tqdm']}, 
        include_package_data=True)
