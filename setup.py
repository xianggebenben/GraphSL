from setuptools import setup, find_packages

setup(
    name='GraphSL',         # How you named your package folder (MyLib)
    packages=find_packages(),   # Chose the same as "name"
    version='0.10',      # Start with a small number and increase it with every change you make
    # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    license='MIT',
    # Give a short description about your library
    description='Graph Source Localization Approaches and Benchmark Datasets',
    author='Junxiang Wang',                   # Type in your name
    author_email='junxiang.wang@alumni.emory.edu',      # Type in your E-Mail
    # Provide either the link to your github or to your website
    url='https://xianggebenben.github.io/Junxiang_Wang.github.io/',
    # I explain this later on
    download_url='https://github.com/xianggebenben/GraphSL/archive/refs/tags/v.0.8.tar.gz',
    keywords=['Graph Diffusion', 'Graph Source Localization', 'Prescribed Methods',
              'GNN Methods', 'Benchmark'],   # Keywords that define your package best
    install_requires=[            # I get to this in a second
        'numpy',
        'networkx',
        'ndlib',
        'torch',
        'scikit-learn',
        'scipy',
        'torch_geometric'
    ],
    classifiers=[
        # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',   # Again, pick a license
        # Specify which pyhton versions that you want to support
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10'
    ],
    python_requires=">=3.9",
)
