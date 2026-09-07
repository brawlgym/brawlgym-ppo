from setuptools import setup, find_packages


__version__ = None  # This will get replaced when reading version.py
exec(open('brawlgym_ppo/version.py').read())

with open('README.md', 'r') as readme_file:
    long_description = readme_file.read()


setup(
    name='brawlgym-ppo',
    packages=find_packages(),
    version=__version__,
    description='A multi-processed implementation of PPO for use with brawlgym.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='chrisrca',
    url='https://github.com/chrisrca/brawlgym-ppo',
    install_requires=[
        'brawlgym>=0.1.0',
        'numpy>1.21',
        'torch>1.13',
        'wandb>0.15',
    ],
    python_requires='>=3.11',
    license='Apache 2.0',
    license_file='LICENSE',
    keywords=['brawlhalla', 'gym', 'reinforcement-learning', 'ppo', 'brawlgym'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.11',
        'Operating System :: Microsoft :: Windows',
    ],
)
