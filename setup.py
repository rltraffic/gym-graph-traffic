from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='gym_graph_traffic',
    version='0.0.1',
    author="Dawid Borys, Maria Oparka",
    description="Road traffic simulator for OpenAI Gym",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rltraffic/gym-graph-traffic.git",
    install_requires=['gym>=0.15.3', 'numpy>=1.16.4', 'attrdict>=2.0.1', 'pygame>=1.9.6'],
    packages=find_packages(),
    package_data={
        'gym_graph_traffic': ['envs/*'],
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
