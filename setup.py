from setuptools import setup, find_packages

setup(
    name="rplcopilot",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "typer",
        "langchain",
        "langchain-openai",
        "python-dotenv",
        "faiss-cpu",
        "openai",
        "rich",
    ],
    entry_points={
        "console_scripts": [
            "rpl=rpl.rpl:app",  # path to CLI app
        ]
    },
)
