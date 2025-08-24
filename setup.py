"""
Setup script for Crypto Arbitrage Trading Platform
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="crypto-arbitrage-platform",
    version="1.0.0",
    author="Arbitrage Trading Team",
    author_email="team@arbitragetrading.com",
    description="Production-grade cryptocurrency arbitrage trading system",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/crypto-arbitrage-platform",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.11",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.12.0",
            "black>=23.11.0",
            "flake8>=6.1.0",
            "mypy>=1.7.0",
            "isort>=5.12.0",
            "pre-commit>=3.6.0",
            "bandit>=1.7.5",
            "safety>=2.3.0",
        ],
        "monitoring": [
            "prometheus-client>=0.19.0",
            "grafana-api>=1.0.3",
            "influxdb-client>=1.38.0",
        ],
        "ml": [
            "scikit-learn>=1.3.2",
            "tensorflow>=2.15.0",
            "torch>=2.1.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "arbitrage-demo=demo:main",
            "arbitrage-server=arbi.api.server:main",
            "arbitrage-dashboard=arbi.ui.dashboard:main",
        ],
    },
    include_package_data=True,
    package_data={
        "arbi": [
            "config/*.yaml",
            "ui/static/*",
            "ui/templates/*",
        ],
    },
    zip_safe=False,
    keywords="cryptocurrency trading arbitrage websocket real-time finance",
    project_urls={
        "Bug Reports": "https://github.com/your-org/crypto-arbitrage-platform/issues",
        "Source": "https://github.com/your-org/crypto-arbitrage-platform",
        "Documentation": "https://crypto-arbitrage-platform.readthedocs.io/",
    },
)
