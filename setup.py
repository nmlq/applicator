from pathlib import Path

from setuptools import find_packages, setup


PROJECT_ROOT = Path(__file__).parent


def read_requirements() -> list[str]:
    """Load non-empty, non-comment dependencies from requirements.txt.

    :return: Dependency requirement strings for ``install_requires``.
    """
    # Keeping dependency declarations in one file avoids setup metadata drift.
    requirements_file = PROJECT_ROOT / "requirements.txt"
    return [
        line.strip()
        for line in requirements_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


setup(
    name="applicator",
    version="0.1.0",
    description="Apply an LLM prompt to a CSV column row-by-row.",
    packages=find_packages(include=["applicator", "applicator.*"]),
    python_requires=">=3.9",
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "applicator=applicator.cli:main",
        ],
    },
)
