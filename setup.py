# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from collections import defaultdict

import setuptools


def parse_requirements_file(
    path, allowed_extras: set = None, include_all_extra: bool = True
):
    requirements = []
    extras = defaultdict(list)
    find_links = []
    with open(path) as requirements_file:
        import re

        def fix_url_dependencies(req: str) -> str:
            """Pip and setuptools disagree about how URL dependencies should be handled."""
            m = re.match(
                r"^(git\+)?(https|ssh)://(git@)?github\.com/([\w-]+)/(?P<name>[\w-]+)\.git",
                req,
            )
            if m is None:
                return req
            else:
                return f"{m.group('name')} @ {req}"

        for line in requirements_file:
            line = line.strip()
            if line.startswith("#") or len(line) <= 0:
                continue
            if (
                line.startswith("-f")
                or line.startswith("--find-links")
                or line.startswith("--index-url")
            ):
                find_links.append(line.split(" ", maxsplit=1)[-1].strip())
                continue

            req, *needed_by = line.split("# needed by:")
            req = fix_url_dependencies(req.strip())
            if needed_by:
                for extra in needed_by[0].strip().split(","):
                    extra = extra.strip()
                    if allowed_extras is not None and extra not in allowed_extras:
                        raise ValueError(f"invalid extra '{extra}' in {path}")
                    extras[extra].append(req)
                if include_all_extra and req not in extras["all"]:
                    if "gpu" in extra:
                        extras["all-gpu"].append(req)
                    elif "cpu" in extra:
                        extras["all-cpu"].append(req)
                    else:
                        extras["all"].append(req)
            else:
                requirements.append(req)
    return requirements, extras, find_links


# allowed_extras = {"all", "all-cpu", "all-gpu", "onnx", "onnx-gpu", "serve", "dev", "faiss"}

# Load requirements.
install_requirements, extras, find_links = parse_requirements_file(
    "requirements.txt", #allowed_extras=allowed_extras
)

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="amzn_long_context_rag",
    author="Francesco Molfese",
    author_email="",
    description="Exploring Fine-Tuning for In-Context Retrieval and Efficient KV-Caching in Long-Context Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="question answering large language models long context rag",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    install_requires=install_requirements,
    extras_require=extras,
    python_requires=">=3.10",
)