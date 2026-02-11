# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

import re
import json
import random
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, MutableMapping
from pathlib import Path
from typing import Any
from datasets import load_dataset, Features, Value, Sequence

try:
    # Python >= 3.12
    from typing import override  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - fallback for older Python versions
    from typing_extensions import override  # type: ignore

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
PromptObject = Mapping[str, str]
DatasetRow = MutableMapping[str, Any]

__all__: list[str] = [
    "DataLoader",
    "LongBenchV2Loader",
]

SEED=42
random.seed(SEED)

# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class DataLoader(ABC):
    """Abstract base class for loaders that stream examples from datasets."""

    def __init__(
        self,
        dataset_path: str,
        dataset_split: str,
        dataset_name: str | None,
        prompt_obj: PromptObject | str,
    ) -> None:
        self.dataset_path: str = dataset_path
        self.dataset_split: str = dataset_split
        self.dataset_name: str | None = dataset_name
        self.prompt_obj: PromptObject | str = prompt_obj

    # ---------------------------------------------------------------------
    # Public API — to be overridden by subclasses
    # ---------------------------------------------------------------------
    @abstractmethod
    def iterate(self) -> Iterator[DatasetRow]:
        """Yield dataset rows formatted for chat-based models."""
        ...

    # Allow ``for row in loader``
    def __iter__(self) -> Iterator[DatasetRow]:  # pragma: no cover
        return self.iterate()

    # Optional convenience — only works for *non-streaming* datasets
    def __len__(self) -> int:  # pragma: no cover
        try:
            return len(self.dataset)  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            err_msg = "Dataset length is unavailable (likely due to streaming mode)."
            raise TypeError(err_msg) from None


# ---------------------------------------------------------------------------
# LongBench-v2 implementation
# ---------------------------------------------------------------------------
class LongBenchV2Loader(DataLoader):
    """Loader for the *LongBench-v2* benchmark.

    Parameters
    ----------
    dataset_path : str, optional
        HuggingFace dataset repository path. Defaults to ``"THUDM/LongBench-v2"``.
    dataset_split : str, optional
        Dataset split to load (e.g. ``"train"``, ``"test"``). Defaults to ``"train"``.
    dataset_name : str | None, optional
        Specific configuration name inside the dataset. Defaults to ``None``.
    prompt_obj : PromptObject | str, optional
        Either a mapping with *system_instruction* and *user_content_template*
        keys or a path to a JSON file containing such a mapping. Defaults to
        ``"prompts/inference/LongBench-v2/zero_shot.json"``.
    """

    DEFAULT_PROMPT_PATH: Path = Path("prompts/inference/LongBench-v2/zero_shot.json")

    def __init__(
        self,
        dataset_path: str = "THUDM/LongBench-v2",
        dataset_split: str = "train",
        dataset_name: str | None = None,
        prompt_obj: PromptObject | str = DEFAULT_PROMPT_PATH,
        continue_final_message: bool = False
    ) -> None:
        super().__init__(dataset_path, dataset_split, dataset_name, prompt_obj)

        # HuggingFace `load_dataset` yields examples as dicts. Enable *streaming*
        # to avoid loading the entire split into memory.
        self.dataset = load_dataset(
            path=self.dataset_path,
            name=self.dataset_name,
            split=self.dataset_split,
            streaming=True,
        )

        # ------------------------------------------------------------------
        # Prompt resolution & validation
        # ------------------------------------------------------------------
        if isinstance(prompt_obj, str | Path):
            prompt_obj = self._load_prompt_obj(Path(prompt_obj))

        self._validate_prompt_obj(prompt_obj)

        self.system_instruction: str = prompt_obj["system_instruction"]  # type: ignore
        self.user_content_template: str = prompt_obj["user_content_template"]  # type: ignore

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_prompt_obj(path: Path) -> PromptObject:
        """Load and return the prompt configuration stored at *path*."""
        with path.expanduser().open(encoding="utf-8") as fh:
            return json.load(fh)

    @staticmethod
    def _validate_prompt_obj(prompt_obj: Mapping[str, Any]) -> None:
        """Ensure *prompt_obj* contains the required keys; otherwise raise."""
        required = {"system_instruction", "user_content_template"}
        missing = required - prompt_obj.keys()
        if missing:
            joined = ", ".join(sorted(missing))
            err_msg = f"Prompt object missing required keys: {joined}"
            raise ValueError(err_msg)


    def _format_user_content(self, **data: Any) -> str:
        """Fill the user template with fields extracted from *data*."""
        try:
            placeholders = {
                "DOC": data["context"],
                "Q": data["question"],
                "C_A": data["choice_A"],
                "C_B": data["choice_B"],
                "C_C": data["choice_C"],
                "C_D": data["choice_D"],
            }
        except KeyError as exc:  # pragma: no cover
            missing_key = exc.args[0]
            err_msg = f"Dataset row is missing expected field '{missing_key}'."
            raise KeyError(err_msg) from exc

        return self.user_content_template.format(**placeholders)

    # ------------------------------------------------------------------
    # Public API implementation
    # ------------------------------------------------------------------
    @override
    def iterate(self) -> Iterator[DatasetRow]:
        """Yield dataset rows enriched with an OpenAI-style ``messages`` list."""
        for example in self.dataset:
            row_dict: DatasetRow = dict(example)  # type: ignore[arg-type]

            user_content = self._format_user_content(**row_dict)
            row_dict["messages"] = [
                {"role": "system", "content": self.system_instruction},
                {"role": "user", "content": user_content},
            ]

            yield row_dict

class InfiniteBenchLoader(DataLoader):
    """Loader for the *InfiniteBench* benchmark.

    Parameters
    ----------
    dataset_path : str, optional
        HuggingFace dataset repository path. Defaults to ``"xinrongzhang2022/InfiniteBench2"``.
    dataset_split : str, optional
        Dataset split to load (e.g. ``"passkey"``, ``"kv_retrieval"``). Defaults to ``"longbook_qa_eng"``.
    dataset_name : str | None, optional
        Specific configuration name inside the dataset. Defaults to ``None``.
    prompt_obj : PromptObject | str, optional
        Either a mapping with *system_instruction* and *user_content_template*
        keys or a path to a JSON file containing such a mapping. Defaults to
        ``"prompts/inference/InfiniteBench/zero_shot.json"``.
    """

    DEFAULT_PROMPT_PATH: Path = Path("prompts/inference/InfiniteBench/zero_shot.json")

    def __init__(
        self,
        dataset_path: str = "xinrongzhang2022/InfiniteBench",
        dataset_split: str = "longbook_qa_eng",
        dataset_name: str | None = None,
        prompt_obj: PromptObject | str = DEFAULT_PROMPT_PATH,
        continue_final_message: bool = False
    ) -> None:
        super().__init__(dataset_path, dataset_split, dataset_name, prompt_obj)

        # HuggingFace `load_dataset` yields examples as dicts. Enable *streaming*
        # to avoid loading the entire split into memory.
        ft = Features({
            "id": Value("int64"),
            "context": Value("string"),
            "input": Value("string"),
            "answer": Sequence(Value("string")),
            "options": Sequence(Value("string"))
        })
        
        self.dataset = load_dataset(
            path=self.dataset_path,
            name=self.dataset_name,
            split=self.dataset_split,
            features=ft,
            streaming=True,
        )

        # ------------------------------------------------------------------
        # Prompt resolution & validation
        # ------------------------------------------------------------------
        if isinstance(prompt_obj, str | Path):
            prompt_obj = self._load_prompt_obj(Path(prompt_obj))
            prompt_obj = [p for p in prompt_obj if p["split"] == dataset_split][0]

        self._validate_prompt_obj(prompt_obj)

        self.split: str = prompt_obj["split"] # type: ignore
        self.system_instruction: str = prompt_obj["system_instruction"]  # type: ignore
        self.user_content_template: str = prompt_obj["user_content_template"]  # type: ignore

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_prompt_obj(path: Path) -> PromptObject:
        """Load and return the prompt configuration stored at *path*."""
        with path.expanduser().open(encoding="utf-8") as fh:
            return json.load(fh)

    @staticmethod
    def _validate_prompt_obj(prompt_obj: Mapping[str, Any]) -> None:
        """Ensure *prompt_obj* contains the required keys; otherwise raise."""
        required = {"system_instruction", "user_content_template"}
        missing = required - prompt_obj.keys()
        if missing:
            joined = ", ".join(sorted(missing))
            err_msg = f"Prompt object missing required keys: {joined}"
            raise ValueError(err_msg)


    def _format_user_content(self, **data: Any) -> str:
        """Fill the user template with fields extracted from *data*."""
        if self.split == "longbook_qa_eng":
            try:
                placeholders = {
                    "context": data["context"],
                    "question": data["input"],
                }
            except KeyError as exc:  # pragma: no cover
                missing_key = exc.args[0]
                err_msg = f"Dataset row is missing expected field '{missing_key}'."
                raise KeyError(err_msg) from exc
        
        elif self.split in ["longbook_sum_eng", "longdialogue_qa_eng", "math_calc"]:
            try:
                placeholders = {
                    "context": data["context"],
                }
            except KeyError as exc:  # pragma: no cover
                missing_key = exc.args[0]
                err_msg = f"Dataset row is missing expected field '{missing_key}'."
                raise KeyError(err_msg) from exc
            
        elif self.split == "longbook_choice_eng":
            try:
                choice_a, choice_b, choice_c, choice_d = data["options"][0], data["options"][1], data["options"][2], data["options"][3]
                placeholders = {
                    "context": data["context"],
                    "question": data["input"],
                    "choice_a": choice_a,
                    "choice_b": choice_b,
                    "choice_c": choice_c,
                    "choice_d": choice_d,
                }
            except KeyError as exc:  # pragma: no cover
                missing_key = exc.args[0]
                err_msg = f"Dataset row is missing expected field '{missing_key}'."
                raise KeyError(err_msg) from exc
        
        elif self.split in ["passkey", "number_string", "kv_retrieval"]:
            try:
                placeholders = {
                    "context": data["context"],
                    "input": data["input"]
                }
            except KeyError as exc:  # pragma: no cover
                missing_key = exc.args[0]
                err_msg = f"Dataset row is missing expected field '{missing_key}'."
                raise KeyError(err_msg) from exc
        
        elif self.split == "math_find":
            try:
                prompt = data["input"]
                context = data["context"]
                find_result = re.findall(r"The .+ of", prompt)
                assert find_result, f"Cannot find the target number in {prompt}"
                target_number = find_result[0].lower()[:-3]
                prefix = f"What is {target_number} in the following list?"
                placeholders = {
                    "prefix": prefix,
                    "context": context,
                    "input": prompt,
                }
            except KeyError as exc:  # pragma: no cover
                missing_key = exc.args[0]
                err_msg = f"Dataset row is missing expected field '{missing_key}'."
                raise KeyError(err_msg) from exc

        elif self.split == "code_run":
            try:
                find_result = re.findall(r"func_[0-9]+\(\-?[0-9]+\)", data["input"])
                func_call = find_result[0]
                func = func_call.split("(")[0]
                placeholders= {
                    "func": func,
                    "func_call": func_call,
                    "context": data["context"],
                }
            except KeyError as exc:  # pragma: no cover
                missing_key = exc.args[0]
                err_msg = f"Dataset row is missing expected field '{missing_key}'."
                raise KeyError(err_msg) from exc
        
        elif self.split == "code_debug":
            try:
                code = data["context"]
                placeholders = {
                    "context": code,
                    "OPTION_A": data["options"][0],
                    "OPTION_B": data["options"][1],
                    "OPTION_C": data["options"][2],
                    "OPTION_D": data["options"][3],
                }
            except KeyError as exc:  # pragma: no cover
                missing_key = exc.args[0]
                err_msg = f"Dataset row is missing expected field '{missing_key}'."
                raise KeyError(err_msg) from exc
            
        return self.user_content_template.format(**placeholders)

    # ------------------------------------------------------------------
    # Public API implementation
    # ------------------------------------------------------------------
    @override
    def iterate(self) -> Iterator[DatasetRow]:
        """Yield dataset rows enriched with an OpenAI-style ``messages`` list."""
        for example in self.dataset:
            row_dict: DatasetRow = dict(example)  # type: ignore[arg-type]

            user_content = self._format_user_content(**row_dict)
            row_dict["messages"] = [
                {"role": "system", "content": self.system_instruction},
                {"role": "user", "content": user_content},
            ]

            yield row_dict


class LoongLoader(DataLoader):
    """Loader for the *Loong* benchmark.

    Parameters
    ----------
    dataset_path : str, optional
        HuggingFace dataset repository path. Defaults to ``"framolfese/Loong"``.
    dataset_split : str, optional
        Dataset split to load (e.g. ``"financial"``, ``"paper"``). Defaults to ``"financial"``.
    dataset_name : str | None, optional
        Specific configuration name inside the dataset. Defaults to ``None``.
    prompt_obj : PromptObject | str, optional
        Either a mapping with *system_instruction* and *user_content_template*
        keys or a path to a JSON file containing such a mapping. Defaults to
        ``"prompts/inference/Loong/zero_shot.json"``.
    """

    DEFAULT_PROMPT_PATH: Path = Path("prompts/inference/Loong/zero_shot.json")

    def __init__(
        self,
        dataset_path: str = "framolfese/Loong",
        dataset_split: str = "financial",
        dataset_name: str | None = None,
        prompt_obj: PromptObject | str = DEFAULT_PROMPT_PATH,
        continue_final_message: bool = False
    ) -> None:
        super().__init__(dataset_path, dataset_split, dataset_name, prompt_obj)

        # HuggingFace `load_dataset` yields examples as dicts. Enable *streaming*
        # to avoid loading the entire split into memory.
        ft = Features({
            "level": Value("int64"),
            "set": Value("int64"),
            "length": Value("int64"),
            "type": Value("string"),
            "language": Value("string"),
            "question": Value("string"),
            "instruction": Value("string"),
            "prompt_template": Value("string"),
            "doc": Sequence(Value("string")),
            "answer": Value("string"),
            "shuffle_doc": Value("bool"),
            "id": Value("string"),
            "docs": Value("string"),
        })

        self.dataset = load_dataset(
            path=self.dataset_path,
            name=self.dataset_name,
            split=self.dataset_split,
            features=ft,
            streaming=True,
        )
        # Changing column name to be consistent with other datasets.
        self.dataset = self.dataset.rename_column("docs", "context") 

        # ------------------------------------------------------------------
        # Prompt resolution & validation
        # ------------------------------------------------------------------
        if isinstance(prompt_obj, str | Path):
            prompt_obj = self._load_prompt_obj(Path(prompt_obj))
            prompt_obj = [p for p in prompt_obj if p["split"] == dataset_split][0]

        self._validate_prompt_obj(prompt_obj)

        self.split: str = prompt_obj["split"] # type: ignore
        self.system_instruction: str = prompt_obj["system_instruction"]  # type: ignore
        self.user_content_template: str = prompt_obj["user_content_template"]  # type: ignore

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_prompt_obj(path: Path) -> PromptObject:
        """Load and return the prompt configuration stored at *path*."""
        with path.expanduser().open(encoding="utf-8") as fh:
            return json.load(fh)

    @staticmethod
    def _validate_prompt_obj(prompt_obj: Mapping[str, Any]) -> None:
        """Ensure *prompt_obj* contains the required keys; otherwise raise."""
        required = {"system_instruction", "user_content_template"}
        missing = required - prompt_obj.keys()
        if missing:
            joined = ", ".join(sorted(missing))
            err_msg = f"Prompt object missing required keys: {joined}"
            raise ValueError(err_msg)


    def _format_user_content(self, **data: Any) -> str:
        """Fill the user template with fields extracted from *data*."""
        try:
            placeholders = {
                "context": data["context"],
                "instruction": data["instruction"],
                "question": data["question"],
            }
        except KeyError as exc:  # pragma: no cover
            missing_key = exc.args[0]
            err_msg = f"Dataset row is missing expected field '{missing_key}'."
            raise KeyError(err_msg) from exc
            
        return self.user_content_template.format(**placeholders)

    # ------------------------------------------------------------------
    # Public API implementation
    # ------------------------------------------------------------------
    @override
    def iterate(self) -> Iterator[DatasetRow]:
        """Yield dataset rows enriched with an OpenAI-style ``messages`` list."""
        for example in self.dataset:
            row_dict: DatasetRow = dict(example)  # type: ignore[arg-type]

            user_content = self._format_user_content(**row_dict)
            row_dict["messages"] = [
                {"role": "system", "content": self.system_instruction},
                {"role": "user", "content": user_content},
            ]

            yield row_dict


# ---------------------------------------------------------------------------
# HELMETLoader implementation
# ---------------------------------------------------------------------------
class HELMETLoader(DataLoader):
    """Loader for the *HELMET* benchmark.

    Parameters
    ----------
    dataset_path : str, optional
        Local dataset path. Defaults to ``"data/benchmarks/HELMET"``.
    dataset_split : str, optional
        Dataset split to load (e.g. ``"kilt"``, ``"alce"``). Defaults to ``"kilt"``.
    dataset_name : str, optional
        Specific file name inside the split. Defaults to ``hotpotqa-dev-multikilt_1000_k1000_dep3.jsonl``.
    prompt_obj : PromptObject | str, optional
        Either a mapping with *system_instruction* and *user_content_template*
        keys or a path to a JSON file containing such a mapping. Defaults to
        ``"prompts/inference/HELMET/zero_shot.json"``.
    """

    DEFAULT_PROMPT_PATH: Path = Path("prompts/inference/HELMET/zero_shot.json")

    def __init__(
        self,
        dataset_path: str = "data/benchmarks/HELMET",
        dataset_split: str = "kilt",
        dataset_name: str = "hotpotqa-dev-multikilt_1000_k1000_dep3",
        max_test_samples: int = 100,
        prompt_obj: PromptObject | str = DEFAULT_PROMPT_PATH,
        continue_final_message: bool = False
    ) -> None:
        super().__init__(dataset_path, dataset_split, dataset_name, prompt_obj)

        # HuggingFace `load_dataset` yields examples as dicts. Enable *streaming*
        # to avoid loading the entire split into memory.
        self.dataset_path = dataset_path
        self.dataset_split = dataset_split
        self.dataset_name = dataset_name
        if "kilt" in self.dataset_split:
            full_dataset_path = f'{dataset_path}/{dataset_split}/{dataset_name}.jsonl'
        elif "alce" in self.dataset_split:
            full_dataset_path = f'{dataset_path}/{dataset_split}/{dataset_name}.json'

        data = load_dataset("json", data_files=full_dataset_path)["train"]

        key = "id" if "id" in data.column_names else "question"
        if max_test_samples is not None:
            # some datasets do not have id (e.g., nq), so we assume unique questions
            # TODO: add the 'id' column manually to the NQ dataset for RAG indexing.
            if "kilt" in self.dataset_split:
                keys = set(data[key])
                keys = random.sample(sorted(keys), min(max_test_samples, len(keys)))
                data = data.filter(lambda x: x[key] in keys)
            elif "alce" in self.dataset_split: ## both alce_asqa and alce_qampari
                data = data.shuffle(seed=SEED).select(range(min(max_test_samples, len(data))))
        
        self.dataset = data
        # ------------------------------------------------------------------
        # Prompt resolution & validation
        # ------------------------------------------------------------------
        if isinstance(prompt_obj, str | Path):
            prompt_obj = self._load_prompt_obj(Path(prompt_obj))
            prompt_obj = [p for p in prompt_obj if p["split"] == dataset_split][0]

        self._validate_prompt_obj(prompt_obj)

        self.system_instruction: str = prompt_obj["system_instruction"]  # type: ignore
        self.user_content_template: str = prompt_obj["user_content_template"]  # type: ignore
        self.continue_final_message: bool = continue_final_message

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_prompt_obj(path: Path) -> PromptObject:
        """Load and return the prompt configuration stored at *path*."""
        with path.expanduser().open(encoding="utf-8") as fh:
            return json.load(fh)

    @staticmethod
    def _validate_prompt_obj(prompt_obj: Mapping[str, Any]) -> None:
        """Ensure *prompt_obj* contains the required keys; otherwise raise."""
        required = {"system_instruction", "user_content_template"}
        missing = required - prompt_obj.keys()
        if missing:
            joined = ", ".join(sorted(missing))
            err_msg = f"Prompt object missing required keys: {joined}"
            raise ValueError(err_msg)


    def _format_user_content(self, **data: Any) -> str:
        """Fill the user template with fields extracted from *data*."""
        if "kilt" in self.dataset_split:
            if "popqa" in self.dataset_name:
                positive_psg_ids = [c["id"] for c in data["ctxs"] if c["has_answer"]]
            else:
                positive_psg_ids = [c["psg_id"] for c in data["positive_ctxs"]]
            full_context = ""
            gold_doc_ids = []
            gold_docs = []
            for i, doc in enumerate(data["ctxs"]):
                full_context += f'[DOC {i}]\n{doc["text"]}\n\n'
                if "popqa" in self.dataset_name:
                    if "id" in doc and doc["id"] in positive_psg_ids:
                        gold_doc_ids.append(f'[DOC {i}]')
                        gold_docs.append(doc["text"])
                else:
                    if "psg_id" in doc and doc["psg_id"] in positive_psg_ids:
                        gold_doc_ids.append(f'[DOC {i}]')
                        gold_docs.append(doc["text"])

            try:
                placeholders = {
                    "context": full_context,
                    "question": data["question"],
                }
            except KeyError as exc:  # pragma: no cover
                missing_key = exc.args[0]
                err_msg = f"Dataset row is missing expected field '{missing_key}'."
                raise KeyError(err_msg) from exc

            return gold_doc_ids, gold_docs, full_context, self.user_content_template.format(**placeholders)

        elif "alce" in self.dataset_split:
            instruction = self.user_content_template["instruction"]
            demo_prompt = self.user_content_template["demo_prompt"]
            doc_prompt = self.user_content_template["doc_prompt"]
            demo_sep = self.user_content_template["demo_sep"]
            demos = self.user_content_template["demos"]
            shots = len(demos)
            num_docs = int(self.dataset_name.split('top')[1].split('.')[0])

            user_template = "{demo_text}{instruction}\n\nQuestion: {question}\n\n{context}"

            question = data["question"]
            context = "\n\n".join([doc_prompt.format(**d, ID=idx+1) for idx, d in enumerate(data["docs"][:num_docs])])
            demo_text = demo_sep.join([
                demo_prompt.format(**demo, instruction=instruction, context = "\n\n".join([doc_prompt.format(**d, ID=idx+1) for idx, d in enumerate(demo["docs"])]))
                for demo in random.sample(demos, shots)
            ])
            if shots > 0:
                demo_text += demo_sep
            
            return context, user_template.format(demo_text=demo_text, instruction=instruction, question=question, context=context)


    def _format_continue_final_message(self, **data: Any) -> str:
        assistant_text = "Relevant documents:\n"
        for id, content in zip(data["gold_doc_ids"], data["gold_docs"]):
            assistant_text += f'{id}\n{content}\n\n'
        # assistant_text += "Answer: "
        assistant_text += "\n\n"
        return assistant_text

    # ------------------------------------------------------------------
    # Public API implementation
    # ------------------------------------------------------------------
    @override
    def iterate(self) -> Iterator[DatasetRow]:
        """Yield dataset rows enriched with an OpenAI-style ``messages`` list."""
        for example in self.dataset:
            row_dict: DatasetRow = dict(example)  # type: ignore[arg-type]

            if "kilt" in self.dataset_split:
                gold_doc_ids, gold_docs, full_context, user_content = self._format_user_content(**row_dict)

                row_dict["gold_doc_ids"] = gold_doc_ids ## Added to ease the identification of relevant documents.
                row_dict["gold_docs"] = gold_docs       ## Added to check the text of the relevant documents.
                row_dict["context"] = full_context      ## Added to comply with the other dataloaders.

                row_dict["messages"] = [
                    {"role": "system", "content": self.system_instruction},
                    {"role": "user", "content": user_content},
                ]

                if self.continue_final_message:
                    row_dict["messages"].append({
                        "role": "assistant", "content": self._format_continue_final_message(**row_dict)
                    })
            
            elif "alce" in self.dataset_split:
                full_context, user_content = self._format_user_content(**row_dict)

                row_dict["context"] = full_context      ## Added to comply with the other dataloaders.

                row_dict["messages"] = [
                    {"role": "system", "content": self.system_instruction},
                    {"role": "user", "content": user_content},
                ]

            yield row_dict