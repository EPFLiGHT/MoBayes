"""Closed-world MEDDxAgent RAG over the local DDx knowledge base."""

from __future__ import annotations

import csv
import re
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from ddxdriver.logger import log
from ddxdriver.models import init_model
from ddxdriver.rag_agents.base import RAG
from ddxdriver.rag_agents.utils import (
    extract_and_eval_list,
    get_create_keywords_user_prompt,
    get_rag_synthesis_system_prompt,
    get_rag_synthesis_user_prompt,
)
from ddxdriver.utils import Constants, OutputDict


_NEGATIVE_VALUES = {
    "",
    "0",
    "absent",
    "negative",
    "no",
    "none",
    "normal",
    "unknown",
}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


class ClosedWorldDDxRAG(RAG):
    def __init__(self, rag_agent_cfg):
        super().__init__(rag_agent_cfg=rag_agent_cfg)
        self.model = init_model(
            self.config["model"]["class_name"], **self.config["model"]["config"]
        )
        self.data_dir = Path(self.config["data_dir"]).expanduser()
        self.max_card_findings = int(self.config.get("max_card_findings", 8))
        self.cards = self._build_cards()

    def _build_cards(self) -> List[Dict[str, str]]:
        diseases_path = self.data_dir / "diseases.csv"
        features_path = self.data_dir / "features.csv"
        counts_path = self.data_dir / "likelihood_counts.csv"

        disease_rows = list(csv.DictReader(diseases_path.open()))
        feature_rows = list(csv.DictReader(features_path.open()))
        count_rows = csv.DictReader(counts_path.open())

        all_parents = {row["parent_id"] for row in disease_rows if row.get("parent_id")}
        disease_meta = {
            row["disease_id"]: row
            for row in disease_rows
            if row["disease_id"] not in all_parents and row["disease_id"] != "d_other"
        }
        feature_name_by_id = {row["feature_id"]: row["name"] for row in feature_rows}
        parent_name_by_id = {row["disease_id"]: row["name"] for row in disease_rows}

        totals: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        candidates: Dict[str, List[tuple]] = defaultdict(list)

        for row in count_rows:
            disease_id = row["disease_id"]
            if disease_id not in disease_meta:
                continue

            feature_id = row["feature_id"]
            value = str(row["value"]).strip()
            count = int(row["count"])
            totals[disease_id][feature_id] += count

            value_lower = value.lower()
            if value_lower in _NEGATIVE_VALUES or value.isdigit():
                continue

            feature_name = feature_name_by_id.get(feature_id)
            if not feature_name:
                continue

            candidates[disease_id].append((feature_id, feature_name, value, count))

        cards = []
        for disease_id, meta in disease_meta.items():
            scored = []
            for feature_id, feature_name, value, count in candidates.get(disease_id, []):
                total = totals[disease_id].get(feature_id, 0)
                if total <= 0:
                    continue
                prevalence = count / total
                if prevalence < 0.20:
                    continue
                scored.append((prevalence, feature_name, value))

            scored.sort(key=lambda item: (-item[0], item[1], item[2]))

            findings = []
            seen = set()
            for prevalence, feature_name, value in scored:
                key = (feature_name, value)
                if key in seen:
                    continue
                seen.add(key)
                findings.append(f"- {feature_name} -> {value} ({prevalence:.0%})")
                if len(findings) >= self.max_card_findings:
                    break

            parent_id = meta.get("parent_id") or ""
            category = parent_name_by_id.get(parent_id, "")
            name = meta["name"]
            icd10 = (meta.get("icd10") or "").strip()
            body_lines = [f"Disease: {name}"]
            if category:
                body_lines.append(f"Category: {category}")
            if icd10:
                body_lines.append(f"ICD-10: {icd10}")
            if findings:
                body_lines.append("Common findings in this DDx knowledge base:")
                body_lines.extend(findings)

            content = "\n".join(body_lines)
            cards.append(
                {
                    "title": name,
                    "content": content,
                    "_search_blob": f"{name}\n{category}\n{icd10}\n{content}".lower(),
                }
            )

        return cards

    def _score_card(self, query: str, keywords: List[str], card: Dict[str, str]) -> float:
        blob = card["_search_blob"]
        score = 0.0

        if query and card["title"].lower() in query.lower():
            score += 20.0

        title_tokens = set(_tokenize(card["title"]))
        for keyword in keywords:
            keyword = keyword.strip().lower()
            if not keyword:
                continue
            if keyword in blob:
                score += 5.0
            keyword_tokens = set(_tokenize(keyword))
            score += 1.5 * len(keyword_tokens & title_tokens)

        query_tokens = set(_tokenize(query))
        blob_tokens = set(_tokenize(blob))
        score += len(query_tokens & blob_tokens)
        return score

    def _retrieve_local_results(
        self, query: str, diagnosis_options: List[str]
    ) -> List[Dict[str, str]]:
        keywords = [query] + diagnosis_options[: self.max_keyword_searches]
        scored = []
        for card in self.cards:
            score = self._score_card(query, keywords, card)
            if score > 0:
                scored.append((score, card))

        scored.sort(key=lambda item: (-item[0], item[1]["title"]))
        return [
            {"title": card["title"], "content": card["content"]}
            for _score, card in scored[: self.top_k_search]
        ]

    def synthesize_results(
        self,
        input_search: str,
        search_results: List[Dict[str, str]],
        diagnosis_options: List[str] = [],
    ) -> str:
        if not search_results:
            return ""
        search_results_text = "\n\n".join(
            f"Title: {result['title']}\nContent:\n{result['content']}"
            for result in search_results
        )
        system_prompt = get_rag_synthesis_system_prompt()
        user_prompt = get_rag_synthesis_user_prompt(
            input_search=input_search,
            search_results_text=search_results_text,
            diagnosis_options=diagnosis_options,
        )
        return self.model(user_prompt=user_prompt, system_prompt=system_prompt)

    def __call__(
        self,
        input_search: str,
        diagnosis_options: List[str] = [],
    ) -> str:
        retry_counter = 0
        user_prompt = get_create_keywords_user_prompt(
            input_search=input_search, max_keyword_searches=self.max_keyword_searches
        )
        keyword_searches: List[str] = []
        message_history = []

        while retry_counter <= Constants.RAG_RETRIES.value:
            keyword_searches_str = "[]"
            try:
                keyword_searches_str = self.model(
                    user_prompt=user_prompt, message_history=message_history
                )
                keyword_searches = extract_and_eval_list(keyword_searches_str)
                keyword_searches = keyword_searches[: self.max_keyword_searches]
            except Exception as exc:
                log.info(f"ClosedWorldDDxRAG keyword parsing retry: {exc}\n")
                keyword_searches = []

            if keyword_searches and all(isinstance(item, str) for item in keyword_searches):
                break

            retry_counter += 1
            message_history.extend(
                [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": keyword_searches_str},
                ]
            )
            user_prompt = (
                "Your keyword list was not formatted correctly. "
                'Return only a valid Python-style list of short strings, e.g. ["edema", "heart failure edema"].'
            )

        if not keyword_searches:
            keyword_searches = [input_search]

        search_results: List[Dict[str, str]] = []
        for keyword_search in keyword_searches:
            try:
                results = self._retrieve_local_results(keyword_search, diagnosis_options)
                if results:
                    search_results.extend(results)
            except Exception as exc:
                tb = traceback.format_exc()
                log.error(f"ClosedWorldDDxRAG retrieval failed:\n{exc}\nTraceback:\n{tb}")

        if not search_results:
            log.info("ClosedWorldDDxRAG found no local disease cards.\n")
            return ""

        deduped = []
        seen_titles = set()
        for result in search_results:
            title = result["title"]
            if title in seen_titles:
                continue
            seen_titles.add(title)
            deduped.append(result)
            if len(deduped) >= self.top_k_search:
                break

        rag_content = self.synthesize_results(
            input_search=input_search,
            search_results=deduped,
            diagnosis_options=diagnosis_options,
        )
        log.info("Closed-world rag content\n" + rag_content + "\n")
        return {OutputDict.RAG_CONTENT: rag_content}
