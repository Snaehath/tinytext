"""New 120-Prompt Evaluation Suite for TinyText Flash V1.3 Refinement Study.

Focused exclusively on tasks where refinement could potentially help:
1. Exact-Count Lists (30 prompts: 15 Dev, 15 Holdout)
2. Rewriting Without Adding Claims (30 prompts: 15 Dev, 15 Holdout)
3. Concise Classification & Structured Responses (30 prompts: 15 Dev, 15 Holdout)
4. Short Factual Explanations (30 prompts: 15 Dev, 15 Holdout)

Total: 120 balanced prompts (60 Dev, 60 Holdout).
Each prompt includes programmatic verification criteria and explicit rubrics.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List


REFINEMENT_EVAL_SUITE_120: List[Dict[str, Any]] = [
    # =========================================================================
    # 1. Exact-Count Lists (30 prompts: 15 Dev, 15 Holdout)
    # =========================================================================
    # --- Dev (15) ---
    {
        "id": "list_dev_01",
        "category": "Exact-Count Lists",
        "subset": "dev",
        "prompt": "List exactly three citrus fruits. Output only a numbered list.",
        "target_count": 3,
        "type": "numbered_list",
    },
    {
        "id": "list_dev_02",
        "category": "Exact-Count Lists",
        "subset": "dev",
        "prompt": "Name exactly two types of precipitation. Output only a numbered list.",
        "target_count": 2,
        "type": "numbered_list",
    },
    {
        "id": "list_dev_03",
        "category": "Exact-Count Lists",
        "subset": "dev",
        "prompt": "Provide exactly four continents. Output only a numbered list.",
        "target_count": 4,
        "type": "numbered_list",
    },
    {
        "id": "list_dev_04",
        "category": "Exact-Count Lists",
        "subset": "dev",
        "prompt": "List exactly two primary colors in art. Output only a numbered list.",
        "target_count": 2,
        "type": "numbered_list",
    },
    {
        "id": "list_dev_05",
        "category": "Exact-Count Lists",
        "subset": "dev",
        "prompt": "Give exactly three warm-blooded animals. Output only a numbered list.",
        "target_count": 3,
        "type": "numbered_list",
    },
    {
        "id": "list_dev_06",
        "category": "Exact-Count Lists",
        "subset": "dev",
        "prompt": "Name exactly two string instruments. Output only a numbered list.",
        "target_count": 2,
        "type": "numbered_list",
    },
    {
        "id": "list_dev_07",
        "category": "Exact-Count Lists",
        "subset": "dev",
        "prompt": "Provide exactly three countries in South America. Output only a numbered list.",
        "target_count": 3,
        "type": "numbered_list",
    },
    {
        "id": "list_dev_08",
        "category": "Exact-Count Lists",
        "subset": "dev",
        "prompt": "List exactly two root vegetables. Output only a numbered list.",
        "target_count": 2,
        "type": "numbered_list",
    },
    {
        "id": "list_dev_09",
        "category": "Exact-Count Lists",
        "subset": "dev",
        "prompt": "Give exactly four states of matter in physics. Output only a numbered list.",
        "target_count": 4,
        "type": "numbered_list",
    },
    {
        "id": "list_dev_10",
        "category": "Exact-Count Lists",
        "subset": "dev",
        "prompt": "Name exactly three operating systems for computers. Output only a numbered list.",
        "target_count": 3,
        "type": "numbered_list",
    },
    {
        "id": "list_dev_11",
        "category": "Exact-Count Lists",
        "subset": "dev",
        "prompt": "List exactly two ocean names. Output only a numbered list.",
        "target_count": 2,
        "type": "numbered_list",
    },
    {
        "id": "list_dev_12",
        "category": "Exact-Count Lists",
        "subset": "dev",
        "prompt": "Provide exactly three items used for writing by hand. Output only a numbered list.",
        "target_count": 3,
        "type": "numbered_list",
    },
    {
        "id": "list_dev_13",
        "category": "Exact-Count Lists",
        "subset": "dev",
        "prompt": "Name exactly two culinary spices commonly used in baking. Output only a numbered list.",
        "target_count": 2,
        "type": "numbered_list",
    },
    {
        "id": "list_dev_14",
        "category": "Exact-Count Lists",
        "subset": "dev",
        "prompt": "List exactly four cardinal directions on a compass. Output only a numbered list.",
        "target_count": 4,
        "type": "numbered_list",
    },
    {
        "id": "list_dev_15",
        "category": "Exact-Count Lists",
        "subset": "dev",
        "prompt": "Give exactly three metals found on Earth. Output only a numbered list.",
        "target_count": 3,
        "type": "numbered_list",
    },
    # --- Holdout (15) ---
    {
        "id": "list_holdout_01",
        "category": "Exact-Count Lists",
        "subset": "holdout",
        "prompt": "List exactly three berries. Output only a numbered list.",
        "target_count": 3,
        "type": "numbered_list",
    },
    {
        "id": "list_holdout_02",
        "category": "Exact-Count Lists",
        "subset": "holdout",
        "prompt": "Name exactly two internal human organs. Output only a numbered list.",
        "target_count": 2,
        "type": "numbered_list",
    },
    {
        "id": "list_holdout_03",
        "category": "Exact-Count Lists",
        "subset": "holdout",
        "prompt": "Provide exactly four major world currencies. Output only a numbered list.",
        "target_count": 4,
        "type": "numbered_list",
    },
    {
        "id": "list_holdout_04",
        "category": "Exact-Count Lists",
        "subset": "holdout",
        "prompt": "Give exactly three nocturnal animals. Output only a numbered list.",
        "target_count": 3,
        "type": "numbered_list",
    },
    {
        "id": "list_holdout_05",
        "category": "Exact-Count Lists",
        "subset": "holdout",
        "prompt": "List exactly two branches of a national government. Output only a numbered list.",
        "target_count": 2,
        "type": "numbered_list",
    },
    {
        "id": "list_holdout_06",
        "category": "Exact-Count Lists",
        "subset": "holdout",
        "prompt": "Name exactly three green vegetables. Output only a numbered list.",
        "target_count": 3,
        "type": "numbered_list",
    },
    {
        "id": "list_holdout_07",
        "category": "Exact-Count Lists",
        "subset": "holdout",
        "prompt": "Provide exactly four winter Olympic sports. Output only a numbered list.",
        "target_count": 4,
        "type": "numbered_list",
    },
    {
        "id": "list_holdout_08",
        "category": "Exact-Count Lists",
        "subset": "holdout",
        "prompt": "List exactly two common indoor house plants. Output only a numbered list.",
        "target_count": 2,
        "type": "numbered_list",
    },
    {
        "id": "list_holdout_09",
        "category": "Exact-Count Lists",
        "subset": "holdout",
        "prompt": "Give exactly three natural disasters caused by weather. Output only a numbered list.",
        "target_count": 3,
        "type": "numbered_list",
    },
    {
        "id": "list_holdout_10",
        "category": "Exact-Count Lists",
        "subset": "holdout",
        "prompt": "Name exactly two modes of air transportation. Output only a numbered list.",
        "target_count": 2,
        "type": "numbered_list",
    },
    {
        "id": "list_holdout_11",
        "category": "Exact-Count Lists",
        "subset": "holdout",
        "prompt": "List exactly three European capital cities. Output only a numbered list.",
        "target_count": 3,
        "type": "numbered_list",
    },
    {
        "id": "list_holdout_12",
        "category": "Exact-Count Lists",
        "subset": "holdout",
        "prompt": "Provide exactly four ingredients in a classic salad. Output only a numbered list.",
        "target_count": 4,
        "type": "numbered_list",
    },
    {
        "id": "list_holdout_13",
        "category": "Exact-Count Lists",
        "subset": "holdout",
        "prompt": "Name exactly two types of renewable clean energy. Output only a numbered list.",
        "target_count": 2,
        "type": "numbered_list",
    },
    {
        "id": "list_holdout_14",
        "category": "Exact-Count Lists",
        "subset": "holdout",
        "prompt": "Give exactly three common pet dog breeds. Output only a numbered list.",
        "target_count": 3,
        "type": "numbered_list",
    },
    {
        "id": "list_holdout_15",
        "category": "Exact-Count Lists",
        "subset": "holdout",
        "prompt": "List exactly two seasons of the year. Output only a numbered list.",
        "target_count": 2,
        "type": "numbered_list",
    },

    # =========================================================================
    # 2. Rewriting Without Adding Claims (30 prompts: 15 Dev, 15 Holdout)
    # =========================================================================
    # --- Dev (15) ---
    {
        "id": "rewrite_dev_01",
        "category": "Rewriting Without Added Claims",
        "subset": "dev",
        "prompt": "Rewrite this professionally: We gotta push back the release date 'cause the build is broken.",
        "required_keywords": ["delay", "postpone", "release", "build", "schedule"],
        "forbidden_phrases": ["fired", "disaster", "I am an AI"],
        "type": "rewrite",
    },
    {
        "id": "rewrite_dev_02",
        "category": "Rewriting Without Added Claims",
        "subset": "dev",
        "prompt": "Rewrite in the active voice: The quarterly audit was completed by Elena.",
        "required_keywords": ["Elena completed the quarterly audit"],
        "forbidden_phrases": ["was completed by"],
        "type": "rewrite",
    },
    {
        "id": "rewrite_dev_03",
        "category": "Rewriting Without Added Claims",
        "subset": "dev",
        "prompt": "Make this sentence concise: In light of the fact that the weather was stormy, they postponed the match.",
        "required_keywords": ["storm", "postponed", "match"],
        "forbidden_phrases": ["raincoat", "lightning struck"],
        "type": "rewrite",
    },
    {
        "id": "rewrite_dev_04",
        "category": "Rewriting Without Added Claims",
        "subset": "dev",
        "prompt": "Rewrite without negative words: Do not forget to lock the laboratory door.",
        "required_keywords": ["remember", "lock", "laboratory", "door"],
        "forbidden_phrases": ["not", "never", "don't"],
        "type": "rewrite",
    },
    {
        "id": "rewrite_dev_05",
        "category": "Rewriting Without Added Claims",
        "subset": "dev",
        "prompt": "Rewrite politely: I need your feedback on the document right now.",
        "required_keywords": ["please", "feedback", "document", "could you"],
        "forbidden_phrases": ["threat", "immediately or else"],
        "type": "rewrite",
    },
    {
        "id": "rewrite_dev_06",
        "category": "Rewriting Without Added Claims",
        "subset": "dev",
        "prompt": "Fix grammatical errors in: He don't know where the keys is.",
        "required_keywords": ["does not know", "doesn't know", "keys are"],
        "forbidden_phrases": ["lost forever"],
        "type": "rewrite",
    },
    {
        "id": "rewrite_dev_07",
        "category": "Rewriting Without Added Claims",
        "subset": "dev",
        "prompt": "Rewrite in the past tense: The bell rings at noon every weekday.",
        "required_keywords": ["rang"],
        "forbidden_phrases": ["church", "clock broke"],
        "type": "rewrite",
    },
    {
        "id": "rewrite_dev_08",
        "category": "Rewriting Without Added Claims",
        "subset": "dev",
        "prompt": "Simplify for clarity: Subsequent to the conclusion of the seminar, refreshments will be served.",
        "required_keywords": ["after", "seminar", "refreshments"],
        "forbidden_phrases": ["cookies", "wine"],
        "type": "rewrite",
    },
    {
        "id": "rewrite_dev_09",
        "category": "Rewriting Without Added Claims",
        "subset": "dev",
        "prompt": "Rewrite to sound confident: I guess we might perhaps meet the project milestone.",
        "required_keywords": ["will meet", "confident", "milestone", "project"],
        "forbidden_phrases": ["fail", "maybe tomorrow"],
        "type": "rewrite",
    },
    {
        "id": "rewrite_dev_10",
        "category": "Rewriting Without Added Claims",
        "subset": "dev",
        "prompt": "Change to passive voice: Alexander Graham Bell patented the telephone.",
        "required_keywords": ["telephone was patented by Alexander Graham Bell"],
        "forbidden_phrases": ["smartphone", "Edison"],
        "type": "rewrite",
    },
    {
        "id": "rewrite_dev_11",
        "category": "Rewriting Without Added Claims",
        "subset": "dev",
        "prompt": "Rewrite into a single polite sentence: You missed the signature. Sign the form.",
        "required_keywords": ["please", "sign", "form"],
        "forbidden_phrases": ["incompetent", "careless"],
        "type": "rewrite",
    },
    {
        "id": "rewrite_dev_12",
        "category": "Rewriting Without Added Claims",
        "subset": "dev",
        "prompt": "Rewrite concisely: At the current point in time, the library remains open.",
        "required_keywords": ["library is open", "currently", "now"],
        "forbidden_phrases": ["books", "librarian"],
        "type": "rewrite",
    },
    {
        "id": "rewrite_dev_13",
        "category": "Rewriting Without Added Claims",
        "subset": "dev",
        "prompt": "Rewrite casually: We must immediately commence the scheduled conversation.",
        "required_keywords": ["let's", "start", "talk", "chat"],
        "forbidden_phrases": ["board of directors"],
        "type": "rewrite",
    },
    {
        "id": "rewrite_dev_14",
        "category": "Rewriting Without Added Claims",
        "subset": "dev",
        "prompt": "Rewrite without negative words: Do not fail to submit your timesheet.",
        "required_keywords": ["submit", "timesheet", "ensure", "remember"],
        "forbidden_phrases": ["fail", "not", "penalty"],
        "type": "rewrite",
    },
    {
        "id": "rewrite_dev_15",
        "category": "Rewriting Without Added Claims",
        "subset": "dev",
        "prompt": "Fix grammatical errors in: Each of the students have finished their exams.",
        "required_keywords": ["has finished"],
        "forbidden_phrases": ["failed"],
        "type": "rewrite",
    },
    # --- Holdout (15) ---
    {
        "id": "rewrite_holdout_01",
        "category": "Rewriting Without Added Claims",
        "subset": "holdout",
        "prompt": "Rewrite professionally: Tell me how much this stuff costs before I buy it.",
        "required_keywords": ["pricing", "cost", "quote", "provide", "purchase"],
        "forbidden_phrases": ["too expensive", "cheap"],
        "type": "rewrite",
    },
    {
        "id": "rewrite_holdout_02",
        "category": "Rewriting Without Added Claims",
        "subset": "holdout",
        "prompt": "Rewrite in the active voice: The award was received by Dr. Vance.",
        "required_keywords": ["Dr. Vance received the award"],
        "forbidden_phrases": ["was received by"],
        "type": "rewrite",
    },
    {
        "id": "rewrite_holdout_03",
        "category": "Rewriting Without Added Claims",
        "subset": "holdout",
        "prompt": "Make this sentence concise: Due to the circumstance that power was lost, the workshop ended early.",
        "required_keywords": ["power outage", "power was lost", "ended early"],
        "forbidden_phrases": ["generator", "darkness"],
        "type": "rewrite",
    },
    {
        "id": "rewrite_holdout_04",
        "category": "Rewriting Without Added Claims",
        "subset": "holdout",
        "prompt": "Rewrite without negative words: Never leave confidential records unattended.",
        "required_keywords": ["secure", "confidential records", "protect", "keep"],
        "forbidden_phrases": ["never", "not", "don't"],
        "type": "rewrite",
    },
    {
        "id": "rewrite_holdout_05",
        "category": "Rewriting Without Added Claims",
        "subset": "holdout",
        "prompt": "Rewrite politely: Resend the spreadsheet with the missing columns.",
        "required_keywords": ["please", "could you", "spreadsheet", "resend"],
        "forbidden_phrases": ["careless error"],
        "type": "rewrite",
    },
    {
        "id": "rewrite_holdout_06",
        "category": "Rewriting Without Added Claims",
        "subset": "holdout",
        "prompt": "Fix grammatical errors in: The team are working good together.",
        "required_keywords": ["is working well", "are working well"],
        "forbidden_phrases": ["badly"],
        "type": "rewrite",
    },
    {
        "id": "rewrite_holdout_07",
        "category": "Rewriting Without Added Claims",
        "subset": "holdout",
        "prompt": "Rewrite in the past tense: The athlete runs five miles before breakfast.",
        "required_keywords": ["ran"],
        "forbidden_phrases": ["marathon"],
        "type": "rewrite",
    },
    {
        "id": "rewrite_holdout_08",
        "category": "Rewriting Without Added Claims",
        "subset": "holdout",
        "prompt": "Simplify for clarity: It is imperative that all personnel vacate the premises immediately.",
        "required_keywords": ["please leave", "evacuate", "everyone must leave"],
        "forbidden_phrases": ["fire", "bomb"],
        "type": "rewrite",
    },
    {
        "id": "rewrite_holdout_09",
        "category": "Rewriting Without Added Claims",
        "subset": "holdout",
        "prompt": "Rewrite to sound confident: Maybe our team will sort of try to deliver the software.",
        "required_keywords": ["will deliver", "commit", "software", "team"],
        "forbidden_phrases": ["delay", "next year"],
        "type": "rewrite",
    },
    {
        "id": "rewrite_holdout_10",
        "category": "Rewriting Without Added Claims",
        "subset": "holdout",
        "prompt": "Change to passive voice: Johannes Gutenberg invented the printing press.",
        "required_keywords": ["printing press was invented by Johannes Gutenberg"],
        "forbidden_phrases": ["books"],
        "type": "rewrite",
    },
    {
        "id": "rewrite_holdout_11",
        "category": "Rewriting Without Added Claims",
        "subset": "holdout",
        "prompt": "Rewrite into a single polite sentence: You did not attach your resume. Email it to us.",
        "required_keywords": ["please", "send", "resume", "attach"],
        "forbidden_phrases": ["rejected"],
        "type": "rewrite",
    },
    {
        "id": "rewrite_holdout_12",
        "category": "Rewriting Without Added Claims",
        "subset": "holdout",
        "prompt": "Rewrite concisely: For all intents and purposes, the project has reached completion.",
        "required_keywords": ["project is complete", "project has finished", "essentially complete"],
        "forbidden_phrases": ["celebration"],
        "type": "rewrite",
    },
    {
        "id": "rewrite_holdout_13",
        "category": "Rewriting Without Added Claims",
        "subset": "holdout",
        "prompt": "Rewrite warmly: Your submission arrived on time.",
        "required_keywords": ["thank you", "received", "submission", "on time"],
        "forbidden_phrases": ["grade", "score"],
        "type": "rewrite",
    },
    {
        "id": "rewrite_holdout_14",
        "category": "Rewriting Without Added Claims",
        "subset": "holdout",
        "prompt": "Rewrite without negative words: Do not disobey safety protocols in the warehouse.",
        "required_keywords": ["follow", "safety protocols", "comply", "adhere"],
        "forbidden_phrases": ["disobey", "not", "penalty"],
        "type": "rewrite",
    },
    {
        "id": "rewrite_holdout_15",
        "category": "Rewriting Without Added Claims",
        "subset": "holdout",
        "prompt": "Fix grammatical errors in: Neither the teacher nor the students was ready.",
        "required_keywords": ["were ready"],
        "forbidden_phrases": ["homework"],
        "type": "rewrite",
    },

    # =========================================================================
    # 3. Concise Classification & Structured Responses (30 prompts: 15 Dev, 15 Holdout)
    # =========================================================================
    # --- Dev (15) ---
    {
        "id": "class_dev_01",
        "category": "Concise Classification & Structured Output",
        "subset": "dev",
        "prompt": "Classify sentiment as Positive or Negative: The battery drains completely after just two hours. Output only the label.",
        "expected_label": "Negative",
        "type": "classification",
    },
    {
        "id": "class_dev_02",
        "category": "Concise Classification & Structured Output",
        "subset": "dev",
        "prompt": "Classify sentiment as Positive or Negative: The setup was seamless and finished ahead of schedule. Output only the single label.",
        "expected_label": "Positive",
        "type": "classification",
    },
    {
        "id": "class_dev_03",
        "category": "Concise Classification & Structured Output",
        "subset": "dev",
        "prompt": "Classify the priority as High or Low: A typo was found in an internal training footnote. Output only the label.",
        "expected_label": "Low",
        "type": "classification",
    },
    {
        "id": "class_dev_04",
        "category": "Concise Classification & Structured Output",
        "subset": "dev",
        "prompt": "Classify the priority as High or Low: The main billing gateway is dropping transactions worldwide. Output only the label.",
        "expected_label": "High",
        "type": "classification",
    },
    {
        "id": "class_dev_05",
        "category": "Concise Classification & Structured Output",
        "subset": "dev",
        "prompt": "Classify as Fact or Opinion: Water boils at 100 degrees Celsius at sea level. Output only the label.",
        "expected_label": "Fact",
        "type": "classification",
    },
    {
        "id": "class_dev_06",
        "category": "Concise Classification & Structured Output",
        "subset": "dev",
        "prompt": "Classify as Fact or Opinion: Summer is the best season of the year. Output only the label.",
        "expected_label": "Opinion",
        "type": "classification",
    },
    {
        "id": "class_dev_07",
        "category": "Concise Classification & Structured Output",
        "subset": "dev",
        "prompt": "Classify as Animal or Plant: Oak tree. Output only the label.",
        "expected_label": "Plant",
        "type": "classification",
    },
    {
        "id": "class_dev_08",
        "category": "Concise Classification & Structured Output",
        "subset": "dev",
        "prompt": "Classify as Animal or Plant: Bengal tiger. Output only the label.",
        "expected_label": "Animal",
        "type": "classification",
    },
    {
        "id": "class_dev_09",
        "category": "Concise Classification & Structured Output",
        "subset": "dev",
        "prompt": "Classify temperature condition as Hot or Cold: Freshly fallen snow on a December morning. Output only the label.",
        "expected_label": "Cold",
        "type": "classification",
    },
    {
        "id": "class_dev_10",
        "category": "Concise Classification & Structured Output",
        "subset": "dev",
        "prompt": "Classify temperature condition as Hot or Cold: Boiling tea freshly poured into a ceramic mug. Output only the label.",
        "expected_label": "Hot",
        "type": "classification",
    },
    {
        "id": "class_dev_11",
        "category": "Concise Classification & Structured Output",
        "subset": "dev",
        "prompt": "Classify the environment as Indoor or Outdoor: A living room sofa. Output only the label.",
        "expected_label": "Indoor",
        "type": "classification",
    },
    {
        "id": "class_dev_12",
        "category": "Concise Classification & Structured Output",
        "subset": "dev",
        "prompt": "Classify the environment as Indoor or Outdoor: Hiking along a mountain forest ridge. Output only the label.",
        "expected_label": "Outdoor",
        "type": "classification",
    },
    {
        "id": "class_dev_13",
        "category": "Concise Classification & Structured Output",
        "subset": "dev",
        "prompt": "Classify the urgency as Urgent or Non-Urgent: Routine quarterly software update scheduled for next month. Output only the label.",
        "expected_label": "Non-Urgent",
        "type": "classification",
    },
    {
        "id": "class_dev_14",
        "category": "Concise Classification & Structured Output",
        "subset": "dev",
        "prompt": "Classify the urgency as Urgent or Non-Urgent: A broken water pipe is actively flooding the basement server room. Output only the label.",
        "expected_label": "Urgent",
        "type": "classification",
    },
    {
        "id": "class_dev_15",
        "category": "Concise Classification & Structured Output",
        "subset": "dev",
        "prompt": "Classify as Renewable or Non-Renewable: Solar energy captured by photovoltaic panels. Output only the label.",
        "expected_label": "Renewable",
        "type": "classification",
    },
    # --- Holdout (15) ---
    {
        "id": "class_holdout_01",
        "category": "Concise Classification & Structured Output",
        "subset": "holdout",
        "prompt": "Classify sentiment as Positive or Negative: The package arrived damaged and three items were missing. Output only the label.",
        "expected_label": "Negative",
        "type": "classification",
    },
    {
        "id": "class_holdout_02",
        "category": "Concise Classification & Structured Output",
        "subset": "holdout",
        "prompt": "Classify sentiment as Positive or Negative: The customer support agent resolved my issue within minutes. Output only the single label.",
        "expected_label": "Positive",
        "type": "classification",
    },
    {
        "id": "class_holdout_03",
        "category": "Concise Classification & Structured Output",
        "subset": "holdout",
        "prompt": "Classify the priority as High or Low: Changing the background color of an optional developer documentation banner. Output only the label.",
        "expected_label": "Low",
        "type": "classification",
    },
    {
        "id": "class_holdout_04",
        "category": "Concise Classification & Structured Output",
        "subset": "holdout",
        "prompt": "Classify the priority as High or Low: Core customer authentication is completely down and users cannot log in. Output only the label.",
        "expected_label": "High",
        "type": "classification",
    },
    {
        "id": "class_holdout_05",
        "category": "Concise Classification & Structured Output",
        "subset": "holdout",
        "prompt": "Classify as Fact or Opinion: The Earth orbits the Sun once approximately every 365.25 days. Output only the label.",
        "expected_label": "Fact",
        "type": "classification",
    },
    {
        "id": "class_holdout_06",
        "category": "Concise Classification & Structured Output",
        "subset": "holdout",
        "prompt": "Classify as Fact or Opinion: Classical music is more enjoyable than rock music. Output only the label.",
        "expected_label": "Opinion",
        "type": "classification",
    },
    {
        "id": "class_holdout_07",
        "category": "Concise Classification & Structured Output",
        "subset": "holdout",
        "prompt": "Classify as Animal or Plant: Sunflower. Output only the label.",
        "expected_label": "Plant",
        "type": "classification",
    },
    {
        "id": "class_holdout_08",
        "category": "Concise Classification & Structured Output",
        "subset": "holdout",
        "prompt": "Classify as Animal or Plant: Bald eagle. Output only the label.",
        "expected_label": "Animal",
        "type": "classification",
    },
    {
        "id": "class_holdout_09",
        "category": "Concise Classification & Structured Output",
        "subset": "holdout",
        "prompt": "Classify temperature condition as Hot or Cold: An iced lemonade served with crushed cubes in July. Output only the label.",
        "expected_label": "Cold",
        "type": "classification",
    },
    {
        "id": "class_holdout_10",
        "category": "Concise Classification & Structured Output",
        "subset": "holdout",
        "prompt": "Classify temperature condition as Hot or Cold: Molten lava flowing from an active volcanic vent. Output only the label.",
        "expected_label": "Hot",
        "type": "classification",
    },
    {
        "id": "class_holdout_11",
        "category": "Concise Classification & Structured Output",
        "subset": "holdout",
        "prompt": "Classify the environment as Indoor or Outdoor: A bedroom nightstand. Output only the label.",
        "expected_label": "Indoor",
        "type": "classification",
    },
    {
        "id": "class_holdout_12",
        "category": "Concise Classification & Structured Output",
        "subset": "holdout",
        "prompt": "Classify the environment as Indoor or Outdoor: A coral reef in the Caribbean Sea. Output only the label.",
        "expected_label": "Outdoor",
        "type": "classification",
    },
    {
        "id": "class_holdout_13",
        "category": "Concise Classification & Structured Output",
        "subset": "holdout",
        "prompt": "Classify the urgency as Urgent or Non-Urgent: A critical security breach leaking live production database passwords. Output only the label.",
        "expected_label": "Urgent",
        "type": "classification",
    },
    {
        "id": "class_holdout_14",
        "category": "Concise Classification & Structured Output",
        "subset": "holdout",
        "prompt": "Classify the urgency as Urgent or Non-Urgent: Requesting an additional monitor stand for an employee arriving in three weeks. Output only the label.",
        "expected_label": "Non-Urgent",
        "type": "classification",
    },
    {
        "id": "class_holdout_15",
        "category": "Concise Classification & Structured Output",
        "subset": "holdout",
        "prompt": "Classify as Renewable or Non-Renewable: Coal extracted from an underground mine. Output only the label.",
        "expected_label": "Non-Renewable",
        "type": "classification",
    },

    # =========================================================================
    # 4. Short Factual Explanations (30 prompts: 15 Dev, 15 Holdout)
    # =========================================================================
    # --- Dev (15) ---
    {
        "id": "explain_dev_01",
        "category": "Short Factual Explanations",
        "subset": "dev",
        "prompt": "Explain why leaves change color in autumn in one simple sentence.",
        "keywords": ["chlorophyll", "green", "pigment", "sunlight", "break down"],
        "max_sentences": 1,
        "type": "concise_explanation",
    },
    {
        "id": "explain_dev_02",
        "category": "Short Factual Explanations",
        "subset": "dev",
        "prompt": "Explain what causes day and night on Earth in one sentence.",
        "keywords": ["rotation", "rotates", "axis", "spins", "sun"],
        "max_sentences": 1,
        "type": "concise_explanation",
    },
    {
        "id": "explain_dev_03",
        "category": "Short Factual Explanations",
        "subset": "dev",
        "prompt": "Explain how clouds form in one or two simple sentences.",
        "keywords": ["water vapor", "condenses", "cools", "droplets", "rises"],
        "max_sentences": 2,
        "type": "concise_explanation",
    },
    {
        "id": "explain_dev_04",
        "category": "Short Factual Explanations",
        "subset": "dev",
        "prompt": "Explain why the ocean appears blue in one concise sentence.",
        "keywords": ["absorbs", "red", "scatters", "reflects", "blue", "light"],
        "max_sentences": 1,
        "type": "concise_explanation",
    },
    {
        "id": "explain_dev_05",
        "category": "Short Factual Explanations",
        "subset": "dev",
        "prompt": "Explain what causes an earthquake in one simple sentence.",
        "keywords": ["tectonic plates", "fault", "movement", "energy", "crust"],
        "max_sentences": 1,
        "type": "concise_explanation",
    },
    {
        "id": "explain_dev_06",
        "category": "Short Factual Explanations",
        "subset": "dev",
        "prompt": "Explain how honeybees make honey in one or two sentences.",
        "keywords": ["nectar", "evaporate", "enzymes", "hive", "flowers"],
        "max_sentences": 2,
        "type": "concise_explanation",
    },
    {
        "id": "explain_dev_07",
        "category": "Short Factual Explanations",
        "subset": "dev",
        "prompt": "Explain the role of the heart in the human body in one sentence.",
        "keywords": ["pumps", "blood", "oxygen", "circulates", "body"],
        "max_sentences": 1,
        "type": "concise_explanation",
    },
    {
        "id": "explain_dev_08",
        "category": "Short Factual Explanations",
        "subset": "dev",
        "prompt": "Explain why ice floats on liquid water in one simple sentence.",
        "keywords": ["less dense", "density", "expands", "molecules"],
        "max_sentences": 1,
        "type": "concise_explanation",
    },
    {
        "id": "explain_dev_09",
        "category": "Short Factual Explanations",
        "subset": "dev",
        "prompt": "Explain what photosynthesis is in one concise sentence.",
        "keywords": ["plants", "sunlight", "food", "energy", "glucose", "carbon dioxide"],
        "max_sentences": 1,
        "type": "concise_explanation",
    },
    {
        "id": "explain_dev_10",
        "category": "Short Factual Explanations",
        "subset": "dev",
        "prompt": "Explain how a rainbow forms in one or two simple sentences.",
        "keywords": ["refraction", "water droplets", "sunlight", "scatters", "prism"],
        "max_sentences": 2,
        "type": "concise_explanation",
    },
    {
        "id": "explain_dev_11",
        "category": "Short Factual Explanations",
        "subset": "dev",
        "prompt": "Explain what causes thunder in one sentence.",
        "keywords": ["lightning", "heats", "air", "expands", "shockwave", "sound"],
        "max_sentences": 1,
        "type": "concise_explanation",
    },
    {
        "id": "explain_dev_12",
        "category": "Short Factual Explanations",
        "subset": "dev",
        "prompt": "Explain why exercise benefits human health in one sentence.",
        "keywords": ["strengthens", "cardiovascular", "heart", "muscles", "health", "fitness"],
        "max_sentences": 1,
        "type": "concise_explanation",
    },
    {
        "id": "explain_dev_13",
        "category": "Short Factual Explanations",
        "subset": "dev",
        "prompt": "Explain what an antibody does in the human immune system in one concise sentence.",
        "keywords": ["binds", "recognizes", "pathogens", "neutralizes", "antigen", "immune"],
        "max_sentences": 1,
        "type": "concise_explanation",
    },
    {
        "id": "explain_dev_14",
        "category": "Short Factual Explanations",
        "subset": "dev",
        "prompt": "Explain how sound travels through air in one sentence.",
        "keywords": ["vibrations", "waves", "particles", "molecules"],
        "max_sentences": 1,
        "type": "concise_explanation",
    },
    {
        "id": "explain_dev_15",
        "category": "Short Factual Explanations",
        "subset": "dev",
        "prompt": "Explain what the ozone layer does in one sentence.",
        "keywords": ["absorbs", "protects", "ultraviolet", "uv radiation", "sun"],
        "max_sentences": 1,
        "type": "concise_explanation",
    },
    # --- Holdout (15) ---
    {
        "id": "explain_holdout_01",
        "category": "Short Factual Explanations",
        "subset": "holdout",
        "prompt": "Explain what causes ocean tides in one simple sentence.",
        "keywords": ["gravitational", "moon", "gravity", "pull"],
        "max_sentences": 1,
        "type": "concise_explanation",
    },
    {
        "id": "explain_holdout_02",
        "category": "Short Factual Explanations",
        "subset": "holdout",
        "prompt": "Explain how birds fly in one or two sentences.",
        "keywords": ["wings", "lift", "air", "airfoil", "pressure"],
        "max_sentences": 2,
        "type": "concise_explanation",
    },
    {
        "id": "explain_holdout_03",
        "category": "Short Factual Explanations",
        "subset": "holdout",
        "prompt": "Explain what an ecosystem is in one sentence.",
        "keywords": ["community", "organisms", "environment", "living", "interact"],
        "max_sentences": 1,
        "type": "concise_explanation",
    },
    {
        "id": "explain_holdout_04",
        "category": "Short Factual Explanations",
        "subset": "holdout",
        "prompt": "Explain why metals conduct electricity well in one sentence.",
        "keywords": ["free electrons", "electrons", "move", "flow"],
        "max_sentences": 1,
        "type": "concise_explanation",
    },
    {
        "id": "explain_holdout_05",
        "category": "Short Factual Explanations",
        "subset": "holdout",
        "prompt": "Explain what causes a solar eclipse in one sentence.",
        "keywords": ["moon", "passes between", "blocks", "sun", "earth", "shadow"],
        "max_sentences": 1,
        "type": "concise_explanation",
    },
    {
        "id": "explain_holdout_06",
        "category": "Short Factual Explanations",
        "subset": "holdout",
        "prompt": "Explain how vaccines protect against diseases in one or two sentences.",
        "keywords": ["immune system", "antibodies", "pathogen", "recognize", "infection"],
        "max_sentences": 2,
        "type": "concise_explanation",
    },
    {
        "id": "explain_holdout_07",
        "category": "Short Factual Explanations",
        "subset": "holdout",
        "prompt": "Explain why planets have circular or elliptical orbits in one sentence.",
        "keywords": ["gravity", "sun", "inertia", "velocity", "orbit"],
        "max_sentences": 1,
        "type": "concise_explanation",
    },
    {
        "id": "explain_holdout_08",
        "category": "Short Factual Explanations",
        "subset": "holdout",
        "prompt": "Explain how thermal insulators work in one sentence.",
        "keywords": ["slow", "heat transfer", "conduct", "barrier", "reduce"],
        "max_sentences": 1,
        "type": "concise_explanation",
    },
    {
        "id": "explain_holdout_09",
        "category": "Short Factual Explanations",
        "subset": "holdout",
        "prompt": "Explain the purpose of the lungs in the human body in one sentence.",
        "keywords": ["oxygen", "carbon dioxide", "exchange", "breathe", "blood"],
        "max_sentences": 1,
        "type": "concise_explanation",
    },
    {
        "id": "explain_holdout_10",
        "category": "Short Factual Explanations",
        "subset": "holdout",
        "prompt": "Explain what causes wind to blow across Earth in one sentence.",
        "keywords": ["uneven heating", "pressure differences", "air moves", "temperature"],
        "max_sentences": 1,
        "type": "concise_explanation",
    },
    {
        "id": "explain_holdout_11",
        "category": "Short Factual Explanations",
        "subset": "holdout",
        "prompt": "Explain what a black hole is in one concise sentence.",
        "keywords": ["gravity", "dense", "light cannot escape", "spacetime"],
        "max_sentences": 1,
        "type": "concise_explanation",
    },
    {
        "id": "explain_holdout_12",
        "category": "Short Factual Explanations",
        "subset": "holdout",
        "prompt": "Explain why soap cleans oily hands in one or two sentences.",
        "keywords": ["molecules", "hydrophobic", "hydrophilic", "binds oil", "water", "rinses"],
        "max_sentences": 2,
        "type": "concise_explanation",
    },
    {
        "id": "explain_holdout_13",
        "category": "Short Factual Explanations",
        "subset": "holdout",
        "prompt": "Explain why deserts get very cold at night in one sentence.",
        "keywords": ["dry air", "humidity", "moisture", "heat escapes", "radiates"],
        "max_sentences": 1,
        "type": "concise_explanation",
    },
    {
        "id": "explain_holdout_14",
        "category": "Short Factual Explanations",
        "subset": "holdout",
        "prompt": "Explain what friction is in one concise sentence.",
        "keywords": ["resistance", "motion", "surfaces", "contact", "rubbing"],
        "max_sentences": 1,
        "type": "concise_explanation",
    },
    {
        "id": "explain_holdout_15",
        "category": "Short Factual Explanations",
        "subset": "holdout",
        "prompt": "Explain why trees produce oxygen in one sentence.",
        "keywords": ["byproduct", "photosynthesis", "sunlight", "plants", "carbon dioxide"],
        "max_sentences": 1,
        "type": "concise_explanation",
    },
]


def score_refinement_response(item: Dict[str, Any], text: str) -> Dict[str, Any]:
    """Scores a response against the item's programmatic rubric."""
    clean_text = text.strip()
    item_type = item.get("type", "unknown")

    result: Dict[str, Any] = {
        "passed": False,
        "constraint_satisfied": False,
        "has_repetition": False,
        "has_contradiction": False,
        "detail": "",
    }

    # Repetition detector (trigram loops)
    words = clean_text.lower().split()
    if len(words) >= 9:
        trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
        for tg in set(trigrams):
            if trigrams.count(tg) >= 3:
                result["has_repetition"] = True
                result["detail"] = f"Excessive repetition loop detected for trigram: '{tg}'"
                return result

    if item_type == "numbered_list":
        target_count = item["target_count"]
        list_items = re.findall(r"(?:^|\n)\s*(\d+)[\.\)]\s*(.+)", clean_text)
        if list_items:
            found_count = len(list_items)
            result["constraint_satisfied"] = (found_count == target_count)
            item_texts = [it[1].strip().lower() for it in list_items]
            unique_items = len(set(item_texts)) == len(item_texts)
            # Rejection of dummy placeholder repetitions
            from tinytext.orchestrator import TaskVerifier
            has_placeholder = any(it in TaskVerifier.PLACEHOLDER_ITEMS for it in item_texts)

            if result["constraint_satisfied"] and unique_items and not has_placeholder and not result["has_repetition"]:
                result["passed"] = True
                result["detail"] = f"Matched exact list count of {target_count} with unique items"
            else:
                result["detail"] = f"Found {found_count} items (expected {target_count}), unique={unique_items}, placeholder={has_placeholder}"
        else:
            result["detail"] = "No numbered list format detected"

    elif item_type == "rewrite":
        req_kw = item.get("required_keywords", [])
        forb_phrases = item.get("forbidden_phrases", [])
        has_forbidden = any(f.lower() in clean_text.lower() for f in forb_phrases)
        has_any_req = any(k.lower() in clean_text.lower() for k in req_kw)

        result["constraint_satisfied"] = (not has_forbidden) and (len(clean_text) < 180)
        if has_any_req and result["constraint_satisfied"] and not result["has_repetition"]:
            result["passed"] = True
            result["detail"] = "Rewrite complied with constraints and preserved context"
        else:
            result["detail"] = f"Forbidden: {has_forbidden}, Req keywords matched: {has_any_req}"

    elif item_type == "classification":
        expected = item.get("expected_label", "").lower()
        tokens = [t.strip(".,;:!?()").lower() for t in clean_text.split()]
        if expected in tokens or clean_text.lower().startswith(expected):
            result["passed"] = True
            result["constraint_satisfied"] = (len(clean_text) < 40)
            result["detail"] = f"Correct classification label: {expected}"
        else:
            result["detail"] = f"Expected '{expected}', got '{clean_text}'"

    elif item_type == "concise_explanation":
        kw = item.get("keywords", [])
        max_sentences = item.get("max_sentences", 2)
        has_kw = any(k.lower() in clean_text.lower() for k in kw)
        sentences = [s.strip() for s in re.split(r"[\.\!\?]", clean_text) if s.strip()]
        sentence_count = len(sentences)
        result["constraint_satisfied"] = (sentence_count <= max_sentences) and (len(clean_text) < 220)

        if has_kw and result["constraint_satisfied"] and not result["has_repetition"]:
            result["passed"] = True
            result["detail"] = "Concise, relevant factual explanation"
        else:
            result["detail"] = f"Keywords matched: {has_kw}, sentences: {sentence_count} (max {max_sentences})"

    return result
