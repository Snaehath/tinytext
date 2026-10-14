"""Targeted Evaluation Suite for TinyText Flash V1.1
Contains 102 carefully designed prompts across 6 specific failure-prone categories:
1. Exact-answer arithmetic (17 prompts)
2. Exact-count lists (17 prompts)
3. Rewriting without added claims (17 prompts)
4. Extraction preserving source facts (17 prompts)
5. Short factual explanations (17 prompts)
6. Concise classification and structured output (17 prompts)

Each prompt includes programmatic criteria for objective verification.
"""

from __future__ import annotations
import re
from typing import Any, Dict, List, Tuple


TARGETED_EVAL_SUITE_102: List[Dict[str, Any]] = [
    # -------------------------------------------------------------
    # Category 1: Exact-Answer Arithmetic (17 prompts)
    # -------------------------------------------------------------
    {
        "id": "arith_01",
        "category": "Exact-Answer Arithmetic",
        "prompt": "What is 2 + 3? Answer with only the number.",
        "target": "5",
        "type": "exact_number",
        "disallowed": ["4", "1", "6", "7", "8"],
    },
    {
        "id": "arith_02",
        "category": "Exact-Answer Arithmetic",
        "prompt": "Compute 7 + 8. Output only the number.",
        "target": "15",
        "type": "exact_number",
        "disallowed": ["14", "16", "13", "12", "0"],
    },
    {
        "id": "arith_03",
        "category": "Exact-Answer Arithmetic",
        "prompt": "What is 10 - 4? Answer with only the single number.",
        "target": "6",
        "type": "exact_number",
        "disallowed": ["5", "7", "14", "4", "0"],
    },
    {
        "id": "arith_04",
        "category": "Exact-Answer Arithmetic",
        "prompt": "What is 3 * 4? Output only the number.",
        "target": "12",
        "type": "exact_number",
        "disallowed": ["7", "11", "13", "14", "0"],
    },
    {
        "id": "arith_05",
        "category": "Exact-Answer Arithmetic",
        "prompt": "Calculate 15 / 3. Answer with only the number.",
        "target": "5",
        "type": "exact_number",
        "disallowed": ["3", "4", "6", "12", "0"],
    },
    {
        "id": "arith_06",
        "category": "Exact-Answer Arithmetic",
        "prompt": "What is 9 + 6? Output only the numerical digit.",
        "target": "15",
        "type": "exact_number",
        "disallowed": ["14", "16", "3", "0"],
    },
    {
        "id": "arith_07",
        "category": "Exact-Answer Arithmetic",
        "prompt": "What is 20 - 7? Answer with only the number.",
        "target": "13",
        "type": "exact_number",
        "disallowed": ["14", "12", "27", "7", "0"],
    },
    {
        "id": "arith_08",
        "category": "Exact-Answer Arithmetic",
        "prompt": "What is 5 * 5? Output only the number.",
        "target": "25",
        "type": "exact_number",
        "disallowed": ["20", "30", "10", "0"],
    },
    {
        "id": "arith_09",
        "category": "Exact-Answer Arithmetic",
        "prompt": "What is 18 / 2? Answer with only the number.",
        "target": "9",
        "type": "exact_number",
        "disallowed": ["8", "10", "16", "20", "0"],
    },
    {
        "id": "arith_10",
        "category": "Exact-Answer Arithmetic",
        "prompt": "Compute 14 + 11. Output only the number.",
        "target": "25",
        "type": "exact_number",
        "disallowed": ["24", "26", "23", "0"],
    },
    {
        "id": "arith_11",
        "category": "Exact-Answer Arithmetic",
        "prompt": "What is 50 - 25? Answer with only the number.",
        "target": "25",
        "type": "exact_number",
        "disallowed": ["30", "20", "75", "0"],
    },
    {
        "id": "arith_12",
        "category": "Exact-Answer Arithmetic",
        "prompt": "What is 6 * 7? Output only the number.",
        "target": "42",
        "type": "exact_number",
        "disallowed": ["40", "44", "13", "0"],
    },
    {
        "id": "arith_13",
        "category": "Exact-Answer Arithmetic",
        "prompt": "Compute 36 / 6. Output only the number.",
        "target": "6",
        "type": "exact_number",
        "disallowed": ["5", "7", "30", "0"],
    },
    {
        "id": "arith_14",
        "category": "Exact-Answer Arithmetic",
        "prompt": "What is 8 + 9? Answer with only the number.",
        "target": "17",
        "type": "exact_number",
        "disallowed": ["16", "18", "15", "0"],
    },
    {
        "id": "arith_15",
        "category": "Exact-Answer Arithmetic",
        "prompt": "What is 30 - 12? Output only the number.",
        "target": "18",
        "type": "exact_number",
        "disallowed": ["17", "19", "42", "0"],
    },
    {
        "id": "arith_16",
        "category": "Exact-Answer Arithmetic",
        "prompt": "What is 4 * 8? Answer with only the number.",
        "target": "32",
        "type": "exact_number",
        "disallowed": ["28", "36", "12", "0"],
    },
    {
        "id": "arith_17",
        "category": "Exact-Answer Arithmetic",
        "prompt": "Compute 45 / 5. Output only the number.",
        "target": "9",
        "type": "exact_number",
        "disallowed": ["8", "10", "40", "0"],
    },

    # -------------------------------------------------------------
    # Category 2: Exact-Count Lists (17 prompts)
    # -------------------------------------------------------------
    {
        "id": "list_01",
        "category": "Exact-Count Lists",
        "prompt": "Give exactly three names for a pet turtle. Output only a numbered list.",
        "target_count": 3,
        "type": "numbered_list",
    },
    {
        "id": "list_02",
        "category": "Exact-Count Lists",
        "prompt": "List exactly two primary colors. Output only a numbered list.",
        "target_count": 2,
        "type": "numbered_list",
    },
    {
        "id": "list_03",
        "category": "Exact-Count Lists",
        "prompt": "Provide exactly four fruits. Output only a numbered list.",
        "target_count": 4,
        "type": "numbered_list",
    },
    {
        "id": "list_04",
        "category": "Exact-Count Lists",
        "prompt": "Name exactly three European capital cities. Output only a numbered list.",
        "target_count": 3,
        "type": "numbered_list",
    },
    {
        "id": "list_05",
        "category": "Exact-Count Lists",
        "prompt": "Give exactly two examples of mammals that live in the ocean. Output only a numbered list.",
        "target_count": 2,
        "type": "numbered_list",
    },
    {
        "id": "list_06",
        "category": "Exact-Count Lists",
        "prompt": "List exactly three healthy breakfast foods. Output only a numbered list.",
        "target_count": 3,
        "type": "numbered_list",
    },
    {
        "id": "list_07",
        "category": "Exact-Count Lists",
        "prompt": "Name exactly two programming languages commonly used for web development. Output only a numbered list.",
        "target_count": 2,
        "type": "numbered_list",
    },
    {
        "id": "list_08",
        "category": "Exact-Count Lists",
        "prompt": "Provide exactly four planets in our solar system. Output only a numbered list.",
        "target_count": 4,
        "type": "numbered_list",
    },
    {
        "id": "list_09",
        "category": "Exact-Count Lists",
        "prompt": "Give exactly three items you should pack for a rainstorm. Output only a numbered list.",
        "target_count": 3,
        "type": "numbered_list",
    },
    {
        "id": "list_10",
        "category": "Exact-Count Lists",
        "prompt": "List exactly two renewable energy sources. Output only a numbered list.",
        "target_count": 2,
        "type": "numbered_list",
    },
    {
        "id": "list_11",
        "category": "Exact-Count Lists",
        "prompt": "Name exactly three musical instruments in an orchestra. Output only a numbered list.",
        "target_count": 3,
        "type": "numbered_list",
    },
    {
        "id": "list_12",
        "category": "Exact-Count Lists",
        "prompt": "Give exactly two benefits of drinking water regularly. Output only a numbered list.",
        "target_count": 2,
        "type": "numbered_list",
    },
    {
        "id": "list_13",
        "category": "Exact-Count Lists",
        "prompt": "List exactly four common kitchen utensils. Output only a numbered list.",
        "target_count": 4,
        "type": "numbered_list",
    },
    {
        "id": "list_14",
        "category": "Exact-Count Lists",
        "prompt": "Name exactly three breeds of dogs. Output only a numbered list.",
        "target_count": 3,
        "type": "numbered_list",
    },
    {
        "id": "list_15",
        "category": "Exact-Count Lists",
        "prompt": "Give exactly two seasons of the year. Output only a numbered list.",
        "target_count": 2,
        "type": "numbered_list",
    },
    {
        "id": "list_16",
        "category": "Exact-Count Lists",
        "prompt": "Provide exactly three common bird species. Output only a numbered list.",
        "target_count": 3,
        "type": "numbered_list",
    },
    {
        "id": "list_17",
        "category": "Exact-Count Lists",
        "prompt": "List exactly two cold desserts. Output only a numbered list.",
        "target_count": 2,
        "type": "numbered_list",
    },

    # -------------------------------------------------------------
    # Category 3: Rewriting Without Added Claims (17 prompts)
    # -------------------------------------------------------------
    {
        "id": "rewrite_01",
        "category": "Rewriting Without Added Claims",
        "prompt": "Rewrite this professionally without adding extra information: Hey, we need to talk about the budget ASAP.",
        "required_keywords": ["budget", "discuss", "talk"],
        "forbidden_phrases": ["family and friends", "I do not have access", "Consultative:"],
        "type": "rewrite",
    },
    {
        "id": "rewrite_02",
        "category": "Rewriting Without Added Claims",
        "prompt": "Rewrite in a formal tone: Can't make it to tomorrow's meeting, got sick.",
        "required_keywords": ["unable", "attend", "meeting", "ill", "sick"],
        "forbidden_phrases": ["hospital", "doctor said", "I do not have access"],
        "type": "rewrite",
    },
    {
        "id": "rewrite_03",
        "category": "Rewriting Without Added Claims",
        "prompt": "Make this sentence concise: In spite of the fact that it was raining heavily, we went outside.",
        "required_keywords": ["rain", "went outside"],
        "forbidden_phrases": ["umbrella", "boots", "storm", "I do not have access"],
        "type": "rewrite",
    },
    {
        "id": "rewrite_04",
        "category": "Rewriting Without Added Claims",
        "prompt": "Rewrite politely: Give me the updated sales report now.",
        "required_keywords": ["sales report", "please", "could you", "provide"],
        "forbidden_phrases": ["manager", "fired", "I do not have access"],
        "type": "rewrite",
    },
    {
        "id": "rewrite_05",
        "category": "Rewriting Without Added Claims",
        "prompt": "Rewrite in the active voice: The novel was written by Mary Shelley.",
        "required_keywords": ["Mary Shelley wrote the novel"],
        "forbidden_phrases": ["Frankenstein", "monster", "published in 1818"],
        "type": "rewrite",
    },
    {
        "id": "rewrite_06",
        "category": "Rewriting Without Added Claims",
        "prompt": "Fix grammatical errors in: She don't like eating apples for lunch.",
        "required_keywords": ["does not like", "doesn't like"],
        "forbidden_phrases": ["oranges", "bananas", "healthy diet"],
        "type": "rewrite",
    },
    {
        "id": "rewrite_07",
        "category": "Rewriting Without Added Claims",
        "prompt": "Rewrite professionally: That design looks really bad and ugly.",
        "required_keywords": ["design", "improvement", "refinement", "adjust"],
        "forbidden_phrases": ["fire the designer", "terrible disaster"],
        "type": "rewrite",
    },
    {
        "id": "rewrite_08",
        "category": "Rewriting Without Added Claims",
        "prompt": "Simplify for clarity: Prior to the commencement of the event, please silence mobile devices.",
        "required_keywords": ["before", "event", "silence", "phones"],
        "forbidden_phrases": ["turn off wifi", "leave the room"],
        "type": "rewrite",
    },
    {
        "id": "rewrite_09",
        "category": "Rewriting Without Added Claims",
        "prompt": "Rewrite to sound confident: I think maybe we might possibly finish this by Friday.",
        "required_keywords": ["finish", "complete", "Friday"],
        "forbidden_phrases": ["Saturday", "next week", "overtime"],
        "type": "rewrite",
    },
    {
        "id": "rewrite_10",
        "category": "Rewriting Without Added Claims",
        "prompt": "Rewrite into a single polite sentence: You forgot to attach the file. Send it to me.",
        "required_keywords": ["please", "attach", "file", "send"],
        "forbidden_phrases": ["mistake", "fired", "incompetent"],
        "type": "rewrite",
    },
    {
        "id": "rewrite_11",
        "category": "Rewriting Without Added Claims",
        "prompt": "Rewrite in the past tense: The baker bakes fresh bread every morning.",
        "required_keywords": ["baked"],
        "forbidden_phrases": ["pastries", "croissants", "bakery closed"],
        "type": "rewrite",
    },
    {
        "id": "rewrite_12",
        "category": "Rewriting Without Added Claims",
        "prompt": "Make this customer greeting professional: Hey what's up, what do you want?",
        "required_keywords": ["hello", "assist", "help", "how can I"],
        "forbidden_phrases": ["bro", "what's up"],
        "type": "rewrite",
    },
    {
        "id": "rewrite_13",
        "category": "Rewriting Without Added Claims",
        "prompt": "Rewrite without negative words: Do not be late for the presentation.",
        "required_keywords": ["on time", "punctual", "presentation"],
        "forbidden_phrases": ["penalty", "punished"],
        "type": "rewrite",
    },
    {
        "id": "rewrite_14",
        "category": "Rewriting Without Added Claims",
        "prompt": "Change to passive voice: The chef prepared a delicious dinner.",
        "required_keywords": ["dinner was prepared by the chef"],
        "forbidden_phrases": ["dessert", "restaurant", "menu"],
        "type": "rewrite",
    },
    {
        "id": "rewrite_15",
        "category": "Rewriting Without Added Claims",
        "prompt": "Rewrite concisely: At this present moment in time, the system is operational.",
        "required_keywords": ["currently", "now", "system is operational"],
        "forbidden_phrases": ["offline", "maintenance", "reboot"],
        "type": "rewrite",
    },
    {
        "id": "rewrite_16",
        "category": "Rewriting Without Added Claims",
        "prompt": "Rewrite into a direct statement: It is recommended that employees review the guidelines.",
        "required_keywords": ["employees should review", "review the guidelines"],
        "forbidden_phrases": ["or else", "mandatory test"],
        "type": "rewrite",
    },
    {
        "id": "rewrite_17",
        "category": "Rewriting Without Added Claims",
        "prompt": "Rewrite warmly: Thanks for sending the invoice.",
        "required_keywords": ["thank you", "invoice", "appreciate"],
        "forbidden_phrases": ["payment sent", "bank transfer"],
        "type": "rewrite",
    },

    # -------------------------------------------------------------
    # Category 4: Extraction Preserving Source Facts (17 prompts)
    # -------------------------------------------------------------
    {
        "id": "extract_01",
        "category": "Extraction Preserving Source Facts",
        "prompt": "Extract the time and location: The meeting is at 3:30 PM in Room B. Output: Time: ...; Location: ...",
        "required_facts": ["3:30 PM", "Room B"],
        "forbidden_facts": ["7:30 PM", "Los Angeles", "Manhattan", "Ramadan"],
        "type": "schema_extraction",
        "schema_pattern": r"Time:\s*3:30\s*PM;\s*Location:\s*Room\s*B",
    },
    {
        "id": "extract_02",
        "category": "Extraction Preserving Source Facts",
        "prompt": "Extract the date and city: The festival begins on July 14 in Chicago. Output: Date: ...; City: ...",
        "required_facts": ["July 14", "Chicago"],
        "forbidden_facts": ["New York", "August", "July 15"],
        "type": "schema_extraction",
        "schema_pattern": r"Date:\s*July\s*14;\s*City:\s*Chicago",
    },
    {
        "id": "extract_03",
        "category": "Extraction Preserving Source Facts",
        "prompt": "Extract the price and item: The vintage desk is priced at $120. Output: Item: ...; Price: ...",
        "required_facts": ["vintage desk", "$120"],
        "forbidden_facts": ["$150", "$200", "chair"],
        "type": "schema_extraction",
        "schema_pattern": r"Item:.*desk.*;\s*Price:\s*\$120",
    },
    {
        "id": "extract_04",
        "category": "Extraction Preserving Source Facts",
        "prompt": "Extract the sender and recipient: Alice sent an email to Bob this morning. Output: Sender: ...; Recipient: ...",
        "required_facts": ["Alice", "Bob"],
        "forbidden_facts": ["Charlie", "David"],
        "type": "schema_extraction",
        "schema_pattern": r"Sender:\s*Alice;\s*Recipient:\s*Bob",
    },
    {
        "id": "extract_05",
        "category": "Extraction Preserving Source Facts",
        "prompt": "Extract the vehicle model and color: He bought a blue Camry yesterday. Output: Model: ...; Color: ...",
        "required_facts": ["Camry", "blue"],
        "forbidden_facts": ["red", "Civic", "Corolla"],
        "type": "schema_extraction",
        "schema_pattern": r"Model:\s*Camry;\s*Color:\s*blue",
    },
    {
        "id": "extract_06",
        "category": "Extraction Preserving Source Facts",
        "prompt": "Extract flight number and destination: Flight AA402 is departing for Miami. Output: Flight: ...; Destination: ...",
        "required_facts": ["AA402", "Miami"],
        "forbidden_facts": ["Orlando", "AA500", "Delta"],
        "type": "schema_extraction",
        "schema_pattern": r"Flight:\s*AA402;\s*Destination:\s*Miami",
    },
    {
        "id": "extract_07",
        "category": "Extraction Preserving Source Facts",
        "prompt": "Extract the author and title: George Orwell wrote the novel 1984. Output: Author: ...; Title: ...",
        "required_facts": ["George Orwell", "1984"],
        "forbidden_facts": ["Animal Farm", "Aldous Huxley"],
        "type": "schema_extraction",
        "schema_pattern": r"Author:\s*George Orwell;\s*Title:\s*1984",
    },
    {
        "id": "extract_08",
        "category": "Extraction Preserving Source Facts",
        "prompt": "Extract the temperature and condition: Today will be 72°F and sunny. Output: Temperature: ...; Condition: ...",
        "required_facts": ["72°F", "sunny"],
        "forbidden_facts": ["rainy", "85°F", "60°F"],
        "type": "schema_extraction",
        "schema_pattern": r"Temperature:\s*72°F;\s*Condition:\s*sunny",
    },
    {
        "id": "extract_09",
        "category": "Extraction Preserving Source Facts",
        "prompt": "Extract score and winner: The Lions defeated the Bears 24 to 17. Output: Winner: ...; Score: ...",
        "required_facts": ["Lions", "24 to 17"],
        "forbidden_facts": ["Bears won", "21 to 14"],
        "type": "schema_extraction",
        "schema_pattern": r"Winner:\s*Lions;\s*Score:\s*24\s*to\s*17",
    },
    {
        "id": "extract_10",
        "category": "Extraction Preserving Source Facts",
        "prompt": "Extract doctor name and appointment time: Dr. Smith will see you at 10:15 AM. Output: Doctor: ...; Time: ...",
        "required_facts": ["Dr. Smith", "10:15 AM"],
        "forbidden_facts": ["Dr. Jones", "11:00 AM"],
        "type": "schema_extraction",
        "schema_pattern": r"Doctor:\s*Dr\.\s*Smith;\s*Time:\s*10:15\s*AM",
    },
    {
        "id": "extract_11",
        "category": "Extraction Preserving Source Facts",
        "prompt": "Extract phone number: You can contact customer support at 555-8392. Output: Phone: ...",
        "required_facts": ["555-8392"],
        "forbidden_facts": ["555-0000", "555-1234"],
        "type": "schema_extraction",
        "schema_pattern": r"Phone:\s*555-8392",
    },
    {
        "id": "extract_12",
        "category": "Extraction Preserving Source Facts",
        "prompt": "Extract ingredient and quantity: Add 2 cups of flour to the mixing bowl. Output: Ingredient: ...; Quantity: ...",
        "required_facts": ["flour", "2 cups"],
        "forbidden_facts": ["sugar", "3 cups"],
        "type": "schema_extraction",
        "schema_pattern": r"Ingredient:\s*flour;\s*Quantity:\s*2\s*cups",
    },
    {
        "id": "extract_13",
        "category": "Extraction Preserving Source Facts",
        "prompt": "Extract hotel and room number: Check in at Grand Hotel, room 412. Output: Hotel: ...; Room: ...",
        "required_facts": ["Grand Hotel", "412"],
        "forbidden_facts": ["Hilton", "101", "502"],
        "type": "schema_extraction",
        "schema_pattern": r"Hotel:\s*Grand\s*Hotel;\s*Room:\s*412",
    },
    {
        "id": "extract_14",
        "category": "Extraction Preserving Source Facts",
        "prompt": "Extract discount code and percentage: Use code SAVE20 for a 20% discount. Output: Code: ...; Discount: ...",
        "required_facts": ["SAVE20", "20%"],
        "forbidden_facts": ["SAVE10", "30%"],
        "type": "schema_extraction",
        "schema_pattern": r"Code:\s*SAVE20;\s*Discount:\s*20%",
    },
    {
        "id": "extract_15",
        "category": "Extraction Preserving Source Facts",
        "prompt": "Extract package weight and courier: The 5 kg parcel was handled by FedEx. Output: Courier: ...; Weight: ...",
        "required_facts": ["FedEx", "5 kg"],
        "forbidden_facts": ["UPS", "10 kg", "DHL"],
        "type": "schema_extraction",
        "schema_pattern": r"Courier:\s*FedEx;\s*Weight:\s*5\s*kg",
    },
    {
        "id": "extract_16",
        "category": "Extraction Preserving Source Facts",
        "prompt": "Extract company and CEO: Satya Nadella serves as the CEO of Microsoft. Output: Company: ...; CEO: ...",
        "required_facts": ["Microsoft", "Satya Nadella"],
        "forbidden_facts": ["Google", "Bill Gates", "Tim Cook"],
        "type": "schema_extraction",
        "schema_pattern": r"Company:\s*Microsoft;\s*CEO:\s*Satya\s*Nadella",
    },
    {
        "id": "extract_17",
        "category": "Extraction Preserving Source Facts",
        "prompt": "Extract course code and credits: CS101 is an introductory class worth 3 credits. Output: Course: ...; Credits: ...",
        "required_facts": ["CS101", "3 credits"],
        "forbidden_facts": ["CS201", "4 credits"],
        "type": "schema_extraction",
        "schema_pattern": r"Course:\s*CS101;\s*Credits:\s*3(\s*credits)?",
    },

    # -------------------------------------------------------------
    # Category 5: Short Factual Explanations (17 prompts)
    # -------------------------------------------------------------
    {
        "id": "explain_01",
        "category": "Short Factual Explanations",
        "prompt": "Explain gravity in one simple sentence.",
        "keywords": ["force", "attract", "mass", "pull", "earth"],
        "max_sentences": 2,
        "type": "concise_explanation",
    },
    {
        "id": "explain_02",
        "category": "Short Factual Explanations",
        "prompt": "Explain photosynthesis in one simple sentence.",
        "keywords": ["plants", "sunlight", "energy", "food", "convert"],
        "max_sentences": 2,
        "type": "concise_explanation",
    },
    {
        "id": "explain_03",
        "category": "Short Factual Explanations",
        "prompt": "What is an atom? Answer in one simple sentence.",
        "keywords": ["basic", "unit", "matter", "building block", "chemical"],
        "max_sentences": 2,
        "type": "concise_explanation",
    },
    {
        "id": "explain_04",
        "category": "Short Factual Explanations",
        "prompt": "Explain what a mammal is in one simple sentence.",
        "keywords": ["warm-blooded", "vertebrate", "milk", "hair", "give birth"],
        "max_sentences": 2,
        "type": "concise_explanation",
    },
    {
        "id": "explain_05",
        "category": "Short Factual Explanations",
        "prompt": "Explain what the solar system is in one simple sentence.",
        "keywords": ["sun", "planets", "orbit", "celestial"],
        "max_sentences": 2,
        "type": "concise_explanation",
    },
    {
        "id": "explain_06",
        "category": "Short Factual Explanations",
        "prompt": "What is DNA? Explain in one simple sentence.",
        "keywords": ["molecule", "genetic", "instructions", "hereditary", "organism"],
        "max_sentences": 2,
        "type": "concise_explanation",
    },
    {
        "id": "explain_07",
        "category": "Short Factual Explanations",
        "prompt": "Explain what an ecosystem is in one simple sentence.",
        "keywords": ["community", "living", "organisms", "environment", "interact"],
        "max_sentences": 2,
        "type": "concise_explanation",
    },
    {
        "id": "explain_08",
        "category": "Short Factual Explanations",
        "prompt": "What is evaporation? Explain in one simple sentence.",
        "keywords": ["liquid", "water", "gas", "vapor", "heat"],
        "max_sentences": 2,
        "type": "concise_explanation",
    },
    {
        "id": "explain_09",
        "category": "Short Factual Explanations",
        "prompt": "Explain what a telescope does in one simple sentence.",
        "keywords": ["instrument", "distant", "stars", "planets", "magnify", "see"],
        "max_sentences": 2,
        "type": "concise_explanation",
    },
    {
        "id": "explain_10",
        "category": "Short Factual Explanations",
        "prompt": "What is friction? Explain in one simple sentence.",
        "keywords": ["force", "resists", "motion", "surfaces", "contact"],
        "max_sentences": 2,
        "type": "concise_explanation",
    },
    {
        "id": "explain_11",
        "category": "Short Factual Explanations",
        "prompt": "Explain what an algorithm is in one simple sentence.",
        "keywords": ["step", "instructions", "solve", "problem", "rules"],
        "max_sentences": 2,
        "type": "concise_explanation",
    },
    {
        "id": "explain_12",
        "category": "Short Factual Explanations",
        "prompt": "What is oxygen used for in the human body? Answer in one simple sentence.",
        "keywords": ["cells", "energy", "breathe", "produce", "respiration"],
        "max_sentences": 2,
        "type": "concise_explanation",
    },
    {
        "id": "explain_13",
        "category": "Short Factual Explanations",
        "prompt": "Explain what a volcano is in one simple sentence.",
        "keywords": ["opening", "mountain", "lava", "earth", "crust", "erupt"],
        "max_sentences": 2,
        "type": "concise_explanation",
    },
    {
        "id": "explain_14",
        "category": "Short Factual Explanations",
        "prompt": "What is a glacier? Explain in one simple sentence.",
        "keywords": ["large", "mass", "ice", "slowly", "moving"],
        "max_sentences": 2,
        "type": "concise_explanation",
    },
    {
        "id": "explain_15",
        "category": "Short Factual Explanations",
        "prompt": "Explain what an electric current is in one simple sentence.",
        "keywords": ["flow", "electrons", "charge", "conductor", "electricity"],
        "max_sentences": 2,
        "type": "concise_explanation",
    },
    {
        "id": "explain_16",
        "category": "Short Factual Explanations",
        "prompt": "What is condensation? Answer in one simple sentence.",
        "keywords": ["gas", "vapor", "turns into", "liquid", "cools"],
        "max_sentences": 2,
        "type": "concise_explanation",
    },
    {
        "id": "explain_17",
        "category": "Short Factual Explanations",
        "prompt": "Explain what a constellation is in one simple sentence.",
        "keywords": ["group", "stars", "pattern", "sky", "recognize"],
        "max_sentences": 2,
        "type": "concise_explanation",
    },

    # -------------------------------------------------------------
    # Category 6: Concise Classification & Structured Output (17 prompts)
    # -------------------------------------------------------------
    {
        "id": "class_01",
        "category": "Concise Classification & Structured Output",
        "prompt": "Classify this review sentiment as Positive or Negative. Output only the label: The product broke within an hour.",
        "expected_label": "Negative",
        "type": "classification",
    },
    {
        "id": "class_02",
        "category": "Concise Classification & Structured Output",
        "prompt": "Classify this review sentiment as Positive or Negative. Output only the label: Absolutely loved the quality, exceeded my expectations!",
        "expected_label": "Positive",
        "type": "classification",
    },
    {
        "id": "class_03",
        "category": "Concise Classification & Structured Output",
        "prompt": "Classify as Animal, Plant, or Mineral: An oak tree. Output only the category.",
        "expected_label": "Plant",
        "type": "classification",
    },
    {
        "id": "class_04",
        "category": "Concise Classification & Structured Output",
        "prompt": "Classify as Animal, Plant, or Mineral: A golden eagle. Output only the category.",
        "expected_label": "Animal",
        "type": "classification",
    },
    {
        "id": "class_05",
        "category": "Concise Classification & Structured Output",
        "prompt": "Classify as Animal, Plant, or Mineral: Granite rock. Output only the category.",
        "expected_label": "Mineral",
        "type": "classification",
    },
    {
        "id": "class_06",
        "category": "Concise Classification & Structured Output",
        "prompt": "Classify the weather condition as Hot or Cold: Blizzard with sub-zero temperatures. Output only the label.",
        "expected_label": "Cold",
        "type": "classification",
    },
    {
        "id": "class_07",
        "category": "Concise Classification & Structured Output",
        "prompt": "Classify the weather condition as Hot or Cold: Scorching afternoon in Death Valley at 115°F. Output only the label.",
        "expected_label": "Hot",
        "type": "classification",
    },
    {
        "id": "class_08",
        "category": "Concise Classification & Structured Output",
        "prompt": "Is a dolphin a Mammal or a Fish? Output only the label.",
        "expected_label": "Mammal",
        "type": "classification",
    },
    {
        "id": "class_09",
        "category": "Concise Classification & Structured Output",
        "prompt": "Is iron a Solid, Liquid, or Gas at room temperature? Output only the label.",
        "expected_label": "Solid",
        "type": "classification",
    },
    {
        "id": "class_10",
        "category": "Concise Classification & Structured Output",
        "prompt": "Is water a Solid, Liquid, or Gas at 20°C? Output only the label.",
        "expected_label": "Liquid",
        "type": "classification",
    },
    {
        "id": "class_11",
        "category": "Concise Classification & Structured Output",
        "prompt": "Classify this statement as True or False: The Earth is the third planet from the Sun. Output only the label.",
        "expected_label": "True",
        "type": "classification",
    },
    {
        "id": "class_12",
        "category": "Concise Classification & Structured Output",
        "prompt": "Classify this statement as True or False: Humans can breathe underwater without equipment. Output only the label.",
        "expected_label": "False",
        "type": "classification",
    },
    {
        "id": "class_13",
        "category": "Concise Classification & Structured Output",
        "prompt": "Classify as Fiction or Non-Fiction: The historical biography of Abraham Lincoln. Output only the label.",
        "expected_label": "Non-Fiction",
        "type": "classification",
    },
    {
        "id": "class_14",
        "category": "Concise Classification & Structured Output",
        "prompt": "Classify as Fiction or Non-Fiction: A fantasy story about wizards and dragons. Output only the label.",
        "expected_label": "Fiction",
        "type": "classification",
    },
    {
        "id": "class_15",
        "category": "Concise Classification & Structured Output",
        "prompt": "Classify as Indoor or Outdoor activity: Playing chess on a board inside a library. Output only the label.",
        "expected_label": "Indoor",
        "type": "classification",
    },
    {
        "id": "class_16",
        "category": "Concise Classification & Structured Output",
        "prompt": "Classify as Indoor or Outdoor activity: Hiking up a mountain trail. Output only the label.",
        "expected_label": "Outdoor",
        "type": "classification",
    },
    {
        "id": "class_17",
        "category": "Concise Classification & Structured Output",
        "prompt": "Classify the priority as High or Low: The main production server is down and losing customer data. Output only the label.",
        "expected_label": "High",
        "type": "classification",
    },
]


def score_response(item: Dict[str, Any], text: str) -> Dict[str, Any]:
    """Evaluates a generated response against objective criteria."""
    clean_text = text.strip()
    result = {
        "passed": False,
        "constraint_satisfied": False,
        "has_repetition": False,
        "has_contradiction": False,
        "detail": "",
    }

    # Check repetition loops (any 3-gram repeated 4+ times, or exact sentence repeated)
    words = clean_text.lower().split()
    if len(words) >= 9:
        trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
        for tg in set(trigrams):
            if trigrams.count(tg) >= 3:
                result["has_repetition"] = True
                break

    item_type = item.get("type")

    if item_type == "exact_number":
        target = item["target"]
        disallowed = item.get("disallowed", [])
        # Extract digits from response
        found_numbers = re.findall(r"\b\d+\b", clean_text)
        if target in found_numbers:
            # Check for conflicting numbers
            contradictions = [n for n in found_numbers if n != target and n in disallowed]
            if len(contradictions) > 0:
                result["has_contradiction"] = True
                result["detail"] = f"Found target {target} but also contradictory numbers {contradictions}"
            else:
                result["passed"] = True
                result["constraint_satisfied"] = (len(clean_text) < 30)  # Conciseness check
                result["detail"] = f"Correct exact number: {target}"
        else:
            result["detail"] = f"Missing target {target}, found {found_numbers}"

    elif item_type == "numbered_list":
        target_count = item["target_count"]
        # Find numbered items like "1.", "1)", "1 -"
        list_items = re.findall(r"(?:^|\n)\s*(\d+)[\.\)]\s*(.+)", clean_text)
        if list_items:
            found_count = len(list_items)
            result["constraint_satisfied"] = (found_count == target_count)
            # Check distinct items
            item_texts = [it[1].strip().lower() for it in list_items]
            unique_items = len(set(item_texts)) == len(item_texts)
            if result["constraint_satisfied"] and unique_items and not result["has_repetition"]:
                result["passed"] = True
                result["detail"] = f"Matched exact list count of {target_count}"
            else:
                result["detail"] = f"Found {found_count} items (expected {target_count}), unique={unique_items}"
        else:
            result["detail"] = "No numbered list format detected"

    elif item_type == "rewrite":
        # Check required keywords
        req_kw = item.get("required_keywords", [])
        forb_phrases = item.get("forbidden_phrases", [])
        
        has_forbidden = any(f.lower() in clean_text.lower() for f in forb_phrases)
        has_any_req = any(k.lower() in clean_text.lower() for k in req_kw)

        result["constraint_satisfied"] = not has_forbidden and (len(clean_text) < 180)
        if has_any_req and result["constraint_satisfied"] and not result["has_repetition"]:
            result["passed"] = True
            result["detail"] = "Professional rewrite preserved context without added forbidden claims"
        else:
            result["detail"] = f"Forbidden: {has_forbidden}, Req keywords matched: {has_any_req}"

    elif item_type == "schema_extraction":
        req_facts = item.get("required_facts", [])
        forb_facts = item.get("forbidden_facts", [])
        schema_pattern = item.get("schema_pattern")

        has_all_req = all(rf.lower() in clean_text.lower() for rf in req_facts)
        has_any_forb = any(ff.lower() in clean_text.lower() for ff in forb_facts)
        matches_schema = bool(re.search(schema_pattern, clean_text, re.IGNORECASE)) if schema_pattern else True

        result["has_contradiction"] = has_any_forb
        result["constraint_satisfied"] = matches_schema
        if has_all_req and not has_any_forb and matches_schema and not result["has_repetition"]:
            result["passed"] = True
            result["detail"] = "Accurate schema extraction with source facts preserved"
        else:
            result["detail"] = f"Req facts: {has_all_req}, Forbidden hallucination: {has_any_forb}, Schema: {matches_schema}"

    elif item_type == "concise_explanation":
        kw = item.get("keywords", [])
        has_kw = any(k.lower() in clean_text.lower() for k in kw)
        sentences = [s.strip() for s in re.split(r"[\.\!\?]", clean_text) if s.strip()]
        sentence_count = len(sentences)
        result["constraint_satisfied"] = (sentence_count <= 2) and (len(clean_text) < 200)

        if has_kw and result["constraint_satisfied"] and not result["has_repetition"]:
            result["passed"] = True
            result["detail"] = "Concise, relevant factual explanation"
        else:
            result["detail"] = f"Keywords matched: {has_kw}, sentences: {sentence_count}, repetition: {result['has_repetition']}"

    elif item_type == "classification":
        expected = item.get("expected_label", "").lower()
        # Look for expected label in text
        tokens = [t.strip(".,;:!?()").lower() for t in clean_text.split()]
        if expected in tokens or clean_text.lower().startswith(expected):
            result["passed"] = True
            result["constraint_satisfied"] = (len(clean_text) < 40)
            result["detail"] = f"Correct classification label: {expected}"
        else:
            result["detail"] = f"Expected '{expected}', got '{clean_text}'"

    return result
