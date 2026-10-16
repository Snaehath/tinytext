"""Independently Held-Out Evaluation Suite for TinyText Flash V1.5.6.

Contains 30 newly authored tasks with graded difficulty (10 Easy, 10 Medium, 10 Hard)
across:
- Arithmetic Reasoning (10 tasks)
- Symbolic Manipulation (10 tasks)
- Constrained Problem Solving & Error Detection (10 tasks)

Strictly isolated from training and validation distributions, and independent
from historical 88- and 50-prompt benchmarks.
"""

from __future__ import annotations

from typing import Any, Dict, List

HELDOUT_EVAL_TASKS: List[Dict[str, Any]] = [
    # =========================================================================
    # Arithmetic Reasoning (10 tasks)
    # =========================================================================
    {
        "id": "eval_arith_01",
        "domain": "arithmetic",
        "difficulty": "easy",
        "prompt": "What is (17 * 4) - 23?",
        "ground_truth": "45",
        "intermediate_steps": ["17 * 4 = 68", "68 - 23 = 45"],
        "check": "45 + 23 = 68, 68 / 4 = 17",
    },
    {
        "id": "eval_arith_02",
        "domain": "arithmetic",
        "difficulty": "easy",
        "prompt": "A warehouse had 140 boxes. They shipped 55 boxes in the morning and 42 boxes in the afternoon. How many boxes remain?",
        "ground_truth": "43",
        "intermediate_steps": ["Shipped total = 55 + 42 = 97", "Remaining = 140 - 97 = 43"],
        "check": "43 + 97 = 140",
    },
    {
        "id": "eval_arith_03",
        "domain": "arithmetic",
        "difficulty": "easy",
        "prompt": "Compute 84 / 4 + 19.",
        "ground_truth": "40",
        "intermediate_steps": ["84 / 4 = 21", "21 + 19 = 40"],
        "check": "40 - 19 = 21, 21 * 4 = 84",
    },
    {
        "id": "eval_arith_04",
        "domain": "arithmetic",
        "difficulty": "medium",
        "prompt": "Calculate (105 / 7) * 6 - 48.",
        "ground_truth": "42",
        "intermediate_steps": ["105 / 7 = 15", "15 * 6 = 90", "90 - 48 = 42"],
        "check": "42 + 48 = 90, 90 / 6 = 15",
    },
    {
        "id": "eval_arith_05",
        "domain": "arithmetic",
        "difficulty": "medium",
        "prompt": "A jacket originally costs $160. It is discounted by 25%. A sales tax of 5% is then applied to the discounted price. What is the final price in dollars?",
        "ground_truth": "126",
        "intermediate_steps": ["Discount = 160 * 0.25 = 40", "Discounted = 120", "Tax = 120 * 0.05 = 6", "Final = 120 + 6 = 126"],
        "check": "120 * 1.05 = 126",
    },
    {
        "id": "eval_arith_06",
        "domain": "arithmetic",
        "difficulty": "medium",
        "prompt": "Find the remainder when 629 is divided by 17.",
        "ground_truth": "0",
        "intermediate_steps": ["17 * 37 = 629", "Remainder = 629 - 629 = 0"],
        "check": "17 * 37 = 629, remainder is 0",
    },
    {
        "id": "eval_arith_07",
        "domain": "arithmetic",
        "difficulty": "medium",
        "prompt": "What is the average of 18, 26, 34, 42, and 50?",
        "ground_truth": "34",
        "intermediate_steps": ["Sum = 18 + 26 + 34 + 42 + 50 = 170", "Average = 170 / 5 = 34"],
        "check": "Symmetric arithmetic progression around middle value 34",
    },
    {
        "id": "eval_arith_08",
        "domain": "arithmetic",
        "difficulty": "hard",
        "prompt": "A vehicle drives at 50 mph for 2 hours, then 70 mph for 1.5 hours, and finally 30 mph for 30 minutes (0.5 hour). What is the total distance traveled in miles?",
        "ground_truth": "220",
        "intermediate_steps": ["Leg 1: 50 * 2 = 100", "Leg 2: 70 * 1.5 = 105", "Leg 3: 30 * 0.5 = 15", "Total = 100 + 105 + 15 = 220"],
        "check": "100 + 105 + 15 = 220 miles",
    },
    {
        "id": "eval_arith_09",
        "domain": "arithmetic",
        "difficulty": "hard",
        "prompt": "What is the sum of all prime numbers between 40 and 60?",
        "ground_truth": "243",
        "intermediate_steps": ["Primes between 40 and 60: 41, 43, 47, 53, 59", "41 + 43 = 84", "84 + 47 = 131", "131 + 53 = 184", "184 + 59 = 243"],
        "check": "41 + 43 + 47 + 53 + 59 = 243",
    },
    {
        "id": "eval_arith_10",
        "domain": "arithmetic",
        "difficulty": "hard",
        "prompt": "If 3x + 14 = 59, what is the value of 2x^2 - 15?",
        "ground_truth": "435",
        "intermediate_steps": ["3x = 59 - 14 = 45 -> x = 15", "x^2 = 225", "2 * 225 = 450", "450 - 15 = 435"],
        "check": "x = 15, 3(15)+14 = 59, 2(15^2)-15 = 435",
    },

    # =========================================================================
    # Symbolic Manipulation (10 tasks)
    # =========================================================================
    {
        "id": "eval_symb_01",
        "domain": "symbolic",
        "difficulty": "easy",
        "prompt": "Reverse the characters of the word 'LANTERN'.",
        "ground_truth": "NRETNAL",
        "intermediate_steps": ["L-A-N-T-E-R-N reversed is N-R-E-T-N-A-L"],
        "check": "Length 7, starts with N, ends with L",
    },
    {
        "id": "eval_symb_02",
        "domain": "symbolic",
        "difficulty": "easy",
        "prompt": "Count the total number of vowels (A, E, I, O, U) in the word 'METEOROLOGY'.",
        "ground_truth": "5",
        "intermediate_steps": ["Vowels: E(1), E(2), O(3), O(4), O(5)", "Count = 5"],
        "check": "E, E, O, O, O are vowels (total 5)",
    },
    {
        "id": "eval_symb_03",
        "domain": "symbolic",
        "difficulty": "easy",
        "prompt": "Extract all consonants from the word 'COSMOS' in original order.",
        "ground_truth": "CSMS",
        "intermediate_steps": ["Consonants: C, S, M, S", "Vowels: O, O"],
        "check": "Length 4, all non-vowels",
    },
    {
        "id": "eval_symb_04",
        "domain": "symbolic",
        "difficulty": "medium",
        "prompt": "Sort the letters in the word 'CHESTNUT' in alphabetical order.",
        "ground_truth": "CEHNSTTU",
        "intermediate_steps": ["Letters: C, H, E, S, T, N, U, T", "Sorted: C, E, H, N, S, T, T, U"],
        "check": "Length 8, multiset matches",
    },
    {
        "id": "eval_symb_05",
        "domain": "symbolic",
        "difficulty": "medium",
        "prompt": "What is the next number in the sequence: 4, 9, 19, 39, 79, ...?",
        "ground_truth": "159",
        "intermediate_steps": ["Rule: x_{n+1} = 2 * x_n + 1 (or +5, +10, +20, +40, +80)", "79 * 2 + 1 = 159", "79 + 80 = 159"],
        "check": "159 - 79 = 80 = 2 * 40",
    },
    {
        "id": "eval_symb_06",
        "domain": "symbolic",
        "difficulty": "medium",
        "prompt": "What is the 4th letter from the end of the word 'REVOLUTIONARY'?",
        "ground_truth": "N",
        "intermediate_steps": ["Letters from end: Y(1st), R(2nd), A(3rd), N(4th)", "The 4th letter from end is N"],
        "check": "Length 13. Index 13 - 4 = 9 (0-indexed). 0:R, 1:E, 2:V, 3:O, 4:L, 5:U, 6:T, 7:I, 8:O, 9:N",
    },
    {
        "id": "eval_symb_07",
        "domain": "symbolic",
        "difficulty": "medium",
        "prompt": "Apply a Caesar cipher with a shift of +2 (A->C, B->D, etc.) to encrypt the word 'TINY'.",
        "ground_truth": "VKPA",
        "intermediate_steps": ["T + 2 = V", "I + 2 = K", "N + 2 = P", "Y + 2 = A (wraparound)"],
        "check": "V-2=T, K-2=I, P-2=N, A-2=Y",
    },
    {
        "id": "eval_symb_08",
        "domain": "symbolic",
        "difficulty": "hard",
        "prompt": "Determine the character that appears most frequently in the string 'PARALLELEPIPED' and how many times it appears.",
        "ground_truth": "P appears 3 times (or E appears 3 times, L appears 3 times)",
        "ground_truth_clean": "3",
        "intermediate_steps": ["Counts: P:3, A:2, R:1, L:3, E:3, I:1, D:1", "Max frequency is 3"],
        "check": "P, L, and E each appear 3 times",
    },
    {
        "id": "eval_symb_09",
        "domain": "symbolic",
        "difficulty": "hard",
        "prompt": "Replace every vowel in the word 'QUANTUM' with '#' and every consonant with '$'. What is the resulting pattern?",
        "ground_truth": "$##$$#$",
        "intermediate_steps": [
            "Q: consonant ($)", "U: vowel (#)", "A: vowel (#)",
            "N: consonant ($)", "T: consonant ($)", "U: vowel (#)", "M: consonant ($)"
        ],
        "check": "Pattern length 7: $ # # $ $ # $",
    },
    {
        "id": "eval_symb_10",
        "domain": "symbolic",
        "difficulty": "hard",
        "prompt": "Reverse the order of words in: 'precision reasoning requires verified intermediate steps'.",
        "ground_truth": "steps intermediate verified requires reasoning precision",
        "intermediate_steps": ["Words: [precision, reasoning, requires, verified, intermediate, steps]", "Reversed: steps intermediate verified requires reasoning precision"],
        "check": "6 words reversed exactly",
    },

    # =========================================================================
    # Constrained Problem Solving & Error Detection (10 tasks)
    # =========================================================================
    {
        "id": "eval_logic_01",
        "domain": "constrained",
        "difficulty": "easy",
        "prompt": "From the list [15, 22, 37, 44, 59, 68, 73], list all numbers that are even AND less than 50 in ascending order.",
        "ground_truth": "22, 44",
        "intermediate_steps": ["Evens: 22, 44, 68", "Less than 50: 22, 44"],
        "check": "22 < 50 (even), 44 < 50 (even), 68 >= 50 (excluded)",
    },
    {
        "id": "eval_logic_02",
        "domain": "constrained",
        "difficulty": "easy",
        "prompt": "Emma is taller than Noah. Noah is taller than Liam. Sophia is taller than Emma. Who is the tallest person?",
        "ground_truth": "Sophia",
        "intermediate_steps": ["Sophia > Emma > Noah > Liam", "Tallest is Sophia"],
        "check": "Sophia exceeds Emma, who exceeds everyone else",
    },
    {
        "id": "eval_logic_03",
        "domain": "constrained",
        "difficulty": "easy",
        "prompt": "A bag contains red, blue, green, and yellow marbles. It is not red, not green, and not yellow. What color is the marble?",
        "ground_truth": "blue",
        "intermediate_steps": ["Options: [red, blue, green, yellow]", "Eliminate: red, green, yellow", "Remaining: blue"],
        "check": "Only blue satisfies all 3 exclusions",
    },
    {
        "id": "eval_logic_04",
        "domain": "constrained",
        "difficulty": "medium",
        "prompt": "Find the two-digit number where the units digit is 3 times the tens digit and their sum is 12.",
        "ground_truth": "39",
        "intermediate_steps": ["Let number be 10T + U", "U = 3T", "T + U = 12 -> T + 3T = 12 -> 4T = 12 -> T = 3", "U = 3 * 3 = 9", "Number = 39"],
        "check": "Tens is 3, units is 9. 9 = 3 * 3. 3 + 9 = 12.",
    },
    {
        "id": "eval_logic_05",
        "domain": "constrained",
        "difficulty": "medium",
        "prompt": "Three tasks (A, B, C) take 1 hour, 2 hours, and 3 hours. Task C cannot be done first. Task A must be completed before Task B. What is the only valid sequence of tasks?",
        "ground_truth": "A, B, C or A, C, B",
        "ground_truth_clean": "A, C, B or A, B, C",
        "intermediate_steps": [
            "A before B means A must precede B.",
            "C cannot be first, so 1st task must be A (since B cannot precede A).",
            "Possible orders starting with A where A precedes B: [A, B, C] and [A, C, B]."
        ],
        "check": "In both [A, B, C] and [A, C, B], C is not first and A precedes B.",
    },
    {
        "id": "eval_logic_06",
        "domain": "constrained",
        "difficulty": "medium",
        "prompt": "Check this calculation: 'A student claimed that 24 * 7 = 158.' Is this claim correct or incorrect, and what is the actual product?",
        "ground_truth": "incorrect, 168",
        "intermediate_steps": ["24 * 7 = (20 * 7) + (4 * 7) = 140 + 28 = 168", "168 != 158, so the claim is incorrect"],
        "check": "Claimed 158 is wrong by 10. Actual product is 168.",
    },
    {
        "id": "eval_logic_07",
        "domain": "constrained",
        "difficulty": "hard",
        "prompt": "Verify this proposed deduction: 'All squares are rectangles. Some rectangles are red. Therefore, all squares are red.' Is this deduction valid or invalid, and why?",
        "ground_truth": "invalid",
        "intermediate_steps": [
            "Premise 1: Squares subset of Rectangles.",
            "Premise 2: Red intersects Rectangles.",
            "Conclusion: Squares subset of Red.",
            "Fallacy: Undistributed middle / existential fallacy. Squares could belong to the non-red portion of rectangles."
        ],
        "check": "Counterexample: A blue square is a rectangle, but not red. The deduction is invalid.",
    },
    {
        "id": "eval_logic_08",
        "domain": "constrained",
        "difficulty": "hard",
        "prompt": "Four runners (P, Q, R, S) finish a race. P finished before Q. R finished before S. S finished before P. What was the exact finishing order from 1st to 4th?",
        "ground_truth": "R, S, P, Q",
        "intermediate_steps": [
            "P before Q: P > Q",
            "R before S: R > S",
            "S before P: S > P",
            "Chaining relationships: R > S > P > Q",
            "Order: 1st R, 2nd S, 3rd P, 4th Q"
        ],
        "check": "R > S (yes), S > P (yes), P > Q (yes). Order is R, S, P, Q.",
    },
    {
        "id": "eval_logic_09",
        "domain": "constrained",
        "difficulty": "hard",
        "prompt": "A clock shows 3:15. What is the angle in degrees between the hour hand and the minute hand?",
        "ground_truth": "7.5",
        "intermediate_steps": [
            "Minute hand at 15 minutes is at 15 * 6 = 90 degrees.",
            "Hour hand at 3 hours is at 3 * 30 = 90 degrees.",
            "In 15 minutes, the hour hand advances: 15 * 0.5 degrees = 7.5 degrees.",
            "Position of hour hand = 90 + 7.5 = 97.5 degrees.",
            "Angle difference = 97.5 - 90 = 7.5 degrees."
        ],
        "check": "360 degrees / 12 hours = 30 deg/hr = 0.5 deg/min. 15 min * 0.5 deg/min = 7.5 deg.",
    },
    {
        "id": "eval_logic_10",
        "domain": "constrained",
        "difficulty": "hard",
        "prompt": "Audit this intermediate step: In solving 5x - 3(2x - 4) = 18, a draft wrote '5x - 6x - 12 = 18'. Detect the error in the draft step and provide the corrected value of x.",
        "ground_truth": "sign error, -6",
        "intermediate_steps": [
            "Expanding -3(2x - 4): -3 * 2x = -6x, and -3 * (-4) = +12 (draft had -12, sign error).",
            "Correct equation: 5x - 6x + 12 = 18",
            "-x + 12 = 18 -> -x = 6 -> x = -6."
        ],
        "check": "Check x = -6 in original: 5(-6) - 3(2(-6) - 4) = -30 - 3(-12 - 4) = -30 - 3(-16) = -30 + 48 = 18. Verified.",
    },
]
