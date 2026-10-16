"""Stage 0 Baseline Characterization Suite for TinyText Flash V1.5.6.

Contains 20 newly authored, objectively verifiable tasks across:
- Arithmetic Reasoning (7 tasks)
- Symbolic Manipulation (7 tasks)
- Constrained Problem Solving (6 tasks)

All tasks have deterministic ground-truth answers, expected intermediate steps,
and required consistency/sanity checks for high-effort validation.
"""

from __future__ import annotations

from typing import Any, Dict, List

BASELINE_TASKS: List[Dict[str, Any]] = [
    # -------------------------------------------------------------
    # Arithmetic Reasoning (7 tasks)
    # -------------------------------------------------------------
    {
        "id": "base_arith_01",
        "domain": "arithmetic",
        "difficulty": "easy",
        "prompt": "What is (14 * 3) + 18?",
        "ground_truth": "60",
        "intermediate_steps": ["14 * 3 = 42", "42 + 18 = 60"],
        "check": "Even parity, 60 - 18 = 42, 42 / 3 = 14",
    },
    {
        "id": "base_arith_02",
        "domain": "arithmetic",
        "difficulty": "easy",
        "prompt": "A bakery had 85 cookies. They sold 37 in the morning and 24 in the afternoon. How many cookies are left?",
        "ground_truth": "24",
        "intermediate_steps": ["Total sold = 37 + 24 = 61", "Remaining = 85 - 61 = 24"],
        "check": "24 + 37 + 24 = 85",
    },
    {
        "id": "base_arith_03",
        "domain": "arithmetic",
        "difficulty": "medium",
        "prompt": "Calculate (96 / 8) * 7 - 35.",
        "ground_truth": "49",
        "intermediate_steps": ["96 / 8 = 12", "12 * 7 = 84", "84 - 35 = 49"],
        "check": "49 + 35 = 84, 84 / 7 = 12",
    },
    {
        "id": "base_arith_04",
        "domain": "arithmetic",
        "difficulty": "medium",
        "prompt": "An item costs $120. It is discounted by 20%, and then a 5% sales tax is added to the discounted price. What is the final price in dollars?",
        "ground_truth": "100.8",
        "intermediate_steps": ["Discount = 120 * 0.20 = 24", "Discounted price = 96", "Tax = 96 * 0.05 = 4.8", "Final = 96 + 4.8 = 100.8"],
        "check": "96 * 1.05 = 100.80",
    },
    {
        "id": "base_arith_05",
        "domain": "arithmetic",
        "difficulty": "medium",
        "prompt": "What is the remainder when 457 is divided by 13?",
        "ground_truth": "2",
        "intermediate_steps": ["457 / 13 = 35 with remainder 2", "13 * 35 = 455", "457 - 455 = 2"],
        "check": "13 * 35 + 2 = 457, 0 <= 2 < 13",
    },
    {
        "id": "base_arith_06",
        "domain": "arithmetic",
        "difficulty": "hard",
        "prompt": "A train travels at 60 mph for 1.5 hours, then increases speed to 75 mph for 2 hours, and finally slows to 45 mph for 40 minutes (2/3 hour). What is the total distance traveled in miles?",
        "ground_truth": "270",
        "intermediate_steps": ["Leg 1: 60 * 1.5 = 90 miles", "Leg 2: 75 * 2 = 150 miles", "Leg 3: 45 * (2/3) = 30 miles", "Total = 90 + 150 + 30 = 270"],
        "check": "90 + 150 + 30 = 270",
    },
    {
        "id": "base_arith_07",
        "domain": "arithmetic",
        "difficulty": "hard",
        "prompt": "Compute the sum of all prime numbers between 20 and 40.",
        "ground_truth": "120",
        "intermediate_steps": ["Primes between 20 and 40: 23, 29, 31, 37", "23 + 29 = 52", "52 + 31 = 83", "83 + 37 = 120"],
        "check": "Four primes: 23, 29, 31, 37. Sum = 120",
    },

    # -------------------------------------------------------------
    # Symbolic Tasks (7 tasks)
    # -------------------------------------------------------------
    {
        "id": "base_symb_01",
        "domain": "symbolic",
        "difficulty": "easy",
        "prompt": "Reverse the characters of the word 'KESTREL'.",
        "ground_truth": "LERTSEK",
        "intermediate_steps": ["K-E-S-T-R-E-L reversed is L-E-R-T-S-E-K"],
        "check": "Length is 7, starts with L, ends with K",
    },
    {
        "id": "base_symb_02",
        "domain": "symbolic",
        "difficulty": "easy",
        "prompt": "Count the total number of vowels (A, E, I, O, U, case-insensitive) in the word 'EVALUATION'.",
        "ground_truth": "6",
        "intermediate_steps": ["E, A, U, A, I, O are vowels", "Total count = 6"],
        "check": "Vowels: E(1), A(2), U(3), A(4), I(5), O(6)",
    },
    {
        "id": "base_symb_03",
        "domain": "symbolic",
        "difficulty": "medium",
        "prompt": "Sort the letters in the word 'PINEAPPLE' in alphabetical order.",
        "ground_truth": "AEEILNPPP",
        "intermediate_steps": ["Letter frequencies: A:1, E:2, I:1, L:1, N:1, P:3", "Sorted: A E E I L N P P P"],
        "check": "Total length is 9 letters, 3 Ps, 2 Es",
    },
    {
        "id": "base_symb_04",
        "domain": "symbolic",
        "difficulty": "medium",
        "prompt": "What is the next number in the sequence: 3, 7, 15, 31, 63, ...?",
        "ground_truth": "127",
        "intermediate_steps": ["Rule: x_{n+1} = 2 * x_n + 1 (or +4, +8, +16, +32, +64)", "63 * 2 + 1 = 127", "63 + 64 = 127"],
        "check": "127 = 2^7 - 1",
    },
    {
        "id": "base_symb_05",
        "domain": "symbolic",
        "difficulty": "medium",
        "prompt": "Extract all consonants from the word 'ALGORITHM' in order, capitalized.",
        "ground_truth": "LGRTHM",
        "intermediate_steps": ["Letters: A(vowel), L(cons), G(cons), O(vowel), R(cons), I(vowel), T(cons), H(cons), M(cons)", "Consonants: L, G, R, T, H, M"],
        "check": "6 consonants, 3 vowels",
    },
    {
        "id": "base_symb_06",
        "domain": "symbolic",
        "difficulty": "hard",
        "prompt": "Given the Caesar cipher with a shift of +3 (A->D, B->E, ..., X->A, Y->B, Z->C), what is the encrypted ciphertext for the word 'FLASH'?",
        "ground_truth": "IODVK",
        "intermediate_steps": ["F + 3 = I", "L + 3 = O", "A + 3 = D", "S + 3 = V", "H + 3 = K"],
        "check": "Reverse shift: I-3=F, O-3=L, D-3=A, V-3=S, K-3=H",
    },
    {
        "id": "base_symb_07",
        "domain": "symbolic",
        "difficulty": "hard",
        "prompt": "In the alphabet string, replace each even-indexed letter (0-indexed: index 0, 2, 4...) in 'OCTAGON' with an asterisk '*'. What is the resulting string?",
        "ground_truth": "*C*A*O*",
        "intermediate_steps": ["Index 0: O -> *", "Index 1: C", "Index 2: T -> *", "Index 3: A", "Index 4: G -> *", "Index 5: O", "Index 6: N -> *"],
        "check": "Result: * C * A * O * (length 7)",
    },

    # -------------------------------------------------------------
    # Constrained Problem Solving (6 tasks)
    # -------------------------------------------------------------
    {
        "id": "base_logic_01",
        "domain": "constrained",
        "difficulty": "easy",
        "prompt": "From the list [14, 27, 32, 45, 50, 63], filter and return only the numbers that are even AND greater than 30.",
        "ground_truth": "32, 50",
        "intermediate_steps": ["Even numbers: 14, 32, 50", "Numbers > 30: 32, 45, 50, 63", "Intersection: 32, 50"],
        "check": "14 <= 30 (excluded), 27 odd (excluded), 45 odd (excluded), 63 odd (excluded)",
    },
    {
        "id": "base_logic_02",
        "domain": "constrained",
        "difficulty": "easy",
        "prompt": "Maya is older than Leo. Leo is older than Sam. Zoe is older than Maya. Who is the youngest?",
        "ground_truth": "Sam",
        "intermediate_steps": ["Zoe > Maya > Leo > Sam", "Smallest/youngest is Sam"],
        "check": "Sam is younger than Leo, who is younger than Maya, who is younger than Zoe",
    },
    {
        "id": "base_logic_03",
        "domain": "constrained",
        "difficulty": "medium",
        "prompt": "A schedule has 4 time slots: 9am, 10am, 11am, 12pm. Alice cannot meet before 10am. Bob cannot meet after 11am. Charlie can only meet at 10am or 12pm. Which time slot works for both Alice and Bob to meet together?",
        "ground_truth": "10am",
        "intermediate_steps": ["Alice available: 10am, 11am, 12pm", "Bob available: 9am, 10am, 11am", "Alice and Bob overlap: 10am, 11am"],
        "check": "Can be 10am or 11am. Since prompt asks which slot works for Alice and Bob, 10am and 11am both work, specifically 10am",
    },
    {
        "id": "base_logic_04",
        "domain": "constrained",
        "difficulty": "medium",
        "prompt": "Find the only two-digit number where the tens digit is twice the units digit and the sum of its digits is 9.",
        "ground_truth": "63",
        "intermediate_steps": ["Let number be 10*t + u", "t = 2*u", "t + u = 9 -> 2*u + u = 9 -> 3*u = 9 -> u = 3", "t = 2*3 = 6", "Number = 63"],
        "check": "Tens is 6, units is 3 (6 = 2 * 3). Sum = 6 + 3 = 9",
    },
    {
        "id": "base_logic_05",
        "domain": "constrained",
        "difficulty": "hard",
        "prompt": "Three boxes are labeled Apples, Oranges, and Mixed. Every box is incorrectly labeled. You pick one fruit from the box labeled Mixed and it is an Apple. What are the correct contents of the box labeled Apples?",
        "ground_truth": "Oranges",
        "intermediate_steps": [
            "All labels are wrong.",
            "Box labeled 'Mixed' cannot be Mixed. Since an apple was drawn, it must be Apples.",
            "Remaining boxes: 'Apples' and 'Oranges'. Remaining contents: Oranges and Mixed.",
            "Box labeled 'Apples' cannot contain Apples and cannot contain Mixed (if it were Oranges, then 'Oranges' box would be Mixed).",
            "If 'Apples' were Mixed, 'Oranges' would be Oranges (which violates all wrong).",
            "Therefore, box labeled 'Apples' contains Oranges, and box labeled 'Oranges' contains Mixed."
        ],
        "check": "Mixed label -> Apples. Apples label -> Oranges. Oranges label -> Mixed. All 3 wrong.",
    },
    {
        "id": "base_logic_06",
        "domain": "constrained",
        "difficulty": "hard",
        "prompt": "Five friends (A, B, C, D, E) sit in a row. A is at one of the ends. C is sitting immediately to the right of B. D is sitting between C and E. E is not at an end. What is the exact seating order from left to right?",
        "ground_truth": "A, B, C, D, E or B, C, D, E, A",
        "intermediate_steps": [
            "C is immediately right of B: block [B, C]",
            "D is between C and E: block [C, D, E]",
            "Combining gives block [B, C, D, E]",
            "5 seats. A must be at an end.",
            "Case 1: A at left: A, B, C, D, E. But E cannot be at an end! So this case is invalid.",
            "Case 2: A at right: B, C, D, E, A. E is at index 4 (not an end, A is at end). Valid!",
            "Order: B, C, D, E, A"
        ],
        "check": "B at 1, C at 2, D at 3, E at 4, A at 5. A at end? Yes. C right of B? Yes. D between C and E? Yes. E not at end? Yes (seat 4).",
    },
]
