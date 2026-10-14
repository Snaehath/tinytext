# 100 Held-Out Instruction Prompts across 6 Categories for TinyText Flash V1.0 Evaluation

INSTRUCTION_BENCHMARK_100 = [
    # 1. Question Answering (17 prompts)
    {"id": 1, "category": "Question Answering", "prompt": "What is the primary function of red blood cells in the human body?", "reference": "To transport oxygen from the lungs to body tissues and carry carbon dioxide back to the lungs."},
    {"id": 2, "category": "Question Answering", "prompt": "Who was the first person to walk on the Moon, and in what year did this occur?", "reference": "Neil Armstrong in 1969 during the Apollo 11 mission."},
    {"id": 3, "category": "Question Answering", "prompt": "What are the three primary states of matter commonly observed in daily life?", "reference": "Solid, liquid, and gas."},
    {"id": 4, "category": "Question Answering", "prompt": "What is the capital city of Japan?", "reference": "Tokyo."},
    {"id": 5, "category": "Question Answering", "prompt": "What causes the ocean tides on Earth?", "reference": "The gravitational pull of the Moon and the Sun on Earth's oceans."},
    {"id": 6, "category": "Question Answering", "prompt": "In what continent is the Sahara Desert located?", "reference": "Africa."},
    {"id": 7, "category": "Question Answering", "prompt": "What is the chemical symbol for table salt?", "reference": "NaCl (sodium chloride)."},
    {"id": 8, "category": "Question Answering", "prompt": "How many continents are there on Earth, and can you name three?", "reference": "There are 7 continents (e.g., Africa, Asia, Europe)."},
    {"id": 9, "category": "Question Answering", "prompt": "What is the boiling point of pure water at sea level in degrees Celsius?", "reference": "100 degrees Celsius."},
    {"id": 10, "category": "Question Answering", "prompt": "Who wrote the play Romeo and Juliet?", "reference": "William Shakespeare."},
    {"id": 11, "category": "Question Answering", "prompt": "What is the largest internal organ in the human body?", "reference": "The liver."},
    {"id": 12, "category": "Question Answering", "prompt": "What gas do plants absorb from the atmosphere for photosynthesis?", "reference": "Carbon dioxide (CO2)."},
    {"id": 13, "category": "Question Answering", "prompt": "Which planet is known as the Red Planet?", "reference": "Mars."},
    {"id": 14, "category": "Question Answering", "prompt": "What is the hardest known natural mineral on Earth?", "reference": "Diamond."},
    {"id": 15, "category": "Question Answering", "prompt": "In what year did World War II end?", "reference": "1945."},
    {"id": 16, "category": "Question Answering", "prompt": "What instrument is used to measure atmospheric pressure?", "reference": "A barometer."},
    {"id": 17, "category": "Question Answering", "prompt": "What is the currency used in the United Kingdom?", "reference": "Pound sterling (GBP)."},

    # 2. Summarization & Extraction (17 prompts)
    {"id": 18, "category": "Summarization & Extraction", "prompt": "Summarize this passage in one sentence: The Amazon Basin contains the largest tropical rainforest in the world. It covers over two million square miles and is home to roughly ten percent of all known species on Earth. Deforestation has significantly threatened this rich biodiversity.", "reference": "The Amazon rainforest is a vast, biodiverse ecosystem covering over two million square miles whose rich species are threatened by deforestation."},
    {"id": 19, "category": "Summarization & Extraction", "prompt": "Extract the meeting time and location from this note: Team, our weekly sync will happen at 3:30 PM on Thursday in Conference Room B. Please bring your updated reports.", "reference": "Time: 3:30 PM on Thursday; Location: Conference Room B."},
    {"id": 20, "category": "Summarization & Extraction", "prompt": "Summarize the key benefit mentioned in this text: Regular cardiovascular exercise improves heart health, lowers resting blood pressure, and helps maintain a healthy body weight over time.", "reference": "Cardiovascular exercise enhances heart health, reduces blood pressure, and supports weight management."},
    {"id": 21, "category": "Summarization & Extraction", "prompt": "Extract the names of the two founders from this statement: In 1976, Steve Jobs and Steve Wozniak founded Apple Computer in a garage in Los Altos, California.", "reference": "Steve Jobs and Steve Wozniak."},
    {"id": 22, "category": "Summarization & Extraction", "prompt": "Provide a two-bullet summary of this passage: Solar panels convert sunlight directly into clean electricity through the photovoltaic effect. While the initial installation cost can be high, panels typically pay for themselves within five to ten years through reduced utility bills.", "reference": "- Solar panels produce clean energy from sunlight via the photovoltaic effect.\n- High initial costs are typically recouped within 5 to 10 years through utility savings."},
    {"id": 23, "category": "Summarization & Extraction", "prompt": "Extract the three ingredients listed: To make simple syrup, gently heat equal parts granulated sugar and filtered water, then add a drop of vanilla extract.", "reference": "Granulated sugar, filtered water, vanilla extract."},
    {"id": 24, "category": "Summarization & Extraction", "prompt": "Summarize the main problem in this message: The shipping carrier notified us that blizzard conditions have blocked the mountain pass, causing all regional deliveries to be delayed by at least 48 hours.", "reference": "Severe blizzard conditions have blocked roads, delaying regional shipments by at least 48 hours."},
    {"id": 25, "category": "Summarization & Extraction", "prompt": "Extract the phone number and email address from: For assistance, please call 555-0199 or email support@tinytext.org.", "reference": "Phone: 555-0199; Email: support@tinytext.org."},
    {"id": 26, "category": "Summarization & Extraction", "prompt": "Condense this description into a short title: A comprehensive historical examination of the architectural styles used in medieval European cathedrals between the 11th and 14th centuries.", "reference": "Medieval European Cathedral Architecture (11th-14th Centuries)."},
    {"id": 27, "category": "Summarization & Extraction", "prompt": "Extract the project deadline from this message: We are progressing well with the client audit, but remember that the final audit package must be submitted to compliance before Friday at 5:00 PM.", "reference": "Deadline: Friday at 5:00 PM."},
    {"id": 28, "category": "Summarization & Extraction", "prompt": "Summarize the conclusion of this study: After monitoring 500 participants over five years, researchers concluded that getting seven to eight hours of sleep nightly significantly reduces the risk of cardiovascular disease.", "reference": "Getting 7 to 8 hours of sleep per night substantially lowers cardiovascular disease risk."},
    {"id": 29, "category": "Summarization & Extraction", "prompt": "Extract the price and model number from: The UltraTab 400X is currently on sale for $299, down from the standard retail price of $399.", "reference": "Model: UltraTab 400X; Price: $299."},
    {"id": 30, "category": "Summarization & Extraction", "prompt": "Summarize in one sentence why honey does not spoil: Honey has extremely low moisture content and high acidity, creating an inhospitable environment where bacteria and microorganisms cannot survive.", "reference": "Honey does not spoil because its low moisture and high acidity prevent bacterial growth."},
    {"id": 31, "category": "Summarization & Extraction", "prompt": "Extract the author and publication year: According to the study published by Dr. Elena Ramos in 2021, urban green spaces lower local ambient temperatures.", "reference": "Author: Dr. Elena Ramos; Year: 2021."},
    {"id": 32, "category": "Summarization & Extraction", "prompt": "Summarize the main argument: Electric bicycles offer a sustainable urban transit alternative because they combine low operational energy costs with reduced traffic congestion and zero direct tailpipe emissions.", "reference": "Electric bicycles provide an eco-friendly urban transit solution with zero direct emissions and less traffic congestion."},
    {"id": 33, "category": "Summarization & Extraction", "prompt": "Extract the warranty period mentioned: The Apex Blender comes with a full 3-year limited warranty covering motor defects and mechanical failures.", "reference": "Warranty: 3-year limited warranty."},
    {"id": 34, "category": "Summarization & Extraction", "prompt": "Summarize the primary caution in this lab manual: Always pour acid slowly into water rather than water into acid, as adding water to concentrated acid can cause violent splattering due to extreme heat release.", "reference": "Always add acid slowly to water to prevent dangerous splattering from rapid heat release."},

    # 3. Rewriting & Writing Assistance (17 prompts)
    {"id": 35, "category": "Rewriting & Writing Assistance", "prompt": "Rewrite this sentence to sound more formal: Hey, we gotta talk about the budget real soon because we're running out of cash.", "reference": "We need to discuss our budget promptly, as our available funds are nearly depleted."},
    {"id": 36, "category": "Rewriting & Writing Assistance", "prompt": "Make this customer service apology concise and polite: We are super sorry about the big delay with your package, it was totally our fault and we feel terrible about it.", "reference": "We sincerely apologize for the delay with your shipment and appreciate your patience."},
    {"id": 37, "category": "Rewriting & Writing Assistance", "prompt": "Rewrite in the active voice: The quarterly financial report was prepared by the accounting department.", "reference": "The accounting department prepared the quarterly financial report."},
    {"id": 38, "category": "Rewriting & Writing Assistance", "prompt": "Simplify this sentence for an elementary school student: Terrestrial precipitation encompasses atmospheric water vapor condensation manifesting as rain, sleet, or snow.", "reference": "Precipitation means water falling from the sky as rain, sleet, or snow."},
    {"id": 39, "category": "Rewriting & Writing Assistance", "prompt": "Rewrite this email opening to be more engaging and professional: I am writing this email to let you know that our team finished the draft.", "reference": "I am pleased to share that our team has completed the draft for your review."},
    {"id": 40, "category": "Rewriting & Writing Assistance", "prompt": "Proofread and correct grammatical errors: She don't have no time to attend the meeting yesterday because her flight was delay.", "reference": "She did not have time to attend the meeting yesterday because her flight was delayed."},
    {"id": 41, "category": "Rewriting & Writing Assistance", "prompt": "Condense this wordy phrase into one word: At the present point in time.", "reference": "Currently (or Now)."},
    {"id": 42, "category": "Rewriting & Writing Assistance", "prompt": "Rewrite this positive feedback to sound enthusiastic yet professional: Good job on the project, it was fine.", "reference": "Outstanding work on this project; your contributions made a significant positive impact."},
    {"id": 43, "category": "Rewriting & Writing Assistance", "prompt": "Rewrite this negative feedback diplomatically: Your presentation was boring and had way too much text on every slide.", "reference": "The presentation contains valuable insights; reducing text on the slides would make it even more engaging."},
    {"id": 44, "category": "Rewriting & Writing Assistance", "prompt": "Change this sentence from informal to academic tone: A bunch of scientists tried this experiment and found out it worked pretty well.", "reference": "Researchers conducted this experiment and demonstrated statistically significant positive results."},
    {"id": 45, "category": "Rewriting & Writing Assistance", "prompt": "Rewrite as a polite decline to an invitation: I can't come to your party on Saturday.", "reference": "Thank you so much for the invitation, but unfortunately I will not be able to attend this Saturday."},
    {"id": 46, "category": "Rewriting & Writing Assistance", "prompt": "Eliminate redundancies: The end result was a completely unanimous decision agreed upon by all members.", "reference": "The result was a unanimous decision by all members."},
    {"id": 47, "category": "Rewriting & Writing Assistance", "prompt": "Rewrite to emphasize speed: We will deliver your package as soon as we possibly can.", "reference": "We will expedite your shipment for the earliest possible delivery."},
    {"id": 48, "category": "Rewriting & Writing Assistance", "prompt": "Convert to a question: You should restart your computer to apply the security updates.", "reference": "Could you please restart your computer to apply the security updates?"},
    {"id": 49, "category": "Rewriting & Writing Assistance", "prompt": "Rewrite to remove passive aggressive tone: As I already told you three times, the link is in the email.", "reference": "For your convenience, the link is included in the previous email."},
    {"id": 50, "category": "Rewriting & Writing Assistance", "prompt": "Rewrite for a social media caption (under 15 words): Our local bakery has officially opened its doors for the spring season with fresh pastries.", "reference": "Fresh spring pastries are here! Our bakery doors are officially open. 🥐✨"},
    {"id": 51, "category": "Rewriting & Writing Assistance", "prompt": "Make this technical sentence accessible: The operational velocity was attenuated by thermal dissipation constraints.", "reference": "The machine slowed down to prevent overheating."},

    # 4. Classification & Formatting (17 prompts)
    {"id": 52, "category": "Classification & Formatting", "prompt": "Classify the sentiment of this review as Positive, Neutral, or Negative: The battery lasted only two hours and the screen flickered constantly.", "reference": "Negative."},
    {"id": 53, "category": "Classification & Formatting", "prompt": "Format the following data as a clean JSON object with keys 'city', 'country', and 'population': Tokyo, Japan, 14 million.", "reference": "{\"city\": \"Tokyo\", \"country\": \"Japan\", \"population\": \"14 million\"}"},
    {"id": 54, "category": "Classification & Formatting", "prompt": "Classify this email as either 'Spam' or 'Legitimate': Dear user, you have won $5,000,000 in the international lottery! Click here now to claim your funds.", "reference": "Spam."},
    {"id": 55, "category": "Classification & Formatting", "prompt": "Format the following names into an alphabetical bulleted list: Charlie, Alice, David, Bob.", "reference": "- Alice\n- Bob\n- Charlie\n- David"},
    {"id": 56, "category": "Classification & Formatting", "prompt": "Classify the topic of this article as Science, Sports, or Politics: The spacecraft entered lunar orbit after executing a four-minute thruster burn.", "reference": "Science."},
    {"id": 57, "category": "Classification & Formatting", "prompt": "Format this address onto three standard lines: John Doe, 123 Maple Street, Springfield, IL 62701.", "reference": "John Doe\n123 Maple Street\nSpringfield, IL 62701"},
    {"id": 58, "category": "Classification & Formatting", "prompt": "Classify whether this statement is 'Fact' or 'Opinion': Chocolate ice cream tastes far superior to vanilla.", "reference": "Opinion."},
    {"id": 59, "category": "Classification & Formatting", "prompt": "Format these key-value pairs as a Markdown table with columns 'Metric' and 'Value': Revenue: $50M, Growth: 12%, Churn: 2%.", "reference": "| Metric | Value |\n| --- | --- |\n| Revenue | $50M |\n| Growth | 12% |\n| Churn | 2% |"},
    {"id": 60, "category": "Classification & Formatting", "prompt": "Classify the urgency of this ticket as Low, Medium, or High: The production database is completely down and customers cannot check out.", "reference": "High."},
    {"id": 61, "category": "Classification & Formatting", "prompt": "Output only the integer count of words in this sentence: 'Ants work together to build complex underground colonies.'", "reference": "8"},
    {"id": 62, "category": "Classification & Formatting", "prompt": "Classify whether this animal is a Mammal, Bird, or Reptile: Penguin.", "reference": "Bird."},
    {"id": 63, "category": "Classification & Formatting", "prompt": "Format this list of dates in chronological order: March 15 2024, January 1 2024, December 31 2023.", "reference": "1. December 31, 2023\n2. January 1, 2024\n3. March 15, 2024"},
    {"id": 64, "category": "Classification & Formatting", "prompt": "Classify this statement as True or False: All prime numbers are odd.", "reference": "False (2 is an even prime number)."},
    {"id": 65, "category": "Classification & Formatting", "prompt": "Extract the keywords and output them separated by commas: The algorithm optimizes memory allocation in distributed high-performance computing clusters.", "reference": "algorithm, memory allocation, distributed computing, high-performance computing, clusters"},
    {"id": 66, "category": "Classification & Formatting", "prompt": "Classify the tone of this letter as Friendly, Angry, or Formal: Dear Sir or Madam, I am submitting my formal curriculum vitae for your consideration regarding the senior analyst position.", "reference": "Formal."},
    {"id": 67, "category": "Classification & Formatting", "prompt": "Convert this sentence to uppercase: safety first in the workshop.", "reference": "SAFETY FIRST IN THE WORKSHOP."},
    {"id": 68, "category": "Classification & Formatting", "prompt": "Classify the following query intent as Informational, Navigational, or Transactional: 'Buy wireless noise cancelling headphones online.'", "reference": "Transactional."},

    # 5. Explanations & Procedures (16 prompts)
    {"id": 69, "category": "Explanations & Procedures", "prompt": "Explain why the sky appears red or orange during sunset in two simple sentences.", "reference": "At sunset, sunlight travels through more of the atmosphere, scattering shorter blue wavelengths away. Longer red and orange wavelengths pass through to our eyes."},
    {"id": 70, "category": "Explanations & Procedures", "prompt": "Provide a 3-step guide on how to brew green tea properly.", "reference": "1. Heat water to approximately 80°C (175°F), not boiling.\n2. Steep the tea leaves for 2 to 3 minutes.\n3. Strain the leaves immediately and serve."},
    {"id": 71, "category": "Explanations & Procedures", "prompt": "Explain what inflation means in economics using a brief everyday example.", "reference": "Inflation is the general rise in prices over time, meaning money loses purchasing power (e.g., if a loaf of bread costs $2 this year and $2.20 next year)."},
    {"id": 72, "category": "Explanations & Procedures", "prompt": "List the four fundamental steps to wash your hands effectively according to health guidelines.", "reference": "1. Wet hands with clean water and apply soap.\n2. Lather and scrub thoroughly for at least 20 seconds.\n3. Rinse thoroughly under running water.\n4. Dry with a clean towel."},
    {"id": 73, "category": "Explanations & Procedures", "prompt": "Briefly explain the greenhouse effect and its role in warming the Earth.", "reference": "Gases in Earth's atmosphere trap heat radiating from the planet's surface, keeping the climate warm enough to sustain life."},
    {"id": 74, "category": "Explanations & Procedures", "prompt": "Provide two clear instructions for jump-starting a car battery with booster cables.", "reference": "1. Connect the red clamp to the positive terminal of both batteries.\n2. Connect the black clamp to the working battery's negative terminal and the other to an unpainted metal surface on the stalled car."},
    {"id": 75, "category": "Explanations & Procedures", "prompt": "Explain why metals are generally good conductors of electricity.", "reference": "Metals have free, delocalized electrons that move easily through the atomic lattice, carrying electrical current."},
    {"id": 76, "category": "Explanations & Procedures", "prompt": "Outline a 3-step checklist for backing up important personal files.", "reference": "1. Identify critical files and documents.\n2. Copy them to an external drive or secure cloud storage.\n3. Verify the backup by opening test files."},
    {"id": 77, "category": "Explanations & Procedures", "prompt": "Explain the difference between weather and climate in one clear comparison.", "reference": "Weather describes short-term atmospheric conditions day-to-day, while climate represents long-term averages over decades."},
    {"id": 78, "category": "Explanations & Procedures", "prompt": "How do sound waves travel from an instrument to human ears? Explain briefly.", "reference": "Vibrations create pressure waves in the air, which travel outward and vibrate the eardrum inside the ear."},
    {"id": 79, "category": "Explanations & Procedures", "prompt": "Provide three steps to store fresh bread so it stays fresh longer.", "reference": "1. Keep in a paper bag or bread box at room temperature.\n2. Avoid refrigerating, which accelerates staling.\n3. Freeze sliced portions for long-term storage."},
    {"id": 80, "category": "Explanations & Procedures", "prompt": "Explain how a thermos keeps liquids hot or cold.", "reference": "A thermos uses a double-walled container with a vacuum between the walls, preventing heat transfer by conduction and convection."},
    {"id": 81, "category": "Explanations & Procedures", "prompt": "Describe the basic scientific purpose of a control group in an experiment.", "reference": "A control group serves as a baseline comparison by remaining untreated, isolating the effect of the experimental variable."},
    {"id": 82, "category": "Explanations & Procedures", "prompt": "What should you do if an oil pan catches fire while cooking? List two safety actions.", "reference": "1. Turn off the heat source immediately.\n2. Slide a metal lid over the pan to smother the flames (never use water)."},
    {"id": 83, "category": "Explanations & Procedures", "prompt": "Explain why bicycles balance more easily when moving forward than when standing still.", "reference": "Forward momentum allows the rider to make small steering corrections that keep the center of gravity over the contact points."},
    {"id": 84, "category": "Explanations & Procedures", "prompt": "Provide 3 simple steps to organize a messy computer desktop.", "reference": "1. Delete duplicate or temporary files.\n2. Group remaining documents into categorized folders.\n3. Move frequent apps to the taskbar or dock."},

    # 6. Multi-turn Dialogue & Follow-ups (16 prompts)
    {"id": 85, "category": "Multi-turn Dialogue", "messages": [
        {"role": "user", "content": "I'm planning a weekend trip to a mountain cabin. What are three essential things to pack?"},
        {"role": "assistant", "content": "1. Warm layered clothing and sturdy waterproof boots.\n2. A reliable flashlight or headlamp with extra batteries.\n3. A well-stocked first aid kit and essential medications."},
        {"role": "user", "content": "What about food? What kind of food should I bring if the cabin has no electricity?"}
    ], "reference": "Non-perishable items like canned goods, nuts, dried fruit, granola bars, and plenty of drinking water."},
    {"id": 86, "category": "Multi-turn Dialogue", "messages": [
        {"role": "user", "content": "What is Python in programming?"},
        {"role": "assistant", "content": "Python is a high-level, interpreted programming language known for its clear syntax, readability, and versatile libraries."},
        {"role": "user", "content": "Can you give me an example of what it is commonly used for?"}
    ], "reference": "It is widely used for web development, data analysis, artificial intelligence, and automated scripting."},
    {"id": 87, "category": "Multi-turn Dialogue", "messages": [
        {"role": "user", "content": "Name two carnivorous mammals."},
        {"role": "assistant", "content": "Lions and wolves."},
        {"role": "user", "content": "Which one of them lives primarily in Africa?"}
    ], "reference": "The lion lives primarily in Africa."},
    {"id": 88, "category": "Multi-turn Dialogue", "messages": [
        {"role": "user", "content": "I want to start running for exercise. How long should my first run be?"},
        {"role": "assistant", "content": "For a beginner, start with 20 to 30 minutes of alternating between 1 minute of easy jogging and 2 minutes of walking."},
        {"role": "user", "content": "How many days a week should I do this?"}
    ], "reference": "Aim for 2 to 3 days per week with rest days in between to allow muscle recovery."},
    {"id": 89, "category": "Multi-turn Dialogue", "messages": [
        {"role": "user", "content": "What is a solar eclipse?"},
        {"role": "assistant", "content": "A solar eclipse occurs when the Moon passes directly between the Sun and Earth, blocking sunlight from reaching Earth."},
        {"role": "user", "content": "Is it safe to look directly at it with bare eyes?"}
    ], "reference": "No, looking directly at a solar eclipse without certified eclipse glasses can cause permanent eye damage."},
    {"id": 90, "category": "Multi-turn Dialogue", "messages": [
        {"role": "user", "content": "Give me a synonym for the word 'ubiquitous'."},
        {"role": "assistant", "content": "'Omnipresent' or 'pervasive'."},
        {"role": "user", "content": "Use 'omnipresent' in a short sentence."}
    ], "reference": "Smartphones have become omnipresent in modern daily life."},
    {"id": 91, "category": "Multi-turn Dialogue", "messages": [
        {"role": "user", "content": "I have leftover roasted chicken in my fridge. What is a quick meal I can make with it?"},
        {"role": "assistant", "content": "You can shred the chicken to make a quick chicken salad sandwich with mayonnaise and celery."},
        {"role": "user", "content": "What if I want something warm instead?"}
    ], "reference": "You can make a quick chicken quesadilla by warming shredded chicken and cheese in a tortilla on a skillet."},
    {"id": 92, "category": "Multi-turn Dialogue", "messages": [
        {"role": "user", "content": "What is the capital of Canada?"},
        {"role": "assistant", "content": "Ottawa."},
        {"role": "user", "content": "What province is it located in?"}
    ], "reference": "Ottawa is located in the province of Ontario."},
    {"id": 93, "category": "Multi-turn Dialogue", "messages": [
        {"role": "user", "content": "Define what a carnivore is."},
        {"role": "assistant", "content": "A carnivore is an organism that derives its energy and nutrient requirements primarily from consuming animal tissue."},
        {"role": "user", "content": "What is an animal that eats both plants and animals called?"}
    ], "reference": "An omnivore."},
    {"id": 94, "category": "Multi-turn Dialogue", "messages": [
        {"role": "user", "content": "Suggest a good name for a cozy local bookstore."},
        {"role": "assistant", "content": "The Book Nook, Chapter & Cozy, or The Whispering Spine."},
        {"role": "user", "content": "I like 'The Book Nook'. Give me a matching tagline for it."}
    ], "reference": "'Your quiet corner for stories and coffee.'"},
    {"id": 95, "category": "Multi-turn Dialogue", "messages": [
        {"role": "user", "content": "What does CPU stand for?"},
        {"role": "assistant", "content": "Central Processing Unit."},
        {"role": "user", "content": "What is its main job in a computer?"}
    ], "reference": "It executes instructions and coordinates calculations as the primary processing brain of the computer."},
    {"id": 96, "category": "Multi-turn Dialogue", "messages": [
        {"role": "user", "content": "Who painted the Mona Lisa?"},
        {"role": "assistant", "content": "Leonardo da Vinci."},
        {"role": "user", "content": "In which museum is it displayed today?"}
    ], "reference": "The Louvre Museum in Paris, France."},
    {"id": 97, "category": "Multi-turn Dialogue", "messages": [
        {"role": "user", "content": "I want to adopt a small pet that doesn't need to be walked outside. Any ideas?"},
        {"role": "assistant", "content": "Cats, guinea pigs, and indoor rabbits are great pets that do not require outdoor walks."},
        {"role": "user", "content": "Which of those is the quietest?"}
    ], "reference": "Rabbits and guinea pigs are generally very quiet compared to other pets."},
    {"id": 98, "category": "Multi-turn Dialogue", "messages": [
        {"role": "user", "content": "What is the boiling point of water in Fahrenheit?"},
        {"role": "assistant", "content": "212 degrees Fahrenheit."},
        {"role": "user", "content": "And what is its freezing point in Fahrenheit?"}
    ], "reference": "32 degrees Fahrenheit."},
    {"id": 99, "category": "Multi-turn Dialogue", "messages": [
        {"role": "user", "content": "Give me a quick tip to improve my public speaking."},
        {"role": "assistant", "content": "Pause deliberately between key thoughts instead of using filler words like 'um' or 'ah'."},
        {"role": "user", "content": "How can I calm my nerves right before going on stage?"}
    ], "reference": "Take three deep, slow belly breaths and focus on delivering value to your audience rather than on yourself."},
    {"id": 100, "category": "Multi-turn Dialogue", "messages": [
        {"role": "user", "content": "What is the difference between an asteroid and a comet?"},
        {"role": "assistant", "content": "Asteroids are composed primarily of rock and metal, while comets are composed of ice, dust, and rock, forming a visible tail when near the Sun."},
        {"role": "user", "content": "Where do most asteroids in our solar system live?"}
    ], "reference": "In the asteroid belt between the orbits of Mars and Jupiter."}
]
