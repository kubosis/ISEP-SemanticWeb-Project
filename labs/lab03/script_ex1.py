import spacy
import re

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Input text
text = """
Title: Introduction to Programming (CS101)
Description:
This course introduces the fundamental concepts of programming and problem-solving using a high-level language. 
Topics include variables, data types, control structures (conditionals and loops), functions, and basic algorithms. 
Students will learn to design, code, test, and debug simple programs. 
This is a foundational requirement for the Computer Science major. 
There are no prerequisites for this course.

Title: Introduction to Data Structures (CS201)
Description:
This course provides a comprehensive overview of fundamental data structures. 
Topics include arrays, linked lists, stacks, queues, trees, and graphs. 
Students will learn to analyze time and space complexity. 
This is a core requirement for the Computer Science major. 
The prerequisite for this course is CS101: Introduction to Programming.
"""

# Split by "Title:"
courses = [c.strip() for c in text.split("Title:") if c.strip()]
triples = []

for course in courses:
    lines = course.split("\n")
    title_line = lines[0]
    title, code = title_line.rsplit("(", 1)
    code = code.strip(")")
    title = title.strip()
    description = " ".join(line.strip() for line in lines[2:] if line.strip())
    doc = nlp(description)

    subject = f"uni:{code}"

    # Base triples
    triples.append((subject, "a", "s2a:Course"))
    triples.append((subject, "s2a:title", f'"{title}"'))
    triples.append((subject, "s2a:courseCode", f'"{code}"'))
    triples.append((subject, "s2a:description", f'"{description}"'))

    # 1️⃣ Summary (first sentence)
    if len(list(doc.sents)) > 0:
        summary = list(doc.sents)[0].text.strip()
        triples.append((subject, "s2a:summary", f'"{summary}"'))

    # 2️⃣ Prerequisites
    if "no prerequisites" in description.lower():
        triples.append((subject, "s2a:hasPrerequisite", '"None"'))
    elif "prerequisite" in description.lower():
        match = re.search(r"CS\d+", description)
        if match:
            triples.append((subject, "s2a:hasPrerequisite", f"uni:{match.group(0)}"))

    # 3️⃣ Topics
    if "topics include" in description.lower():
        topics_text = re.split(r"topics include", description, flags=re.IGNORECASE)[1]
        topics_text = topics_text.split(".")[0]
        topics = [t.strip() for t in re.split(r",|and", topics_text) if t.strip()]
        for topic in topics:
            triples.append((subject, "s2a:hasTopic", f'"{topic}"'))

    # 4️⃣ Requirement for major
    if "requirement for" in description.lower():
        match = re.search(r"requirement for the (.+?) major", description, re.IGNORECASE)
        if match:
            major = match.group(1).strip().title().replace(" ", "")
            triples.append((subject, "s2a:requirementFor", f"uni:{major}Major"))

    # 5️⃣ Learning outcomes
    learn_matches = re.findall(r"Students will (?:learn to|be able to|learn how to) (.+?)(?:\.|;)", description, re.IGNORECASE)
    for outcome in learn_matches:
        clean_outcome = outcome.strip().capitalize()
        triples.append((subject, "s2a:learningOutcome", f'"{clean_outcome}"'))

# Output RDF triples in Turtle syntax
print("@prefix s2a: <http://example.org/s2a#> .")
print("@prefix uni: <http://example.org/university#> .\n")

for s, p, o in triples:
    print(f"{s} {p} {o} .")
