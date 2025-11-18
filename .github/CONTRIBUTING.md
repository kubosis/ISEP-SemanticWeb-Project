# 🎓 Contributing Guide

Welcome! We're excited that you're interested in contributing to our Study Path Knowledge Graph project. This guide will help you get set up.

## Getting Started

1.  **Clone** the repository.
2.  Switch to the `devel` branch: `git checkout devel`
3.  Create a **new local branch** based on `devel`: `git checkout -b your-feature-or-fix-name`
4.  Start working on an **assigned issue**.

## Adding a New Feature or Fixing an Issue

1.  Review the list of open **[Issues](httpsloc:link_to_issues)**, or create a new issue if one doesn't exist.
2.  **Create a branch** from the `devel` branch (see above).
    * Name the branch clearly, ideally including the issue number (e.g., `feat/12-add-skill-ontology`).
    * **Do not** make changes directly on `devel`.
3.  Implement your changes on the new branch.
4.  Once complete, open a **Pull Request (PR)** into the `devel` branch.
    * Provide a short description of what your PR addresses and link the issue it resolves.

---

## Project Structure
```
/ 
├── .github/                                     # CI/CD workflows, issue templates, and this guide 
├── app/                                         # Basic Web application for students 
│ ├── frontend/                                  # Frontend code (e.g., React, Vue) 
│ └── backend/                                   # API server & SPARQL query endpoint written in python+fastapi
├── knowledge_graph/                             # The core Knowledge Graph files 
│ ├── ontology/                                  # .ttl or .owl files defining the schema 
│ │   └── career_atlas.ttl                       # Our custom definitions
│ ├── data/                                      # .rdf, .ttl, .jsonld instance data 
│ ├── raw_data/                                  # folder to store raw data && partially cleaned data
│ ├── mappings/                                  # <--- CSVs or JSONs mapping raw terms to URIs
│ │  ├── skills_map.json                         # e.g.: "machine learning" -> <http://data.europa.eu/esco/skill/...>
│ │  └── course_map.json
│ └── queries/                                   # .sparql files for reusable queries 
├── scripts/                                     # Data ingestion, ETL, and validation scripts 
├── .gitignore \
└── README.md\
```

---

## Commits

All commits and PR titles must follow the **Conventional Commits** specification. This helps us automate changelogs and makes the history readable.

### Commit Types

Your commit message must start with one of the following types:

```yaml
# Allowed semantic types (from Conventional Commits)
types:
  - feat      #  A new feature (e.g., a new API endpoint, a new ontology class)
  - fix       #  A bug fix (e.g., correcting a SPARQL query)
  - docs      #  Documentation changes only
  - style     #  Code style/formatting (no logic changes)
  - refactor  # ️ Code restructuring (no behavior change)
  - perf      #  Performance improvements
  - test      #  Adding or updating tests
  - build     #  Build system or dependencies
  - ci        #  CI/CD configuration changes
  - chore     #  Maintenance tasks (e.g., cleaning data)
  - lab       # lab commits
  - ontology  # commits to enhance the knowledge graph
```

Example
After adding a new property to the course ontology, your commit might look like this:
```sh
git commit -m "feat(ontology): add 'hasPrerequisite' property to Course class"
```
## Knowledge Graph Contributions
This is the core of our project. Contributions here are highly valued and fall into three categories:

### 1. Ontology (knowledge_graph/ontology/)
This is the schema or "data model" of our graph. It defines the classes (like Course, Skill, Career) and the properties (like requiresSkill, leadsToCareer).

Format: Please use Turtle (.ttl).

Discuss First: Changes to the ontology are significant. Please open an issue to discuss any proposed changes before starting work.

Standards: Use existing vocabularies (RDFS, OWL, SKOS) where possible.

Documentation: All new classes and properties must have an rdfs:label and rdfs:comment.

### 2. Data (knowledge_graph/data/)
This is the instance data that populates our graph (e.g., specific courses from a university, specific skills for a "Data Scientist" role).

Format: Turtle (.ttl) or JSON-LD (.jsonld) are preferred.

Validation: All new data must validate against our ontology.

Provenance: If you are adding data from an external source, please document where it came from in the PR or as comments in the data file.

Scripts: If you are adding a large amount of data, consider writing an ingestion script and placing it in the scripts/ folder.

### 3. Queries (knowledge_graph/queries/)
These are the SPARQL queries that our backend application uses to find study paths.

Format: Save files as .sparql.

Clarity: Use clear file names (e.g., get_courses_for_skill.sparql).

Documentation: Include comments in the .sparql file explaining what it does, what variables it expects (if any), and what it returns.