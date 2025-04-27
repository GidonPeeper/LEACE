# LEACE on GPT-2 Embeddings

This project applies the LEACE (Least-squares Concept Erasure) method to create modified versions of GPT-2 by removing specific linguistic features from its internal representations.  
The resulting models are fully compatible GPT-2 variants, but without certain encodable concepts — such as part-of-speech tags or syntactic dependencies — making them usable as controlled components in broader NLP pipelines.

Inspired by earlier work applying LEACE to BERT and brain alignment studies, this project focuses purely on modifying the model's internal representations without involving any neural data.

---

## Goals

- **Implement and validate LEACE** on synthetic toy examples.
- **Apply LEACE to GPT-2** embeddings on real English data.
- **Erase increasingly complex linguistic features**, starting from simple POS distinctions and moving towards full syntactic structures.
- **Maintain model usability**: minimize distortion to embeddings and preserve unrelated information.
- **Prepare plug-and-play GPT-2 variants** that can be inserted into larger language processing systems.

---

## 📋 Project Roadmap

### Stage 1: Toy Example Validation
- Generate synthetic embeddings encoding a simple binary concept.
- Apply LEACE and verify successful erasure via probing.

### Stage 2: Simple Feature Erasure (Real Data)
- Use English Universal Dependencies (UD) data.
- Erase simple linguistic concepts (e.g., function vs. content words, noun vs. non-noun).

### Stage 3: Full POS Tag Erasure
- Scale up to removing all POS categories (17+ classes).
- Evaluate probing accuracy before and after LEACE.

### Stage 4: Dependency Label Erasure
- Target syntactic dependencies using auto-annotated corpora.
- Verify erasure using structured probes or classifiers.

### Stage 5: Dependency Distance Erasure
- Tackle highly distributed features like syntactic tree distances.

---

