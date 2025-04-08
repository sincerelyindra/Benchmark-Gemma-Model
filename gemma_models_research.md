# Gemma Models Research

## Gemma Model Variants

### Gemma 3
- Parameter sizes: 1B, 4B, 12B, 27B
- Context window: 128K tokens
- Multilingual capabilities: Support for over 140 languages
- Multimodal capabilities: Can understand text, images, and video
- Based on Gemini 2.0 technology
- Intended platforms:
  - 1B: Mobile devices and single board computers
  - 4B: Desktop computers and small servers
  - 12B: Desktop computers and small servers
  - 27B: Higher-end desktop computers and servers

### Gemma 2
- Parameter sizes: 2B, 9B, 27B
- Based on the same research and technology as Gemini models
- Intended platforms:
  - 2B: Mobile devices and laptops
  - 9B: Higher-end desktop computers and servers
  - 27B: Higher-end desktop computers and servers

### Gemma 1
- Parameter sizes: 2B, 7B
- Intended platforms:
  - 2B: Mobile devices and laptops
  - 7B: Desktop computers and small servers

### Specialized Variants
- CodeGemma: Specialized for code generation and understanding
- PaliGemma 2: Multimodal capabilities (text + images)
- ShieldGemma 2: Enhanced safety features
- TxGemma: Specialized for therapeutic applications

## Academic Benchmarks

### MMLU (Massive Multitask Language Understanding)
- Comprehensive benchmark with 57 subject areas
- Multiple-choice questions format
- Subjects include math, history, law, ethics, science, etc.
- Evaluates broad knowledge and understanding across diverse domains
- Scoring: 0-1 scale based on exact matching of answers
- Supports few-shot learning (typically 5-shot)
- Good at detecting areas where a model lacks understanding in specific topics

### GSM8K (Grade School Math 8K)
- Dataset of 8.5K grade school math word problems
- Evaluates multi-step mathematical reasoning capabilities
- Problems require step-by-step solutions
- Enhanced version: MR-GSM8K (Meta-Reasoning GSM8K)
  - Evaluates meta-reasoning capabilities
  - Tasks models to predict correctness of solutions
  - Requires identifying error locations and explaining reasoning
  - Includes MR-Score metric: weighted combination of solution correctness, error step identification, and error reason explanation
  - Includes variations requiring code solutions and backward reasoning

### Other Notable Benchmarks (for potential inclusion)
- MHPP (Multi-step Human Preference Problems)
- LogiQA (Logical reasoning)
- BIG-Bench (Beyond the Imitation Game Benchmark)
- HumanEval (Code generation and correctness)
- HELM (Holistic Evaluation of Language Models)
