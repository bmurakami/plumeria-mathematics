# Plumeria Mathematics Instructions

## Project Preferences

- Do not use broadcasting semantics: do not implicitly expand scalars or lower-rank tensors across tensor axes for elementwise operations.
- Do not add scalar addition for tensors. Tensor-scalar multiplication and division are acceptable.
- Do not add Hadamard products unless there is a concrete physics use case.
- Reference implementations should be naive and math-like, resembling academic arithmetic rather than optimized storage algorithms.
- BLAS implementations should use BLAS-backed operations where BLAS behavior is the purpose of the implementation.
- Keep protocols storage-agnostic where practical. Put shared behavior at the highest clean protocol level without polluting real tensors or scalars with complex-only behavior.
- Prefer physics conventions for tensor multiplication: repeated indices contract; free indices remain in written order; do not add an explicit binary output clause.
- Keep tests human-checkable: small tensors, concrete values, simple arithmetic, and parameterized reference/BLAS coverage where appropriate.
- Use concise math-native API names where appropriate, such as `t`, `tr`, `det`, `inverse()`, `magnitude()`, `eigen()`, `star`, `mod`, and `arg`.
