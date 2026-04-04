# Next Steps

Derived from a code review of the variable scoping implementation (commit `c0abb5f`).

---

## 1. Reduce Per-Lookup Allocation in Eval — DONE

**Completed in commit `af8e27b` on branch `claude/review-todos-refactor-paths-7WgEn`.**

All `String` allocations were eliminated from the evaluator's hot paths:

- `resolve_path` was removed entirely — replaced by a free function `path_key()` that returns the `Sym` key directly for single-segment paths (zero allocation) and FNV-combines segment keys for multi-segment paths
- `call_fn` now looks up functions by `u64` key instead of `String` (`HashMap<String, Ptr<Fn>>` → `HashMap<u64, Ptr<Fn>>`)
- `Binding::name` (dead field — written but never read) was removed, simplifying `declare()` to `(key: u64, value: Value)`
- `symt: RefCell<SymTable>` was removed from `Eval` — no longer needed
- Error-path strings are constructed lazily via `path_to_string()` only when an error actually occurs
