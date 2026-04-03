# Next Steps

One focused work item derived from a code review of the variable scoping implementation (commit `c0abb5f`).

---

## 1. Reduce Per-Lookup Allocation in Eval

**Goal:** Eliminate the `String` allocation that happens on every variable read, write, and assignment.

**The problem:**
- `resolve_path` (`src/eval.rs:64-73`) builds a `String` via `format!("{}::{}", ...)` and calls `self.symt.borrow().make(full_name)` on every single variable access
- `SymTable::make` hashes the string and potentially inserts into the table
- For local variables (single-segment paths like `x`), this is unnecessary — the parser already produced a `Sym` with the correct key

**The fix:**
- For single-segment paths (`path.list.len() == 1`), read `path.list[0].name.key()` directly — no `String`, no `make()`, no hash
- Only fall back to the current `resolve_path` logic for multi-segment paths (e.g. `foo::bar`)
- This may allow removing the `symt: RefCell<SymTable>` field from `Eval` entirely if multi-segment path resolution can be handled differently, or at minimum makes the hot path allocation-free

**The same pattern applies to `call_fn`** (`src/eval.rs:228-248`): `String::from(func)` allocates on every function call. Functions could also be stored by `u64` key instead of `HashMap<String, Ptr<Fn>>`.

**Verification:** All existing tests should continue to pass. No new tests needed — this is a pure performance refactor. Confirm with `cargo test`.
