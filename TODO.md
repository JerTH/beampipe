# Next Steps

Three focused work items derived from a code review of the variable scoping implementation (commit `c0abb5f`). Each is self-contained and can be tackled independently.

---

## 1. While Loops

**Goal:** Implement `while` loop evaluation and IR emission.

**What exists:**
- `LoopK::While(cond, body)` is already parsed and represented in the AST (`src/ast.rs`)
- `ExprK::Loop` is matched in both `eval.rs:212` and `emit.rs:150`, with `todo!()` stubs
- Three TDD tests are ready in `tests/lang.rs` under `mod while_loops` (currently `#[ignore]`): `simple_while`, `while_sum`, `while_never_enters`

**Eval implementation (`src/eval.rs`):**
- In the `LoopK::While` arm, loop: evaluate `cond`, break if not `Value::Bool(true)`, otherwise evaluate `body`
- The body is a `Block`, so scoping is already handled by the `ExprK::Block` arm
- Return `Value::None` (Rust `while` loops evaluate to `()`)

**IR implementation (`src/emit.rs`):**
- Record the loop head position, emit the condition, emit `JumpFalse` with `Marker::Temporary`, emit the body, emit `Jump` back to head, patch the `JumpFalse` to point past the loop
- Pattern already exists in the `ExprK::If` arm (lines 115-137)

**Tests:** Un-ignore the three `while_loops` tests. Consider adding: nested while, while with reassignment (`let i = 0; while i < 10 { i = i + 1; } i`), while-false-never-enters.

---

## 2. Equality and Logical Operators

**Goal:** Implement `==`, `!=`, `&&`, `||`, `!` (unary not).

**What exists:**
- TDD tests in `tests/lang.rs`: `mod equality` (4 tests), `mod logical_operators` (4 tests), `mod prefix_operators` (4 tests, includes `!`) — all `#[ignore]`
- `BinOpK` enum in `src/ast.rs` currently has: `Add`, `Sub`, `Div`, `Mul`, `CmpLess`, `CmpGreater`
- The tokenizer (`src/token.rs`) needs to be checked for `==`, `!=`, `&&`, `||` token kinds

**Changes needed:**
- **Tokenizer (`src/lex.rs`):** Add token recognition for `==`, `!=`, `&&`, `||` if not already present
- **Token kinds (`src/token.rs`):** Add `OpEqEq`, `OpBangEq`, `OpAnd`, `OpOr` (or whatever naming convention is used), assign binding powers
- **AST (`src/ast.rs`):** Add `BinOpK::Eq`, `BinOpK::Neq`, `BinOpK::And`, `BinOpK::Or`. For `!`, add a unary op representation or a dedicated `ExprK` variant
- **Parser (`src/parse.rs`):** Handle new infix operators in `parse_left_denotation`. Handle `!` as a prefix operator in `parse_prefix_op` (currently returns an error at line 443)
- **Eval (`src/eval.rs`):** Add match arms for the new `BinOpK` variants
- **Value (`src/value.rs`):** `Value` already has `Bool`, so `eq`/`neq` can compare any two equal-typed values, and `and`/`or`/`not` operate on `Bool`
- **IR (`src/ir.rs`, `src/emit.rs`):** Add `Ir::Eq`, `Ir::Neq`, `Ir::And`, `Ir::Or`, `Ir::Not` and emit them

**Tests:** Un-ignore all tests in `equality`, `logical_operators`, and the `logical_not_*` / `double_negation` tests in `prefix_operators`.

---

## 3. Reduce Per-Lookup Allocation in Eval

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
