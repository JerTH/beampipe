# Beampipe Roadmap

Work items derived from a code review of the full codebase. Ordered by dependency
and complexity — earlier items unblock later ones.

---

## Tier 1 — Quick Wins & Foundations

### 1. Verify Fibonacci / Factorial Integration Tests

The `fibonacci` and `factorial` tests (`tests/lang.rs:785–824`) only use features
that already work: `while` loops, `let` bindings, assignment, arithmetic, and
comparison. They are marked `#[ignore]` but likely pass today. Run them, and if
they pass, remove the `#[ignore]` for free test coverage.

### 2. Comments in Source Code

**Goal:** Allow `//` line comments in source programs.

- Lexer-only change: skip from `//` to end-of-line in `src/lex.rs` (around the
  whitespace-skipping logic)
- No parser, eval, or emit changes needed
- Un-ignore tests: `comments::line_comment`, `comments::comment_after_expression`

### 3. Compound Assignment (`+=`, `-=`, `*=`, `/=`)

**Goal:** Implement compound assignment operators.

- The AST node `ExprK::AssignOp` already exists
- Fill in `todo!()` in `src/eval.rs:177` and `src/emit.rs:118`
- Add `+=`, `-=`, `*=`, `/=` tokens to `src/token.rs` and `src/lex.rs`
- Wire up parsing in `src/parse.rs` (similar to existing `OpEq` handling)
- Un-ignore 4 tests in `compound_assignment` module

### 4. Reduce Per-Lookup Allocation in Eval — DONE

**Completed in commit `af8e27b`.**

All `String` allocations were eliminated from the evaluator's hot paths:

- `resolve_path` was removed entirely — replaced by a free function `path_key()` that
  returns the `Sym` key directly for single-segment paths (zero allocation) and
  FNV-combines segment keys for multi-segment paths
- `call_fn` now looks up functions by `u64` key instead of `String`
  (`HashMap<String, Ptr<Fn>>` → `HashMap<u64, Ptr<Fn>>`)
- `Binding::name` (dead field — written but never read) was removed, simplifying
  `declare()` to `(key: u64, value: Value)`
- `symt: RefCell<SymTable>` was removed from `Eval` — no longer needed
- Error-path strings are constructed lazily via `path_to_string()` only when an error
  actually occurs

### 5. Replace `panic!` with Error Propagation in SymTable

**Goal:** Remove the `panic!("internal: poisoned symtable lock")` at `src/ast.rs:478`.

- Return a `Result` instead, using the existing `RuntimeErrorK::Internal` variant
- Callers already propagate `Result` through most of the pipeline
- Small scope, improves robustness

---

## Tier 2 — Core Language Features

### 6. Return Statements

**Goal:** Support early `return <expr>` from functions.

- `KeyReturn` is already lexed (`src/lex.rs:127`) but never parsed
- Add `ExprK::Return(Ptr<Expr>)` to `src/ast.rs`
- Parse `return <expr>` in `parse_null_denotation` (add a `TokenK::KeyReturn` arm
  in `src/parse.rs`)
- Implement in eval via `Result`-based unwinding (e.g. a `ControlFlow::Return(Value)`
  variant that unwinds to the enclosing function call)
- Handle in `src/emit.rs` as well
- Un-ignore 2 tests in `return_statement` module

### 7. For Loops (`for i in 0..N`)

**Goal:** Implement range-based for loops.

- `LoopK::For` AST variant already exists — fill in `todo!()` in `src/eval.rs:261`
  and `src/emit.rs:192`
- Add `..` (range) token to `src/token.rs` and `src/lex.rs`
- Parse `for <ident> in <expr>..<expr> { body }` in `src/parse.rs`
- May need to adjust the `LoopK::For` AST structure to encode the iterator variable,
  start, and end values
- Un-ignore 3 tests in `for_in_range` + 2 integration tests
  (`nested_function_with_loop`, `conditional_in_loop`)

### 8. Labeled / Infinite Loops (`loop {}`, `break`, `continue`)

**Goal:** Implement `loop { }` blocks with `break` and `continue`.

- Fill in `todo!()` in `src/eval.rs:274` and `src/emit.rs:219`
- Add `break` and `continue` keywords to `src/token.rs` and `src/lex.rs`
- Add `ExprK::Break` and `ExprK::Continue` to `src/ast.rs`
- Implement control flow using the same `Result`-based unwinding pattern as return
  statements (item 6)

### 9. String Literals

**Goal:** Support `"hello"` string values.

- Changes span the full pipeline:
  - Lex `"..."` with escape sequences in `src/lex.rs`
  - Add `LitK::Str` to `src/ast.rs`
  - Add `Value::String(String)` to `src/value.rs` (with `Display`, potentially
    `Add` for concatenation)
  - Handle in `src/eval.rs`, `src/emit.rs`, `src/ir.rs`
- Un-ignore 2 tests in `string_literals` module

### 10. Print / Println Builtins

**Goal:** Add built-in `println()` for output.

- Add a builtin function dispatch in `call_fn` (`src/eval.rs:281–302`) — check the
  function name against a builtin table before the user-defined function `HashMap`
  lookup
- Works with any `Value` via `Display`, but benefits from string literals (item 9)
- Un-ignore 2 tests in `print_builtin` module

---

## Tier 3 — Complex Features & Longer-term

### 11. Arrays and Indexing

**Goal:** Support array literals `[1, 2, 3]` and index expressions `arr[0]`.

- Parse `[expr, expr, ...]` (brackets are already tokenized)
- Parse `expr[expr]` as a postfix operator
- Add `Value::Array(Vec<Value>)` to `src/value.rs`
- Add `ExprK::Index` or similar to `src/ast.rs`
- Implement in eval and emit
- Un-ignore 3 tests in `arrays` module

### 12. Structs and Field Access

**Goal:** Support struct definitions, construction, and `expr.field` access.

- Flesh out the empty `ItemK` enum (`src/ast.rs:384`) and `ExprK::Item`
  (`src/eval.rs:102`, `src/emit.rs:54`)
- Parse `struct Name { field: Type, ... }` definitions
- Parse `Name { field: expr, ... }` construction
- Parse `expr.field` access (dot already has binding power 80)
- Add `Value::Struct` to `src/value.rs`
- Un-ignore 2 tests in `structs` module

### 13. Type Annotation Validation

**Goal:** Enforce the type annotations that are already parsed.

- Type annotations on `let` bindings, function params, and return types are parsed
  but never checked
- Would require a semantic analysis pass — likely a new `src/check.rs` module
- Lower priority since the language works without it

### 14. IR Virtual Machine

**Goal:** Execute the IR that `emit.rs` already generates.

- `src/emit.rs` produces `IrCode` with stack-based instructions, but nothing runs it
- Build a stack-based VM (new `src/vm.rs`) to complete the compilation pipeline
- Wire into `src/main.rs` alongside the existing `--ir` / `--bytecode` flags
- Depends on all IR emit paths being implemented first (items 3, 7, 8, etc.)

---

## Key Files

| File | Role |
|------|------|
| `src/eval.rs` | 3 `todo!()` macros remaining |
| `src/emit.rs` | 4 `todo!()` macros |
| `src/lex.rs` | Needs changes for comments, strings, compound assignment, new keywords |
| `src/parse.rs` | Needs changes for return, for-loops, arrays, structs |
| `src/ast.rs` | Panic fix, new `ExprK` variants, `ItemK` fleshing out |
| `src/value.rs` | New `Value` variants (String, Array, Struct) |
| `src/token.rs` | New token kinds |
| `tests/lang.rs` | 24 ignored tests to progressively un-ignore |
