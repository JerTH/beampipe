/// Comprehensive test suite for beampipe language features.
///
/// Tests are organized into two categories:
/// - Regression tests: verify currently working features survive refactoring
/// - TDD tests (#[ignore]): define intended behavior for features not yet implemented
///
/// Beampipe targets a strict subset of Rust syntax for simple scripts:
/// no borrow checker, lifetimes, async, macros, derives, enums, match,
/// closures, modules, or traits.

use beampipe::parse::Parse;
use beampipe::eval::Eval;
use beampipe::value::Value;

fn eval_source(source: &str) -> Value {
    let expr = Parse::parse(source).expect("parse failed");
    Eval::eval(&expr)
}

fn parse_and_eval_ok(source: &str) {
    let expr = Parse::parse(source).expect("parse failed");
    let _ = Eval::eval(&expr);
}

// ============================================================
// Regression tests — currently working features
// ============================================================

mod arithmetic {
    use super::*;

    #[test]
    fn integer_addition() {
        assert_eq!(eval_source("fn main() { 2 + 3 }"), Value::Int(5));
    }

    #[test]
    fn integer_subtraction() {
        assert_eq!(eval_source("fn main() { 10 - 3 }"), Value::Int(7));
    }

    #[test]
    fn integer_multiplication() {
        assert_eq!(eval_source("fn main() { 4 * 5 }"), Value::Int(20));
    }

    #[test]
    fn integer_division() {
        assert_eq!(eval_source("fn main() { 10 / 2 }"), Value::Int(5));
    }

    #[test]
    fn float_addition() {
        assert_eq!(eval_source("fn main() { 1.5 + 2.5 }"), Value::Float(4.0));
    }

    #[test]
    fn float_subtraction() {
        assert_eq!(eval_source("fn main() { 5.0 - 1.5 }"), Value::Float(3.5));
    }

    #[test]
    fn float_multiplication() {
        assert_eq!(eval_source("fn main() { 2.0 * 3.0 }"), Value::Float(6.0));
    }

    #[test]
    fn float_division() {
        assert_eq!(eval_source("fn main() { 9.0 / 3.0 }"), Value::Float(3.0));
    }

    #[test]
    fn operator_precedence_mul_before_add() {
        assert_eq!(eval_source("fn main() { 2 + 3 * 4 }"), Value::Int(14));
    }

    #[test]
    fn operator_precedence_div_before_sub() {
        assert_eq!(eval_source("fn main() { 10 - 6 / 2 }"), Value::Int(7));
    }

    #[test]
    fn chained_operations() {
        assert_eq!(eval_source("fn main() { 1 + 2 + 3 + 4 }"), Value::Int(10));
    }

    #[test]
    fn mixed_precedence() {
        assert_eq!(eval_source("fn main() { 2 * 3 + 4 * 5 }"), Value::Int(26));
    }
}

mod literals {
    use super::*;

    #[test]
    fn boolean_true() {
        assert_eq!(eval_source("fn main() { true }"), Value::Bool(true));
    }

    #[test]
    fn boolean_false() {
        assert_eq!(eval_source("fn main() { false }"), Value::Bool(false));
    }

    #[test]
    fn integer_zero() {
        assert_eq!(eval_source("fn main() { 0 }"), Value::Int(0));
    }

    #[test]
    fn large_integer() {
        assert_eq!(eval_source("fn main() { 1000000 }"), Value::Int(1000000));
    }

    #[test]
    fn float_zero() {
        assert_eq!(eval_source("fn main() { 0.0 }"), Value::Float(0.0));
    }
}

mod comparison {
    use super::*;

    #[test]
    fn greater_than_true() {
        assert_eq!(eval_source("fn main() { 5 > 3 }"), Value::Bool(true));
    }

    #[test]
    fn greater_than_false() {
        assert_eq!(eval_source("fn main() { 3 > 5 }"), Value::Bool(false));
    }

    #[test]
    fn less_than_true() {
        assert_eq!(eval_source("fn main() { 3 < 5 }"), Value::Bool(true));
    }

    #[test]
    fn less_than_false() {
        assert_eq!(eval_source("fn main() { 5 < 3 }"), Value::Bool(false));
    }
}

mod blocks {
    use super::*;

    #[test]
    fn block_returns_last_expression() {
        assert_eq!(eval_source("fn main() { { 1; 2; 3 } }"), Value::Int(3));
    }

    #[test]
    fn nested_blocks() {
        assert_eq!(eval_source("fn main() { { { 42 } } }"), Value::Int(42));
    }

    #[test]
    fn empty_block() {
        assert_eq!(eval_source("fn main() { { } }"), Value::None);
    }

    #[test]
    fn semicolon_discards_value() {
        assert_eq!(eval_source("fn main() { 42; }"), Value::None);
    }

    #[test]
    fn block_with_arithmetic() {
        assert_eq!(eval_source("fn main() { { 1 + 2; 3 + 4 } }"), Value::Int(7));
    }
}

mod variables {
    use super::*;

    #[test]
    fn assign_and_read() {
        assert_eq!(eval_source("fn main() { let x = 10; x + 5 }"), Value::Int(15));
    }

    #[test]
    fn reassignment() {
        assert_eq!(eval_source("fn main() { let x = 1; x = 2; x }"), Value::Int(2));
    }

    #[test]
    fn multiple_variables() {
        assert_eq!(
            eval_source("fn main() { let a = 3; let b = 7; a + b }"),
            Value::Int(10)
        );
    }

    #[test]
    fn variable_in_expression() {
        assert_eq!(
            eval_source("fn main() { let x = 5; x * x + 1 }"),
            Value::Int(26)
        );
    }
}

mod functions {
    use super::*;

    #[test]
    fn simple_function_call() {
        assert_eq!(
            eval_source("fn add(a: i32, b: i32) { a + b } fn main() { add(3, 4) }"),
            Value::Int(7)
        );
    }

    #[test]
    fn function_with_body() {
        assert_eq!(
            eval_source("fn square(x: i32) { x * x } fn main() { square(5) }"),
            Value::Int(25)
        );
    }

    #[test]
    fn nested_function_calls() {
        assert_eq!(
            eval_source("fn double(x: i32) { x * 2 } fn main() { double(double(3)) }"),
            Value::Int(12)
        );
    }

    #[test]
    fn function_multiple_statements() {
        assert_eq!(
            eval_source(
                "fn compute(a: i32, b: i32) { let c = a + b; c * 2 } fn main() { compute(3, 4) }"
            ),
            Value::Int(14)
        );
    }

    #[test]
    fn function_no_params() {
        assert_eq!(
            eval_source("fn forty_two() { 42 } fn main() { forty_two() }"),
            Value::Int(42)
        );
    }
}

mod if_else {
    use super::*;

    #[test]
    fn if_true_branch() {
        assert_eq!(
            eval_source("fn main() { if true { 1 } else { 2 } }"),
            Value::Int(1)
        );
    }

    #[test]
    fn if_false_branch() {
        assert_eq!(
            eval_source("fn main() { if false { 1 } else { 2 } }"),
            Value::Int(2)
        );
    }

    #[test]
    fn if_with_comparison() {
        assert_eq!(
            eval_source("fn main() { if 3 > 2 { 10 } else { 20 } }"),
            Value::Int(10)
        );
    }

    #[test]
    fn nested_if() {
        assert_eq!(
            eval_source("fn main() { if true { if false { 1 } else { 2 } } else { 3 } }"),
            Value::Int(2)
        );
    }

    #[test]
    fn if_without_else() {
        // if without else and condition true should return the body value
        assert_eq!(
            eval_source("fn main() { if true { 42 } }"),
            Value::Int(42)
        );
    }

    #[test]
    fn if_in_function() {
        assert_eq!(
            eval_source(
                "fn max(a: i32, b: i32) { if a > b { a } else { b } } fn main() { max(3, 7) }"
            ),
            Value::Int(7)
        );
    }
}

mod existing_examples {
    use super::*;

    #[test]
    fn expr_example_parses_and_evals() {
        parse_and_eval_ok(include_str!("../src/example/expr.rs"));
    }

    #[test]
    fn foo_example_parses_and_evals() {
        parse_and_eval_ok(include_str!("../src/example/foo.rs"));
    }

    #[test]
    fn nest_example_parses_and_evals() {
        parse_and_eval_ok(include_str!("../src/example/nest.rs"));
    }
}

// ============================================================
// TDD tests — intended features, not yet implemented
// These will fail now but define the target behavior.
// ============================================================

mod let_bindings {
    use super::*;

    #[test]
    fn simple_let() {
        assert_eq!(eval_source("fn main() { let x = 5; x }"), Value::Int(5));
    }

    #[test]
    fn multiple_let_bindings() {
        assert_eq!(
            eval_source("fn main() { let x = 1; let y = 2; x + y }"),
            Value::Int(3)
        );
    }

    #[test]
    fn let_then_reassign() {
        // all let bindings are mutable — reassignment should work
        assert_eq!(
            eval_source("fn main() { let x = 5; x = 10; x }"),
            Value::Int(10)
        );
    }

    #[test]
    fn let_with_expression() {
        assert_eq!(
            eval_source("fn main() { let x = 2 + 3; x * 2 }"),
            Value::Int(10)
        );
    }

    #[test]
    fn let_with_type_annotation() {
        assert_eq!(
            eval_source("fn main() { let x: i32 = 42; x }"),
            Value::Int(42)
        );
    }
}

mod scoping {
    use super::*;

    #[test]
    fn inner_block_does_not_leak() {
        let result = std::panic::catch_unwind(|| {
            eval_source("fn main() { { let x = 5; } x }")
        });
        assert!(result.is_err());
    }

    #[test]
    fn outer_variable_visible_in_inner_block() {
        assert_eq!(
            eval_source("fn main() { let x = 10; { x + 5 } }"),
            Value::Int(15)
        );
    }

    #[test]
    fn shadowing_in_inner_block() {
        assert_eq!(
            eval_source("fn main() { let x = 1; { let x = 2; x } }"),
            Value::Int(2)
        );
    }

    #[test]
    fn shadow_restored_after_block() {
        assert_eq!(
            eval_source("fn main() { let x = 1; { let x = 99; } x }"),
            Value::Int(1)
        );
    }

    #[test]
    fn reassign_outer_from_inner_block() {
        assert_eq!(
            eval_source("fn main() { let x = 1; { x = 42; } x }"),
            Value::Int(42)
        );
    }

    #[test]
    fn function_params_scoped() {
        assert_eq!(
            eval_source("fn foo(x: i32) { x + 1 } fn main() { let y = foo(5); y }"),
            Value::Int(6)
        );
    }

    #[test]
    fn sibling_blocks_independent() {
        assert_eq!(
            eval_source("fn main() { let r = 0; { let x = 10; r = x; } { let x = 20; r = r + x; } r }"),
            Value::Int(30)
        );
    }

    #[test]
    fn same_scope_shadowing() {
        assert_eq!(
            eval_source("fn main() { let x = 1; let x = 2; x }"),
            Value::Int(2)
        );
    }
}

mod equality {
    use super::*;

    #[test]
    #[ignore]
    fn equal_integers() {
        assert_eq!(eval_source("fn main() { 5 == 5 }"), Value::Bool(true));
    }

    #[test]
    #[ignore]
    fn unequal_integers() {
        assert_eq!(eval_source("fn main() { 5 == 3 }"), Value::Bool(false));
    }

    #[test]
    #[ignore]
    fn equal_booleans() {
        assert_eq!(
            eval_source("fn main() { true == true }"),
            Value::Bool(true)
        );
    }

    #[test]
    #[ignore]
    fn not_equal() {
        assert_eq!(eval_source("fn main() { 5 != 3 }"), Value::Bool(true));
    }
}

mod logical_operators {
    use super::*;

    #[test]
    #[ignore]
    fn and_true() {
        assert_eq!(
            eval_source("fn main() { true && true }"),
            Value::Bool(true)
        );
    }

    #[test]
    #[ignore]
    fn and_false() {
        assert_eq!(
            eval_source("fn main() { true && false }"),
            Value::Bool(false)
        );
    }

    #[test]
    #[ignore]
    fn or_true() {
        assert_eq!(
            eval_source("fn main() { false || true }"),
            Value::Bool(true)
        );
    }

    #[test]
    #[ignore]
    fn or_false() {
        assert_eq!(
            eval_source("fn main() { false || false }"),
            Value::Bool(false)
        );
    }
}

mod prefix_operators {
    use super::*;

    #[test]
    #[ignore]
    fn unary_negation_int() {
        assert_eq!(eval_source("fn main() { -5 }"), Value::Int(-5));
    }

    #[test]
    #[ignore]
    fn logical_not_true() {
        assert_eq!(eval_source("fn main() { !true }"), Value::Bool(false));
    }

    #[test]
    #[ignore]
    fn logical_not_false() {
        assert_eq!(eval_source("fn main() { !false }"), Value::Bool(true));
    }

    #[test]
    #[ignore]
    fn double_negation() {
        assert_eq!(eval_source("fn main() { -(-5) }"), Value::Int(5));
    }
}

mod while_loops {
    use super::*;

    #[test]
    fn simple_while() {
        assert_eq!(
            eval_source("fn main() { let x = 0; while x < 5 { x = x + 1 } x }"),
            Value::Int(5)
        );
    }

    #[test]
    fn while_sum() {
        assert_eq!(
            eval_source(
                "fn main() { let sum = 0; let i = 1; while i < 11 { sum = sum + i; i = i + 1 } sum }"
            ),
            Value::Int(55)
        );
    }

    #[test]
    fn while_never_enters() {
        assert_eq!(
            eval_source("fn main() { let x = 10; while x < 0 { x = x + 1 } x }"),
            Value::Int(10)
        );
    }
}

mod for_in_range {
    use super::*;

    #[test]
    #[ignore]
    fn for_range_sum() {
        assert_eq!(
            eval_source("fn main() { let sum = 0; for i in 0..5 { sum = sum + i } sum }"),
            Value::Int(10)
        );
    }

    #[test]
    #[ignore]
    fn for_range_product() {
        assert_eq!(
            eval_source(
                "fn main() { let product = 1; for i in 1..6 { product = product * i } product }"
            ),
            Value::Int(120)
        );
    }

    #[test]
    #[ignore]
    fn for_empty_range() {
        assert_eq!(
            eval_source("fn main() { let x = 42; for i in 5..5 { x = 0 } x }"),
            Value::Int(42)
        );
    }
}

mod return_statement {
    use super::*;

    #[test]
    #[ignore]
    fn early_return() {
        assert_eq!(
            eval_source("fn foo() { return 42; 0 } fn main() { foo() }"),
            Value::Int(42)
        );
    }

    #[test]
    #[ignore]
    fn return_from_conditional() {
        assert_eq!(
            eval_source(
                "fn abs(x: i32) -> i32 { if x < 0 { return 0 - x } x } fn main() { abs(5) }"
            ),
            Value::Int(5)
        );
    }
}

mod compound_assignment {
    use super::*;

    #[test]
    #[ignore]
    fn plus_equals() {
        assert_eq!(
            eval_source("fn main() { let x = 5; x += 3; x }"),
            Value::Int(8)
        );
    }

    #[test]
    #[ignore]
    fn minus_equals() {
        assert_eq!(
            eval_source("fn main() { let x = 10; x -= 2; x }"),
            Value::Int(8)
        );
    }

    #[test]
    #[ignore]
    fn times_equals() {
        assert_eq!(
            eval_source("fn main() { let x = 4; x *= 3; x }"),
            Value::Int(12)
        );
    }

    #[test]
    #[ignore]
    fn divide_equals() {
        assert_eq!(
            eval_source("fn main() { let x = 20; x /= 4; x }"),
            Value::Int(5)
        );
    }
}

mod string_literals {
    use super::*;

    #[test]
    #[ignore]
    fn simple_string() {
        // Value::String doesn't exist yet — this tests the full pipeline
        let result = eval_source("fn main() { \"hello\" }");
        assert_eq!(format!("{}", result), "hello");
    }

    #[test]
    #[ignore]
    fn empty_string() {
        let result = eval_source("fn main() { \"\" }");
        assert_eq!(format!("{}", result), "");
    }
}

mod arrays {
    use super::*;

    #[test]
    #[ignore]
    fn array_literal() {
        // Array indexing should return the element
        assert_eq!(
            eval_source("fn main() { let a = [1, 2, 3]; a[1] }"),
            Value::Int(2)
        );
    }

    #[test]
    #[ignore]
    fn array_first_element() {
        assert_eq!(
            eval_source("fn main() { let a = [10, 20, 30]; a[0] }"),
            Value::Int(10)
        );
    }

    #[test]
    #[ignore]
    fn array_in_loop() {
        assert_eq!(
            eval_source(
                "fn main() { let a = [1, 2, 3, 4, 5]; let sum = 0; for i in 0..5 { sum = sum + a[i] } sum }"
            ),
            Value::Int(15)
        );
    }
}

mod structs {
    use super::*;

    #[test]
    #[ignore]
    fn struct_definition_and_field_access() {
        assert_eq!(
            eval_source(
                "struct Point { x: i32, y: i32 } fn main() { let p = Point { x: 1, y: 2 }; p.x + p.y }"
            ),
            Value::Int(3)
        );
    }

    #[test]
    #[ignore]
    fn struct_field_reassignment() {
        assert_eq!(
            eval_source(
                "struct Point { x: i32, y: i32 } fn main() { let p = Point { x: 1, y: 2 }; p.x = 10; p.x }"
            ),
            Value::Int(10)
        );
    }
}

mod print_builtin {
    use super::*;

    #[test]
    #[ignore]
    fn println_integer() {
        // Should evaluate without panic — println is a side-effect
        parse_and_eval_ok("fn main() { println(42) }");
    }

    #[test]
    #[ignore]
    fn println_expression() {
        parse_and_eval_ok("fn main() { println(2 + 3) }");
    }
}

mod comments {
    use super::*;

    #[test]
    #[ignore]
    fn line_comment() {
        assert_eq!(
            eval_source("fn main() {\n// this is a comment\n42 }"),
            Value::Int(42)
        );
    }

    #[test]
    #[ignore]
    fn comment_after_expression() {
        assert_eq!(
            eval_source("fn main() { 42 // trailing comment\n}"),
            Value::Int(42)
        );
    }
}

// ============================================================
// Integration tests — full programs combining multiple features
// ============================================================

mod integration {
    use super::*;

    #[test]
    #[ignore]
    fn fibonacci() {
        assert_eq!(
            eval_source(
                "fn fib(n: i32) -> i32 {
                    let a = 0;
                    let b = 1;
                    let i = 0;
                    while i < n {
                        let temp = b;
                        b = a + b;
                        a = temp;
                        i = i + 1
                    }
                    a
                }
                fn main() { fib(10) }"
            ),
            Value::Int(55)
        );
    }

    #[test]
    #[ignore]
    fn factorial() {
        assert_eq!(
            eval_source(
                "fn factorial(n: i32) -> i32 {
                    let result = 1;
                    let i = 1;
                    while i < n + 1 {
                        result = result * i;
                        i = i + 1
                    }
                    result
                }
                fn main() { factorial(5) }"
            ),
            Value::Int(120)
        );
    }

    #[test]
    #[ignore]
    fn nested_function_with_loop() {
        assert_eq!(
            eval_source(
                "fn sum_to(n: i32) -> i32 {
                    let total = 0;
                    for i in 0..n {
                        total = total + i
                    }
                    total
                }
                fn main() { sum_to(5) + sum_to(3) }"
            ),
            Value::Int(13)
        );
    }

    #[test]
    #[ignore]
    fn conditional_in_loop() {
        assert_eq!(
            eval_source(
                "fn count_even(n: i32) -> i32 {
                    let count = 0;
                    for i in 0..n {
                        if i / 2 * 2 == i {
                            count = count + 1
                        }
                    }
                    count
                }
                fn main() { count_even(10) }"
            ),
            Value::Int(5)
        );
    }
}
