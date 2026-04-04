use std::error::Error;
use std::fmt::Display;

use crate::ast::Span;
use crate::token::TokenK;

// ── Parser errors ──────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum ParserErrorK {
    Unknown,
    Unexpected,
    UnexpectedToken { expected: TokenK, found: TokenK, span: Span },
    InvalidBinOp { span: Span },
    InvalidInfixOp { span: Span },
    InvalidLValue { span: Span },
    InvalidPrefixOp { span: Span },
    ExpectedExpression { context: &'static str, span: Span },
    ExpectedPath { context: &'static str, span: Span },
}

#[derive(Debug, Clone)]
pub struct ParserError {
    pub errs: Vec<ParserErrorK>,
}

impl ParserError {
    pub fn unknown() -> Self {
        Self {
            errs: vec![ParserErrorK::Unknown],
        }
    }

    pub fn single(kind: ParserErrorK) -> Self {
        Self {
            errs: vec![kind],
        }
    }
}

impl Display for ParserError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for err in &self.errs {
            writeln!(f, "{err:?}")?;
        }
        Ok(())
    }
}

impl std::error::Error for ParserError {}

// ── Runtime / compile errors ───────────────────────────────────

#[derive(Debug, Clone)]
pub enum RuntimeErrorK {
    // Type errors (from value operations)
    TypeMismatch { op: &'static str, lhs: &'static str, rhs: &'static str },
    UnaryTypeMismatch { op: &'static str, operand: &'static str },

    // Eval errors
    UndeclaredVariable { name: String },
    UndeclaredFunction { name: String },
    InvalidAssignmentTarget,
    MultipleDeclarations { name: String },
    NonBooleanCondition { found: &'static str },
    InvalidCallTarget,
    LiteralParseFailure { kind: &'static str, text: String },

    // Emit errors
    InvalidAssignmentLhs,
    FnBodyNotBlock,
    CallTargetNotPath,
    SymbolParseFailure { kind: &'static str, text: String },

    // IR errors
    PatchMismatch { expected: String, found: String },

    // Internal
    Internal { message: String },
}

impl Display for RuntimeErrorK {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TypeMismatch { op, lhs, rhs } =>
                write!(f, "type mismatch: cannot {op} `{lhs}` and `{rhs}`"),
            Self::UnaryTypeMismatch { op, operand } =>
                write!(f, "type mismatch: cannot apply {op} to `{operand}`"),
            Self::UndeclaredVariable { name } =>
                write!(f, "undeclared variable `{name}`"),
            Self::UndeclaredFunction { name } =>
                write!(f, "undeclared function `{name}`"),
            Self::InvalidAssignmentTarget =>
                write!(f, "invalid assignment target"),
            Self::MultipleDeclarations { name } =>
                write!(f, "multiple declarations of `{name}`"),
            Self::NonBooleanCondition { found } =>
                write!(f, "expected boolean condition, found `{found}`"),
            Self::InvalidCallTarget =>
                write!(f, "function call target is not a path"),
            Self::LiteralParseFailure { kind, text } =>
                write!(f, "failed to parse `{text}` as {kind}"),
            Self::InvalidAssignmentLhs =>
                write!(f, "assignment target is not a path"),
            Self::FnBodyNotBlock =>
                write!(f, "function body is not a block"),
            Self::CallTargetNotPath =>
                write!(f, "call target is not a path"),
            Self::SymbolParseFailure { kind, text } =>
                write!(f, "failed to parse symbol `{text}` as {kind}"),
            Self::PatchMismatch { expected, found } =>
                write!(f, "codegen: expected {expected} but found {found} during patch"),
            Self::Internal { message } =>
                write!(f, "internal error: {message}"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RuntimeError {
    pub kind: RuntimeErrorK,
    pub span: Span,
}

impl RuntimeError {
    pub fn new(kind: RuntimeErrorK, span: Span) -> Self {
        Self { kind, span }
    }
}

impl Display for RuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "error: {} [{}..{}]", self.kind, self.span.bgn, self.span.end)
    }
}

impl Error for RuntimeError {}

/// Format an error with source-location context (line:col).
pub fn format_error(err: &RuntimeError, source: &str) -> String {
    let (line, col) = offset_to_line_col(source, err.span.bgn);
    format!("error: {}\n  --> {}:{}", err.kind, line, col)
}

fn offset_to_line_col(source: &str, offset: usize) -> (usize, usize) {
    let mut line = 1;
    let mut col = 1;
    for (i, ch) in source.char_indices() {
        if i >= offset {
            break;
        }
        if ch == '\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
    }
    (line, col)
}
